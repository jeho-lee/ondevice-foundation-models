#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backends.qnn.convert import convert_to_qnn
from backends.qnn.profile import profile_qnn
from common import logging as log
from converters.depth_anything_3 import (
    VARIANT_TO_HF_REPO,
    export_onnx,
    fetch_source,
    fetch_weights,
    model_name,
)
from graph_rewriters.optimize import DA3_DEFAULT_PASSES, optimize_onnx


SUPPORTED_STAGES = ["fetch-source", "fetch-weights", "export", "optimize", "convert", "profile"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-device foundation model deployment pipeline")
    parser.add_argument("--model", default="depth-anything-3", choices=["depth-anything-3"])
    parser.add_argument("--variant", default="small", choices=["small", "base", "large"])
    parser.add_argument("--stages", nargs="+", default=["export"], choices=SUPPORTED_STAGES)
    parser.add_argument("--size", type=int, default=504, choices=[504])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "exported_models")
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "third_party" / "Depth-Anything-3")
    parser.add_argument("--downloads-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--model-repo", default=None, help="Override Hugging Face repo id for DA3 weights.")
    parser.add_argument("--input", type=Path, default=None, help="Existing ONNX or QNN .so to start from.")
    parser.add_argument("--passes", default="default", help="Comma-separated graph rewrite passes, or default.")
    parser.add_argument("--no-verify", action="store_true", help="Skip ONNXRuntime equivalence check after optimize.")
    parser.add_argument("--no-simplify", action="store_true", help="Skip default ONNX simplification during export.")
    parser.add_argument("--simplify-skip-large-tensor", action="store_true")
    parser.add_argument("--quant-mode", choices=["fp16", "int8", "w8a16"], default="fp16")
    parser.add_argument("--quant-config", default=None)
    parser.add_argument("--no-cleanup", action="store_true", help="Keep QNN conversion workspace.")
    parser.add_argument("--device", default="s26")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--vtcm", type=int, nargs="+", default=[8])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--sync-runtime", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.header("On-Device Foundation Models Pipeline")
    log.info(f"Model: {args.model} ({args.variant})")
    log.info(f"Stages: {' -> '.join(args.stages)}")
    log.info(f"Output: {args.output_dir}")

    current_onnx = args.input if args.input and args.input.suffix == ".onnx" else None
    current_qnn = args.input if args.input and args.input.suffix == ".so" else None
    weights_dir: Path | None = None

    try:
        for stage in args.stages:
            if stage == "fetch-source":
                fetch_source(args.source_dir, args.source_revision)
            elif stage == "fetch-weights":
                weights_dir = fetch_weights(args.variant, args.downloads_dir, args.model_repo)
            elif stage == "export":
                if weights_dir is None:
                    candidate = args.downloads_dir / "depth-anything-3" / args.variant
                    weights_dir = candidate if candidate.exists() else None
                current_onnx = export_onnx(
                    variant=args.variant,
                    source_dir=args.source_dir,
                    weights_dir=weights_dir,
                    output_dir=args.output_dir,
                    size=args.size,
                    batch=args.batch,
                    opset=args.opset,
                    simplify=not args.no_simplify,
                    simplify_skip_large_tensor=args.simplify_skip_large_tensor,
                )
            elif stage == "optimize":
                if current_onnx is None:
                    current_onnx = _default_onnx(args)
                passes = _resolve_passes(args.passes)
                current_onnx = Path(
                    optimize_onnx(
                        current_onnx,
                        passes=passes,
                        verify=not args.no_verify,
                    )
                )
            elif stage == "convert":
                if current_onnx is None:
                    current_onnx = args.input if args.input and args.input.suffix == ".onnx" else _default_optimized_onnx(args)
                current_qnn = Path(
                    convert_to_qnn(
                        input_onnx=str(current_onnx),
                        output_dir=args.output_dir / "qnn" / "depth-anything-3",
                        quant_mode=args.quant_mode,
                        quant_config=args.quant_config,
                        cleanup=not args.no_cleanup,
                    )
                )
            elif stage == "profile":
                if current_qnn is None:
                    current_qnn = args.input if args.input and args.input.suffix == ".so" else _default_qnn_so(args)
                result = profile_qnn(
                    model_dir=current_qnn.parent,
                    device=args.device,
                    serial=args.serial,
                    vtcm_sizes=args.vtcm,
                    iterations=args.iterations,
                    onnx_path=str(current_onnx) if current_onnx else None,
                    sync_runtime=args.sync_runtime,
                )
                _write_profile_manifest(args, current_qnn, result)
            else:
                raise ValueError(stage)

        log.success("Pipeline complete")
        return 0
    except Exception as exc:
        log.error(str(exc))
        return 1


def _resolve_passes(raw: str) -> list[str]:
    passes = [item.strip() for item in raw.split(",") if item.strip()]
    if not passes or passes == ["default"]:
        return DA3_DEFAULT_PASSES
    return passes


def _default_onnx(args: argparse.Namespace) -> Path:
    return args.output_dir / "onnx" / "depth-anything-3" / f"{model_name(args.variant, args.batch, args.size)}.onnx"


def _default_optimized_onnx(args: argparse.Namespace) -> Path:
    base = _default_onnx(args)
    return base.with_name(f"{base.stem}_qkv_nr_hs_hl_lsfold_featclean_rank3.onnx")


def _default_qnn_so(args: argparse.Namespace) -> Path:
    name = _default_optimized_onnx(args).stem
    return args.output_dir / "qnn" / "depth-anything-3" / name / f"{name}.so"


def _write_profile_manifest(args: argparse.Namespace, qnn_so: Path, result: dict) -> None:
    reports = args.output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "variant": args.variant,
        "qnn_so": str(qnn_so),
        "upstream_hf": args.model_repo or VARIANT_TO_HF_REPO[args.variant],
        "profile": result,
    }
    (reports / f"depth-anything-3_{args.variant}_{args.size}_profile.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
