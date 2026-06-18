from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from common import logging as log
from graph_rewriters.depth_anything_3 import (
    FeatureConcatSliceCleanupPass,
    FeatureRank3CleanupPass,
    HeadwiseLinearPass,
    LayerScaleFoldPass,
    MhaHeadwiseSplitPass,
    PreQkvExplosionNoRopePass,
)


DA3_DEFAULT_PASSES = [
    "cleanup",
    "pre_qkv_explode_no_rope",
    "head_split",
    "head_linear",
    "layer_scale_fold",
    "feature_concat_slice_cleanup",
    "feature_rank3_cleanup",
]


PASS_REGISTRY = {
    "pre_qkv_explode_no_rope": PreQkvExplosionNoRopePass,
    "head_split": MhaHeadwiseSplitPass,
    "head_linear": HeadwiseLinearPass,
    "layer_scale_fold": LayerScaleFoldPass,
    "feature_concat_slice_cleanup": FeatureConcatSliceCleanupPass,
    "feature_rank3_cleanup": FeatureRank3CleanupPass,
}


def optimize_onnx(
    input_model: str | Path,
    output_model: str | Path | None = None,
    passes: list[str] | None = None,
    verify: bool = True,
) -> str:
    passes = passes or DA3_DEFAULT_PASSES
    input_path = Path(input_model)
    output_path = Path(output_model) if output_model else _output_path(input_path, passes)

    model = onnx.load(str(input_path))
    initial_stats = _stats(model)
    log.stage("optimize", f"{input_path.name}: passes={','.join(passes)}")

    for pass_name in passes:
        if pass_name == "cleanup":
            _remove_unused_initializers(model)
            continue
        pass_cls = PASS_REGISTRY.get(pass_name)
        if pass_cls is None:
            raise ValueError(f"Unsupported DA3 graph rewrite pass: {pass_name}")
        model = pass_cls().run(model)
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as exc:
            log.warning(f"Shape inference warning after {pass_name}: {exc}")

    _remove_unused_initializers(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        log.warning(f"ONNX checker warning for {output_path.name}: {exc}")
    onnx.save(model, str(output_path))

    final_stats = _stats(model)
    log.info(
        f"ONNX nodes: {initial_stats['nodes']} -> {final_stats['nodes']} "
        f"initializers: {initial_stats['initializers']} -> {final_stats['initializers']}"
    )
    if verify:
        _verify_equivalence(input_path, output_path)
    log.success(f"Optimized ONNX: {output_path}")
    return str(output_path)


def _output_path(input_path: Path, passes: list[str]) -> Path:
    suffix_map = {
        "cleanup": "",
        "pre_qkv_explode_no_rope": "qkv_nr",
        "head_split": "hs",
        "head_linear": "hl",
        "layer_scale_fold": "lsfold",
        "feature_concat_slice_cleanup": "featclean",
        "feature_rank3_cleanup": "rank3",
    }
    suffix_parts = [suffix_map[p] for p in passes if suffix_map.get(p)]
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else "_clean"
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def _stats(model: onnx.ModelProto) -> dict[str, Any]:
    ops: dict[str, int] = defaultdict(int)
    for node in model.graph.node:
        ops[node.op_type] += 1
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "ops": dict(ops),
    }


def _verify_equivalence(original_path: Path, optimized_path: Path) -> None:
    try:
        sess_orig = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
        sess_opt = ort.InferenceSession(str(optimized_path), providers=["CPUExecutionProvider"])
        feed = {meta.name: _dummy_input(meta) for meta in sess_orig.get_inputs()}
        orig_outputs = sess_orig.run(None, feed)
        opt_outputs = sess_opt.run(None, feed)
    except Exception as exc:
        raise RuntimeError(f"ONNXRuntime equivalence check failed for {optimized_path.name}: {exc}") from exc

    if len(orig_outputs) != len(opt_outputs):
        raise RuntimeError(f"Output count mismatch after optimization: {len(orig_outputs)} vs {len(opt_outputs)}")
    for idx, (orig, opt) in enumerate(zip(orig_outputs, opt_outputs)):
        if orig.shape != opt.shape:
            raise RuntimeError(f"Output {idx} shape mismatch after optimization: {orig.shape} vs {opt.shape}")
        max_abs = float(np.max(np.abs(orig - opt))) if orig.size else 0.0
        mean_abs = float(np.mean(np.abs(orig - opt))) if orig.size else 0.0
        if not np.allclose(orig, opt, rtol=1e-3, atol=1e-3):
            raise RuntimeError(
                f"Output {idx} mismatch after optimization: max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}"
            )
        log.info(f"ORT equivalence output {idx}: max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}")


def _dummy_input(meta: Any) -> np.ndarray:
    shape = [1 if not isinstance(dim, int) or dim <= 0 else dim for dim in meta.shape]
    if meta.type == "tensor(float16)":
        return np.random.default_rng(0).standard_normal(shape).astype(np.float16)
    if meta.type == "tensor(float)":
        return np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    if meta.type == "tensor(double)":
        return np.random.default_rng(0).standard_normal(shape).astype(np.float64)
    if meta.type == "tensor(int64)":
        return np.zeros(shape, dtype=np.int64)
    if meta.type == "tensor(int32)":
        return np.zeros(shape, dtype=np.int32)
    if meta.type == "tensor(bool)":
        return np.zeros(shape, dtype=bool)
    return np.zeros(shape, dtype=np.float32)


def _remove_unused_initializers(model: onnx.ModelProto) -> None:
    used: set[str] = set()

    def collect(nodes) -> None:
        for node in nodes:
            used.update(node.input)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    collect(attr.g.node)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for graph in attr.graphs:
                        collect(graph.node)

    collect(model.graph.node)
    keep = [init for init in model.graph.initializer if init.name in used]
    if len(keep) != len(model.graph.initializer):
        model.graph.ClearField("initializer")
        model.graph.initializer.extend(keep)
