from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from common import logging as log
from graph_rewriters.simplify import simplify_and_infer


VARIANT_TO_HF_REPO = {
    "small": "depth-anything/DA3-SMALL",
    "base": "depth-anything/DA3-BASE",
    "large": "depth-anything/DA3-LARGE-1.1",
}

VARIANT_TO_MODEL_NAME = {
    "small": "da3-small",
    "base": "da3-base",
    "large": "da3-large",
}


def model_name(variant: str, batch: int, size: int) -> str:
    if variant not in VARIANT_TO_MODEL_NAME:
        raise ValueError(f"Unsupported DA3 variant: {variant}. Use one of: {', '.join(VARIANT_TO_MODEL_NAME)}")
    return f"da3_{variant}_{batch}x3x{size}x{size}"


def fetch_source(source_dir: Path, revision: str | None = None) -> Path:
    if source_dir.exists():
        log.success(f"DA3 source already exists: {source_dir}")
        return source_dir
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "https://github.com/DepthAnything/Depth-Anything-3", str(source_dir)]
    log.stage("fetch-source", "Cloning official Depth Anything 3 repository")
    subprocess.run(cmd, check=True)
    if revision:
        subprocess.run(["git", "-C", str(source_dir), "checkout", revision], check=True)
    return source_dir


def fetch_weights(variant: str, download_dir: Path, model_repo: str | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to fetch DA3 checkpoints: pip install huggingface_hub") from exc
    repo_id = model_repo or VARIANT_TO_HF_REPO[variant]
    target = download_dir / "depth-anything-3" / variant
    log.stage("fetch-weights", f"Downloading official DA3 snapshot from {repo_id}")
    path = Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.yaml", "*.md", "*.txt"],
        )
    )
    log.success(f"Fetched DA3 weights: {path}")
    return path


def export_onnx(
    *,
    variant: str,
    source_dir: Path,
    weights_dir: Path | None,
    output_dir: Path,
    size: int = 504,
    batch: int = 1,
    opset: int = 19,
    outputs: tuple[str, ...] = ("depth",),
    simplify: bool = True,
    simplify_skip_large_tensor: bool = False,
) -> Path:
    if size != 504:
        raise ValueError("Depth Anything 3 uses the official 504x504 export resolution in this repo.")

    import torch
    import torch.nn as nn

    _add_source_to_path(source_dir)
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise RuntimeError(
            f"Depth Anything 3 is not importable from {source_dir}. "
            "Run fetch-source and install upstream dependencies first."
        ) from exc

    class DA3ExportWrapper(nn.Module):
        """Export DA3 as a single-frame model with a 4D BCHW input."""

        def __init__(self, model: nn.Module, output_names: tuple[str, ...]):
            super().__init__()
            self.model = model
            self.output_names = output_names

        def forward(self, image: torch.Tensor):
            image = image.unsqueeze(1)
            result = self.model(image, None, None, [], False, False, "middle")
            values = []
            for name in self.output_names:
                if name not in result:
                    raise RuntimeError(f"DA3 output '{name}' was not produced")
                values.append(result[name])
            return tuple(values)

    name = model_name(variant, batch, size)
    output_path = output_dir / "onnx" / "depth-anything-3" / f"{name}.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hf_or_local = str(weights_dir) if weights_dir else VARIANT_TO_HF_REPO[variant]

    log.stage("export", f"Exporting DA3 {variant} to {output_path}")
    _patch_position_getter_for_onnx()
    model = DepthAnything3.from_pretrained(hf_or_local)
    core_model = model.model if hasattr(model, "model") else model
    wrapper = DA3ExportWrapper(core_model, outputs).to(device="cpu").eval()
    dummy = torch.zeros(batch, 3, size, size, device="cpu")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(output_path),
            opset_version=opset,
            input_names=["image"],
            output_names=list(outputs),
            dynamic_axes=None,
        )

    simplify_and_infer(output_path, enabled=simplify, skip_large_tensor=simplify_skip_large_tensor)
    log.success(f"Exported ONNX: {output_path}")
    return output_path


def _patch_position_getter_for_onnx() -> None:
    """Patch official DA3 runtime code paths that are equivalent but ONNX-safe."""
    import torch

    try:
        from depth_anything_3.model.dinov2.layers import rope
    except ImportError:
        return

    if getattr(rope.PositionGetter, "_ondevice_fm_onnx_patched", False):
        return

    def position_getter_call(self, batch_size: int, height: int, width: int, device: torch.device):
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
            positions = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=-1)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

    rope.PositionGetter.__call__ = position_getter_call
    rope.PositionGetter._ondevice_fm_onnx_patched = True


def _add_source_to_path(source_dir: Path) -> None:
    src = source_dir / "src"
    for path in (src, source_dir):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
