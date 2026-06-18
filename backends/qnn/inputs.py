from __future__ import annotations

from pathlib import Path

from common.onnx_io import generate_raw_inputs


def ensure_profile_inputs(model_dir: Path, model_name: str, onnx_path: str | None) -> Path | None:
    input_list = model_dir / "input_list.txt"
    if (
        input_list.exists()
        and list(model_dir.glob("*.raw"))
        and _uses_device_relative_paths(input_list)
    ):
        return input_list
    if not onnx_path:
        candidate = model_dir / f"{model_name}.onnx"
        onnx_path = str(candidate) if candidate.exists() else None
    if not onnx_path:
        return None
    return generate_raw_inputs(
        onnx_path,
        model_dir,
        model_name=model_name,
        samples=5,
        device_relative=True,
    )


def _uses_device_relative_paths(input_list: Path) -> bool:
    for line in input_list.read_text().splitlines():
        for entry in line.split():
            if ":=" not in entry:
                continue
            _, rhs = entry.split(":=", 1)
            if rhs.startswith("/"):
                return False
    return True
