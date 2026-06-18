from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import onnx


def model_io(onnx_path: str | Path) -> tuple[list[str], list[str], list[list[int]]]:
    model = onnx.load(str(onnx_path))
    initializer_names = {init.name for init in model.graph.initializer}

    input_names: list[str] = []
    input_shapes: list[list[int]] = []
    for value in model.graph.input:
        if value.name in initializer_names:
            continue
        input_names.append(value.name)
        shape = []
        for dim in value.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else 1)
        input_shapes.append(shape)

    output_names = [value.name for value in model.graph.output]
    return input_names, output_names, input_shapes


def generate_raw_inputs(
    onnx_path: str | Path,
    output_dir: str | Path,
    model_name: str,
    samples: int = 5,
    device_relative: bool = False,
    mode: Literal["zeros", "calibration"] = "zeros",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_names, _, input_shapes = model_io(onnx_path)
    input_list = output_dir / "input_list.txt"

    lines: list[str] = []
    for sample_idx in range(samples):
        entries: list[str] = []
        for name, shape in zip(input_names, input_shapes):
            safe_name = name.replace("/", "_").replace("::", "_")
            file_name = f"{safe_name}_{sample_idx}.raw" if samples > 1 else f"{safe_name}.raw"
            raw_path = output_dir / file_name
            data = _make_input_data(name, shape, sample_idx, mode)
            data.tofile(str(raw_path))
            rhs = f"models/{model_name}/{file_name}" if device_relative else str(raw_path)
            entries.append(f"{name}:={rhs}")
        lines.append(" ".join(entries))

    input_list.write_text("\n".join(lines) + "\n")
    return input_list


def _make_input_data(
    name: str,
    shape: list[int],
    sample_idx: int,
    mode: Literal["zeros", "calibration"],
) -> np.ndarray:
    if mode == "zeros":
        return np.zeros(shape, dtype=np.float32)

    lower_name = name.lower()
    if "state" in lower_name or "cache" in lower_name:
        return np.zeros(shape, dtype=np.float32)

    rng = np.random.default_rng(20260517 + sample_idx)
    if _looks_like_image_input(lower_name, shape):
        return rng.random(shape).astype(np.float32)

    return rng.normal(loc=0.0, scale=0.02, size=shape).astype(np.float32)


def _looks_like_image_input(lower_name: str, shape: list[int]) -> bool:
    if "image" in lower_name or "pixel" in lower_name or lower_name in {"input", "x"}:
        return True
    return len(shape) >= 4 and 3 in shape[-3:]
