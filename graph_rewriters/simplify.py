from __future__ import annotations

from pathlib import Path

import onnx

from common import logging as log


def simplify_and_infer(
    path: str | Path,
    *,
    enabled: bool = True,
    skip_large_tensor: bool = False,
) -> bool:
    """Run onnx-simplifier and shape inference in-place.

    Returns True when simplification succeeded. Shape inference is attempted even
    when simplification is disabled or fails.
    """
    onnx_path = Path(path)
    simplified_ok = False
    model = onnx.load(str(onnx_path))

    if enabled:
        try:
            from onnxsim import simplify

            log.info(f"Simplifying ONNX: {onnx_path.name}")
            kwargs = {"skip_large_tensor": skip_large_tensor}
            simplified, check = simplify(model, **kwargs)
            if check:
                model = simplified
                simplified_ok = True
                log.success(f"ONNX simplify passed: {onnx_path.name}")
            else:
                log.warning(f"ONNX simplify check failed; keeping original graph: {onnx_path.name}")
        except TypeError:
            try:
                simplified, check = simplify(model)
                if check:
                    model = simplified
                    simplified_ok = True
                    log.success(f"ONNX simplify passed: {onnx_path.name}")
                else:
                    log.warning(f"ONNX simplify check failed; keeping original graph: {onnx_path.name}")
            except Exception as exc:
                log.warning(f"ONNX simplify skipped for {onnx_path.name}: {exc}")
        except Exception as exc:
            log.warning(f"ONNX simplify skipped for {onnx_path.name}: {exc}")

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        log.warning(f"ONNX shape inference skipped for {onnx_path.name}: {exc}")

    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
    return simplified_ok
