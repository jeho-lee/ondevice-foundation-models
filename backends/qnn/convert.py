from __future__ import annotations

import shutil
from pathlib import Path

import onnx

from common import logging as log
from common.onnx_io import generate_raw_inputs, model_io
from common.subprocess import run_command
from backends.qnn.tools import find_qnn_tool, qairt_env


def convert_to_qnn(
    input_onnx: str,
    output_dir: str | Path,
    quant_mode: str = "fp16",
    quant_config: str | None = None,
    calibration_input_list: str | Path | None = None,
    calibrate_missing_encodings: bool = True,
    model_name_override: str | None = None,
    act_quantizer_calibration: str | None = None,
    param_quantizer_calibration: str | None = None,
    act_quantizer_schema: str | None = None,
    param_quantizer_schema: str | None = None,
    percentile_calibration_value: float | None = None,
    use_per_channel_quantization: bool = True,
    use_per_row_quantization: bool = False,
    enable_per_row_quantized_bias: bool = False,
    quantizer_log: str | Path | None = None,
    quantizer_log_level: str | None = None,
    preserve_io_datatype: bool = False,
    preserve_output_order: bool = True,
    cleanup: bool = True,
) -> str:
    input_onnx_path = Path(input_onnx).resolve()
    if not input_onnx_path.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {input_onnx_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_name_override or input_onnx_path.stem
    if model_name_override:
        pass
    elif quant_mode == "int8" and not model_name.endswith("_int8"):
        model_name = f"{model_name}_int8"
    elif quant_mode == "w8a16" and not model_name.endswith("_w8a16"):
        model_name = f"{model_name}_w8a16"
    elif quant_mode not in {"fp16", "int8", "w8a16"}:
        raise ValueError(f"Unsupported QNN quantization mode: {quant_mode}")

    log.header(f"QNN Conversion: {model_name}")
    log.info(f"Input: {input_onnx_path}")
    log.info(f"Output: {output_dir}")

    workspace = output_dir / "_work" / model_name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    cpp_path = workspace / f"{model_name}.cpp"
    so_out_dir = workspace / "lib"
    so_out_dir.mkdir(parents=True, exist_ok=True)

    input_names, output_names, input_shapes = model_io(input_onnx_path)
    converter_cmd = [
        find_qnn_tool("qnn-onnx-converter"),
        "--input_network",
        str(input_onnx_path),
        "-o",
        str(cpp_path),
        "--preserve_io",
        "layout",
        *input_names,
        *output_names,
        "--float_bitwidth",
        "16",
    ]
    if preserve_output_order:
        # NOTE: the converter's keep_graph_output_order pass crashes
        # (KeyError '<output>_identity') when a graph output is constant-foldable
        # with a single consumer, e.g. SAM3.1's static vision_pos after onnxsim.
        # Callers whose runtimes address outputs by name can pass False.
        converter_cmd.append("--preserve_onnx_output_order")
    if preserve_io_datatype:
        converter_cmd.extend(["--preserve_io", "datatype", *input_names, *output_names])
    if quant_mode == "fp16":
        converter_cmd.append("--use_dynamic_16_bit_weights")

    for name in input_names:
        converter_cmd.extend(["--input_layout", name, "NONTRIVIAL"])

    input_list_path = None
    if quant_mode in {"int8", "w8a16"}:
        should_add_input_list = calibrate_missing_encodings or not quant_config
        if should_add_input_list and calibration_input_list:
            input_list_path = Path(calibration_input_list).resolve()
        elif should_add_input_list:
            input_list_path = generate_raw_inputs(
                input_onnx_path,
                workspace / "calibration_inputs",
                model_name=model_name,
                samples=10,
                device_relative=False,
                mode="calibration",
            )
        else:
            input_list_path = None
        converter_cmd.extend(
            [
                "--act_bitwidth",
                "16" if quant_mode == "w8a16" else "8",
                "--weights_bitwidth",
                "8",
                "--bias_bitwidth",
                "32",
            ]
        )
        if input_list_path:
            converter_cmd.extend(["--input_list", str(input_list_path)])
        if use_per_channel_quantization:
            converter_cmd.append("--use_per_channel_quantization")
        if use_per_row_quantization:
            converter_cmd.append("--use_per_row_quantization")
        if enable_per_row_quantized_bias:
            converter_cmd.append("--enable_per_row_quantized_bias")
        if act_quantizer_calibration:
            converter_cmd.extend(["--act_quantizer_calibration", act_quantizer_calibration])
        if param_quantizer_calibration:
            converter_cmd.extend(["--param_quantizer_calibration", param_quantizer_calibration])
        if act_quantizer_schema:
            converter_cmd.extend(["--act_quantizer_schema", act_quantizer_schema])
        if param_quantizer_schema:
            converter_cmd.extend(["--param_quantizer_schema", param_quantizer_schema])
        if percentile_calibration_value is not None:
            converter_cmd.extend(["--percentile_calibration_value", str(percentile_calibration_value)])
        if quantizer_log:
            converter_cmd.extend(["--quantizer_log", str(quantizer_log)])
            if quantizer_log_level:
                converter_cmd.extend(["--quantizer_log_level", quantizer_log_level])

    if quant_config:
        quant_config_path = Path(quant_config)
        if quant_config_path.exists():
            converter_cmd.extend(["--quantization_overrides", str(quant_config_path)])
        else:
            log.warning(f"Quantization config not found: {quant_config}")

    env = qairt_env()
    ok, output = run_command(converter_cmd, "qnn-onnx-converter", env=env)
    if not ok:
        raise RuntimeError(output)

    generator_cmd = [
        find_qnn_tool("qnn-model-lib-generator"),
        "-c",
        str(cpp_path),
        "-t",
        "aarch64-android",
        "-l",
        f"{model_name}.so",
        "-o",
        str(so_out_dir),
    ]
    bin_path = workspace / f"{model_name}.bin"
    if bin_path.exists():
        generator_cmd.extend(["-b", str(bin_path)])

    ok, output = run_command(generator_cmd, "qnn-model-lib-generator", env=env)
    if not ok:
        raise RuntimeError(output)

    generated_so = so_out_dir / "aarch64-android" / f"lib{model_name}.so"
    if not generated_so.exists():
        generated_so = so_out_dir / "aarch64-android" / f"{model_name}.so"
    if not generated_so.exists():
        raise RuntimeError(f"Generated .so not found under {so_out_dir}")

    model_dir = output_dir / model_name
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    final_so = model_dir / f"{model_name}.so"
    shutil.copy2(generated_so, final_so)

    onnx_copy = model_dir / f"{model_name}.onnx"
    shutil.copy2(input_onnx_path, onnx_copy)
    _copy_external_data_sidecars(input_onnx_path, model_dir)

    if input_list_path and input_list_path.exists():
        shutil.copy2(input_list_path, model_dir / "input_list.txt")
        for raw_file in input_list_path.parent.glob("*.raw"):
            shutil.copy2(raw_file, model_dir / raw_file.name)
        for raw_file in input_list_path.parent.glob("*.f16"):
            shutil.copy2(raw_file, model_dir / raw_file.name)

    if cleanup and workspace.exists():
        shutil.rmtree(workspace)

    log.success(f"QNN library: {final_so}")
    return str(final_so)


def _copy_external_data_sidecars(onnx_path: Path, model_dir: Path) -> None:
    model = onnx.load_model(str(onnx_path), load_external_data=False)
    copied: set[Path] = set()
    for initializer in model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key != "location":
                continue
            sidecar = (onnx_path.parent / entry.value).resolve()
            if sidecar.exists() and sidecar not in copied:
                shutil.copy2(sidecar, model_dir / sidecar.name)
                copied.add(sidecar)
