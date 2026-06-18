from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from backends.qnn.device import DEVICE_TO_SERIAL
from backends.qnn.htp_config import write_backend_configs
from backends.qnn.inputs import ensure_profile_inputs
from backends.qnn.tools import (
    find_qairt_root,
    find_qnn_lib,
    find_qnn_profile_config,
    find_qnn_tool,
)
from common import logging as log


def run_adb(args: list[str], serial: str | None = None) -> tuple[bool, str]:
    cmd = ["adb"]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as exc:
        return False, str(exc)


def get_device_serial(device: str, provided_serial: str | None = None) -> str | None:
    if provided_serial:
        return provided_serial
    mapped = DEVICE_TO_SERIAL.get(device)
    if mapped:
        return mapped

    ok, output = run_adb(["devices"])
    if not ok:
        return None
    devices = []
    for line in output.strip().splitlines()[1:]:
        if "\tdevice" in line:
            devices.append(line.split("\t")[0])
    return devices[0] if len(devices) == 1 else None


def profile_qnn(
    model_dir: str | Path,
    device: str,
    serial: str | None = None,
    vtcm_sizes: list[int] | None = None,
    iterations: int = 10,
    onnx_path: str | None = None,
    sync_runtime: bool = False,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    model_name = model_dir.name
    vtcm_sizes = vtcm_sizes or [8]

    log.header(f"QNN Profile: {model_name}")
    device_serial = get_device_serial(device, serial)
    if not device_serial:
        raise RuntimeError("No ADB device serial resolved. Pass --serial explicitly.")

    ok, output = run_adb(["get-state"], device_serial)
    if not ok:
        raise RuntimeError(f"Device is not connected: {output}")

    ensure_profile_inputs(model_dir, model_name, onnx_path)

    results: dict[str, Any] = {
        "device": device,
        "serial": device_serial,
        "model": model_name,
        "iterations": iterations,
        "vtcm_results": {},
    }

    for vtcm in vtcm_sizes:
        log.stage("profile", f"Running VTCM={vtcm}MB")
        write_backend_configs(model_dir, model_name, device, vtcm)
        _ensure_runtime(device_serial, force=sync_runtime)
        _push_model(model_dir, model_name, device_serial)
        profile_result = _run_inference(model_name, model_dir, device, device_serial, vtcm)
        results["vtcm_results"][str(vtcm)] = profile_result

    valid = {
        int(k): v["latency_ms"]
        for k, v in results["vtcm_results"].items()
        if v.get("latency_ms", 0) > 0
    }
    if valid:
        best_vtcm = min(valid, key=valid.get)
        results["best_vtcm"] = best_vtcm
        results["best_latency_ms"] = valid[best_vtcm]

    output_path = model_dir / f"profile_{device}.json"
    output_path.write_text(json.dumps(results, indent=2))
    log.success(f"Profile results: {output_path}")
    return results


def _push_model(model_dir: Path, model_name: str, serial: str) -> None:
    base_dir = "/data/local/tmp/Qnn"
    run_adb(["shell", f"mkdir -p {base_dir}/models"], serial)
    run_adb(["shell", f"rm -rf {base_dir}/models/{model_name}"], serial)
    ok, output = run_adb(["push", str(model_dir), f"{base_dir}/models/"], serial)
    if not ok:
        raise RuntimeError(f"Failed to push model: {output}")


def _ensure_runtime(serial: str, force: bool = False) -> None:
    root = find_qairt_root()
    if not root:
        log.warning("QAIRT root not found; skipping runtime lib deploy")
        return

    if force:
        _sync_runtime_tree(root, serial)
        return

    remote_lib = "/data/local/tmp/Qnn/lib"
    needed = [
        "libQnnHtpNetRunExtensions.so",
        "libQnnHtpProfilingReader.so",
    ]
    for name in needed:
        ok, _ = run_adb(["shell", f"test -f {remote_lib}/{name}"], serial)
        if ok:
            continue
        local = root / "lib" / "aarch64-android" / name
        if not local.exists():
            log.warning(f"Missing local runtime lib: {local}")
            continue
        log.info(f"Pushing QNN runtime lib: {name}")
        ok, output = run_adb(["push", str(local), f"{remote_lib}/"], serial)
        if not ok:
            raise RuntimeError(f"Failed to push {name}: {output}")
        run_adb(["shell", f"chmod 777 {remote_lib}/{name}"], serial)


def _sync_runtime_tree(root: Path, serial: str) -> None:
    log.info(f"Syncing QAIRT runtime from {root}")
    run_adb(["shell", "mkdir -p /data/local/tmp/Qnn/lib"], serial)

    bin_dir = root / "bin" / "aarch64-android"
    for name in ("qnn-context-binary-generator", "qnn-net-run", "qnn-profile-viewer"):
        local = bin_dir / name
        if local.exists():
            ok, output = run_adb(["push", str(local), "/data/local/tmp/Qnn/"], serial)
            if not ok:
                raise RuntimeError(f"Failed to push {name}: {output}")
            run_adb(["shell", f"chmod 777 /data/local/tmp/Qnn/{name}"], serial)

    lib_dir = root / "lib" / "aarch64-android"
    for local in sorted(lib_dir.glob("libQnn*.so")):
        ok, output = run_adb(["push", str(local), "/data/local/tmp/Qnn/lib/"], serial)
        if not ok:
            raise RuntimeError(f"Failed to push {local.name}: {output}")

    for skel_dir in sorted((root / "lib").glob("hexagon-v*/unsigned")):
        for local in sorted(skel_dir.glob("libQnn*.so")):
            ok, output = run_adb(["push", str(local), "/data/local/tmp/Qnn/lib/"], serial)
            if not ok:
                raise RuntimeError(f"Failed to push {local.name}: {output}")
    run_adb(["shell", "chmod 777 /data/local/tmp/Qnn/lib/libQnn*.so"], serial)


def _run_inference(
    model_name: str,
    model_dir: Path,
    device: str,
    serial: str,
    vtcm_mb: int,
) -> dict[str, Any]:
    backend_config = f"backend_extension_config_{device}_vtcm_{vtcm_mb}.json"
    result: dict[str, Any] = {}

    context_reused = _run_optrace_profile(model_name, model_dir, device, serial, vtcm_mb, backend_config)
    result["context_binary_reused"] = context_reused

    chrometrace = model_dir / f"chrometrace_{device}_vtcm_{vtcm_mb}.json"
    if chrometrace.exists():
        result["chrometrace"] = str(chrometrace)
    optrace_log = model_dir / f"qnn-optrace_{device}_vtcm_{vtcm_mb}.log"
    if optrace_log.exists():
        result["optrace_log"] = str(optrace_log)
    schematic = model_dir / f"{model_name}_schematic.bin"
    if schematic.exists():
        result["schematic"] = str(schematic)
    _, ctx_path, ctx_meta_path = _context_cache_paths(model_name, model_dir, device, vtcm_mb)
    if ctx_path.exists():
        result["context_binary"] = str(ctx_path)
    if ctx_meta_path.exists():
        result["context_binary_metadata"] = str(ctx_meta_path)
    return result


def _cache_metadata(
    model_name: str,
    model_dir: Path,
    device: str,
    vtcm_mb: int,
    backend_config: str,
) -> dict[str, Any]:
    root = find_qairt_root()
    so_path = model_dir / f"{model_name}.so"
    backend_config_path = model_dir / backend_config
    htp_config_path = model_dir / f"htp_config_{device}_vtcm_{vtcm_mb}.json"
    return {
        "model": model_name,
        "device": device,
        "vtcm_mb": vtcm_mb,
        "profiling_level": "detailed",
        "profiling_option": "optrace",
        "qairt_root": str(root) if root else None,
        "qairt_version": root.name if root else None,
        "model_so": _file_fingerprint(so_path),
        "backend_config": _file_fingerprint(backend_config_path),
        "htp_config": _file_fingerprint(htp_config_path),
    }


def _file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def _context_cache_paths(model_name: str, model_dir: Path, device: str, vtcm_mb: int) -> tuple[str, Path, Path]:
    cache_stem = f"{model_name}_ctx_{device}_vtcm_{vtcm_mb}_optrace"
    return (
        cache_stem,
        model_dir / f"{cache_stem}.bin",
        model_dir / f"{cache_stem}.json",
    )


def _has_valid_context_cache(
    model_name: str,
    model_dir: Path,
    device: str,
    vtcm_mb: int,
    backend_config: str,
) -> bool:
    _, ctx_path, meta_path = _context_cache_paths(model_name, model_dir, device, vtcm_mb)
    if not ctx_path.exists() or not meta_path.exists():
        return False
    try:
        old_meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    return old_meta == _cache_metadata(model_name, model_dir, device, vtcm_mb, backend_config)


def _write_context_cache_metadata(
    model_name: str,
    model_dir: Path,
    device: str,
    vtcm_mb: int,
    backend_config: str,
) -> None:
    _, _, meta_path = _context_cache_paths(model_name, model_dir, device, vtcm_mb)
    metadata = _cache_metadata(model_name, model_dir, device, vtcm_mb, backend_config)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")


def _run_optrace_profile(
    model_name: str,
    model_dir: Path,
    device: str,
    serial: str,
    vtcm_mb: int,
    backend_config: str,
) -> bool:
    base_dir = "/data/local/tmp/Qnn"
    log.info("Running mandatory detailed optrace profiling")
    cache_stem, ctx_path, _ = _context_cache_paths(model_name, model_dir, device, vtcm_mb)
    remote_ctx = f"./models/{model_name}/{cache_stem}.bin"

    if _has_valid_context_cache(model_name, model_dir, device, vtcm_mb, backend_config):
        log.success(f"Reusing cached QNN context binary: {ctx_path}")
        context_reused = True
    else:
        log.info(f"Generating QNN context binary cache: {ctx_path}")
        ctx_cmd = f"""
        cd {base_dir} &&
        export ADSP_LIBRARY_PATH={base_dir}/lib &&
        export LD_LIBRARY_PATH={base_dir}/lib &&
        ./qnn-context-binary-generator \
          --profiling_level detailed \
          --profiling_option optrace \
          --backend lib/libQnnHtp.so \
          --model models/{model_name}/{model_name}.so \
          --output_dir ./models/{model_name} \
          --binary_file {cache_stem} \
          --config_file ./models/{model_name}/{backend_config}
        """
        ok, output = run_adb(["shell", ctx_cmd], serial)
        if not ok:
            raise RuntimeError(f"qnn-context-binary-generator (optrace) failed: {output}")
        ok, output = run_adb(
            ["pull", f"{base_dir}/models/{model_name}/{cache_stem}.bin", str(ctx_path)],
            serial,
        )
        if not ok:
            raise RuntimeError(f"Failed to pull context binary cache: {output}")
        _write_context_cache_metadata(model_name, model_dir, device, vtcm_mb, backend_config)
        context_reused = False

    run_cmd = f"""
    cd {base_dir} &&
    export ADSP_LIBRARY_PATH={base_dir}/lib &&
    export LD_LIBRARY_PATH={base_dir}/lib &&
    ./qnn-net-run \
      --profiling_level detailed \
      --profiling_option optrace \
      --output_data_type float_and_native \
      --retrieve_context {remote_ctx} \
      --backend lib/libQnnHtp.so \
      --input_list ./models/{model_name}/input_list.txt \
      --output_dir ./models/{model_name} \
      --log_level verbose \
      --config_file ./models/{model_name}/{backend_config}
    """
    ok, output = run_adb(["shell", run_cmd], serial)
    if not ok:
        raise RuntimeError(f"qnn-net-run (optrace) failed: {output}")

    _pull_optrace_artifacts(model_name, model_dir, device, serial, vtcm_mb)
    _generate_chrometrace(model_name, model_dir, device, vtcm_mb)
    return context_reused


def _pull_optrace_artifacts(
    model_name: str,
    model_dir: Path,
    device: str,
    serial: str,
    vtcm_mb: int,
) -> None:
    base_dir = "/data/local/tmp/Qnn"
    schematic = model_dir / f"{model_name}_schematic.bin"
    schematic_candidates = [
        f"{model_name}_schematic.bin",
        f"{model_name.replace('.', '_')}_schematic.bin",
        f"{re.sub(r'[^0-9A-Za-z_]+', '_', model_name)}_schematic.bin",
    ]
    seen = set()
    pulled_schematic = False
    for candidate in schematic_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ok, _ = run_adb(["pull", f"{base_dir}/{candidate}", str(schematic)], serial)
        if ok:
            pulled_schematic = True
            break
    if not pulled_schematic:
        log.warning("Failed to pull optrace schematic")

    find_log_cmd = (
        f"ls -1 {base_dir}/models/{model_name}/qnn-profiling-data_*.log "
        "2>/dev/null | sort -V | tail -1"
    )
    ok, latest = run_adb(["shell", find_log_cmd], serial)
    latest = latest.strip()
    if latest:
        optrace_log = model_dir / f"qnn-optrace_{device}_vtcm_{vtcm_mb}.log"
        ok, _ = run_adb(["pull", latest, str(optrace_log)], serial)
        if not ok:
            log.warning("Failed to pull optrace profiling log")
    else:
        log.warning("Optrace profiling log not found on device")


def _generate_chrometrace(model_name: str, model_dir: Path, device: str, vtcm_mb: int) -> None:
    schematic = model_dir / f"{model_name}_schematic.bin"
    optrace_log = model_dir / f"qnn-optrace_{device}_vtcm_{vtcm_mb}.log"
    output = model_dir / f"chrometrace_{device}_vtcm_{vtcm_mb}.json"

    missing = [str(p) for p in (schematic, optrace_log) if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing optrace artifacts for chrometrace: {missing}")

    viewer = find_qnn_tool("qnn-profile-viewer")
    reader = find_qnn_lib("libQnnHtpOptraceProfilingReader.so")
    config = find_qnn_profile_config()
    if not reader:
        raise RuntimeError("Could not find libQnnHtpOptraceProfilingReader.so")

    cmd = [
        viewer,
        "--reader",
        reader,
        "--input_log",
        str(optrace_log),
        "--schematic",
        str(schematic),
        "--output",
        str(output),
    ]
    if config:
        cmd[1:1] = ["--config", config]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"qnn-profile-viewer optrace failed: {proc.stderr or proc.stdout}")
    if not output.exists():
        raise RuntimeError(f"Chrometrace was not generated: {output}")
    log.success(f"Chrometrace generated: {output}")
