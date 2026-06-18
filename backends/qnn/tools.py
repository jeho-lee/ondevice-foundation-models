from __future__ import annotations

import os
import shutil
from pathlib import Path


def find_qairt_root() -> Path | None:
    for key in ("QAIRT_SDK_ROOT", "QNN_SDK_ROOT"):
        override = os.environ.get(key)
        if not override:
            continue
        root = Path(override)
        if (root / "bin" / "x86_64-linux-clang").exists():
            return root
    return None


def find_qnn_tool(name: str) -> str:
    qairt_root = find_qairt_root()
    if qairt_root:
        candidate = qairt_root / "bin" / "x86_64-linux-clang" / name
        if candidate.exists():
            return str(candidate)

    in_path = shutil.which(name)
    if in_path:
        return in_path

    return name


def find_qnn_lib(name: str, arch: str = "x86_64-linux-clang") -> str | None:
    root = find_qairt_root()
    if root:
        candidate = root / "lib" / arch / name
        if candidate.exists():
            return str(candidate)
    return None


def find_qnn_profile_config() -> str | None:
    candidates: list[Path] = []
    root = find_qairt_root()
    if root:
        candidates.extend(root.glob("**/config.json"))
    for candidate in candidates:
        if candidate.exists() and "profiler" in str(candidate).lower():
            return str(candidate)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def qairt_env() -> dict[str, str]:
    env = os.environ.copy()
    root = find_qairt_root()
    if not root:
        return env

    env["QAIRT_SDK_ROOT"] = str(root)
    env["QNN_SDK_ROOT"] = str(root)

    prepend = {
        "PATH": root / "bin" / "x86_64-linux-clang",
        "PYTHONPATH": root / "lib" / "python",
        "LD_LIBRARY_PATH": root / "lib" / "x86_64-linux-clang",
    }
    for key, path in prepend.items():
        old = env.get(key)
        env[key] = f"{path}:{old}" if old else str(path)

    benchmarks = root / "benchmarks" / "QNN"
    if benchmarks.exists():
        env["PYTHONPATH"] = f"{benchmarks}:{env['PYTHONPATH']}"
    return env
