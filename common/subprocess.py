from __future__ import annotations

import subprocess
import os
from typing import Iterable

from common import logging as log


def run_command(
    cmd: Iterable[str],
    description: str,
    show_output: bool = True,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> tuple[bool, str]:
    cmd_list = [str(part) for part in cmd]
    log.info(f"{description}: {' '.join(cmd_list)}")

    try:
        if show_output:
            proc = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            lines: list[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if _should_print(line):
                    print(f"  {line}")
                lines.append(line)
            return_code = proc.wait(timeout=timeout)
            output = "\n".join(lines)
        else:
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            return_code = result.returncode
            output = result.stdout + result.stderr

        if return_code != 0:
            log.error(f"{description} failed with code {return_code}")
            return False, output

        log.success(f"{description} complete")
        return True, output
    except FileNotFoundError:
        log.error(f"Command not found: {cmd_list[0]}")
        return False, f"Command not found: {cmd_list[0]}"
    except Exception as exc:
        log.error(f"{description} failed: {exc}")
        return False, str(exc)


def _should_print(line: str) -> bool:
    noisy = (
        "aarch64-linux-android-clang++",
        "-fPIC",
        "-std=c++11",
        "-Wc99-designator",
        "is a C99 extension",
        "warning: mixture of designated",
    )
    return bool(line) and not any(item in line for item in noisy)
