from __future__ import annotations

import json
from pathlib import Path

from backends.qnn.device import DEVICE_TO_DSP, DEVICE_TO_SOC_MODEL


def write_backend_configs(model_dir: Path, model_name: str, device: str, vtcm_mb: int) -> Path:
    dsp_arch = DEVICE_TO_DSP.get(device, "v79")

    device_config = {
        "dsp_arch": dsp_arch,
        "profiling_level": "linting",
        "cores": [{"perf_profile": "burst"}],
    }
    soc_model = DEVICE_TO_SOC_MODEL.get(device)
    if soc_model is not None:
        device_config["soc_model"] = soc_model

    htp_config = {
        "graphs": [
            {
                "graph_names": [model_name],
                "vtcm_mb": vtcm_mb,
                "hvx_threads": 0,
                "O": 3,
            }
        ],
        "devices": [device_config],
    }

    htp_path = model_dir / f"htp_config_{device}_vtcm_{vtcm_mb}.json"
    htp_path.write_text(json.dumps(htp_config, indent=2))

    backend_config = {
        "backend_extensions": {
            "shared_library_path": "./lib/libQnnHtpNetRunExtensions.so",
            "config_file_path": f"./models/{model_name}/{htp_path.name}",
        }
    }
    backend_path = model_dir / f"backend_extension_config_{device}_vtcm_{vtcm_mb}.json"
    backend_path.write_text(json.dumps(backend_config, indent=2))
    return backend_path
