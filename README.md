# On-Device Foundation Models

Reproducible scripts for exporting, rewriting, converting, and profiling
foundation models on mobile NPUs.

This repository is script-only by design. It does not redistribute upstream
checkpoints, converted ONNX models, QNN `.so` libraries, or QNN context
binaries. Users download upstream checkpoints through the official source and
generate deployment artifacts locally.

## Current Models

| Model | Task | Variants | Backend | Status |
|---|---|---|---|---|
| Depth Anything 3 | monocular relative depth | small, base, large | QNN FP16 | ready |

## Repository Layout

```text
models/              User-facing model cards and metadata.
converters/          Official PyTorch repo/checkpoint to ONNX export code.
graph_rewriters/     ONNX graph rewrite passes.
backends/            Backend conversion and profiling wrappers.
common/              Small shared utilities.
benchmarks/          Public summary tables, not raw converted artifacts.
```

Local generated directories are ignored by git:

```text
third_party/         Official GitHub repositories cloned locally.
checkpoints/         Official checkpoints downloaded locally.
exported_models/     ONNX, QNN, and profiling artifacts generated locally.
```

## Quickstart: Depth Anything 3 Small

Install Python dependencies in your own environment, then install the official
Depth Anything 3 requirements from the cloned upstream repository.

```bash
python pipeline.py --model depth-anything-3 --variant small --stages fetch-source fetch-weights export optimize convert --size 504
```

Detailed QNN optrace profiling on a connected Android target:

```bash
python pipeline.py --model depth-anything-3 --variant small --stages profile --input exported_models/qnn/depth-anything-3/da3_small_1x3x504x504_qkv_nr_hs_hl_lsfold_featclean_rank3/da3_small_1x3x504x504_qkv_nr_hs_hl_lsfold_featclean_rank3.so --device s26 --serial R5KL2088TBN --vtcm 8 --iterations 10
```

See [models/depth-anything-3/README.md](models/depth-anything-3/README.md)
for S/B/L commands and profiling numbers.

## Artifact Policy

Do not commit or upload generated model artifacts. In particular, exclude:

- upstream checkpoint snapshots
- ONNX files containing model weights
- QNN `.so` model libraries
- QNN context `.bin` files
- raw calibration/input/output tensors

The generated directories and file patterns are blocked by `.gitignore`.
