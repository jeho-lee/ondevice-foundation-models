# Depth Anything 3

Depth Anything 3 (DA3) is included as a script-only deployment recipe. This
repo does not redistribute DA3 checkpoints or converted ONNX/QNN artifacts.

## Supported Variants

| Variant | Upstream checkpoint |
|---|---|
| small | `depth-anything/DA3-SMALL` |
| base | `depth-anything/DA3-BASE` |
| large | `depth-anything/DA3-LARGE-1.1` |

## End-to-End Pipeline

Run the full S/B/L pipeline one variant at a time:

```bash
python pipeline.py --model depth-anything-3 --variant small --stages fetch-source fetch-weights export optimize convert --size 504
python pipeline.py --model depth-anything-3 --variant base --stages fetch-source fetch-weights export optimize convert --size 504
python pipeline.py --model depth-anything-3 --variant large --stages fetch-source fetch-weights export optimize convert --size 504
```

Profile an already converted QNN library:

```bash
python pipeline.py --model depth-anything-3 --variant small --stages profile --input exported_models/qnn/depth-anything-3/da3_small_1x3x504x504_qkv_nr_hs_hl_lsfold_featclean_rank3/da3_small_1x3x504x504_qkv_nr_hs_hl_lsfold_featclean_rank3.so --device s26 --serial R5KL2088TBN --vtcm 8 --iterations 10
```

## Graph Rewrite Bundle

The default optimized graph suffix is:

```text
qkv_nr_hs_hl_lsfold_featclean_rank3
```

The pass bundle is:

```text
cleanup,pre_qkv_explode_no_rope,head_split,head_linear,layer_scale_fold,feature_concat_slice_cleanup,feature_rank3_cleanup
```

## Current QNN FP16 Results

| Variant | Resolution | Latency |
|---|---:|---:|
| small | 504x504 | 25.465 ms |
| base | 504x504 | 61.094 ms |
| large | 504x504 | 181.561 ms |

These values are QNN HTP optrace `Time (us)` values measured on an S26-class
target with VTCM 8 MB.
