# IFF-PCS

Implementation of **Cross-Layer Foundation Representations for Pseudo-Change Suppression in Remote Sensing Change Detection**. A frozen encoder supplies shallow, middle and deep temporal distances. A learned attenuation gate suppresses appearance-sensitive unchanged pixels while a complementary loss preserves annotated deep-feature changes.

## Install and data

Use Python 3.10+, activate a virtual environment, and install:

```bash
python -m venv .venv
python -m pip install -e .
```

The first model construction downloads DINOv2 ViT-S/14 through the [DINOv2 model repository](https://github.com/facebookresearch/dinov2). Keep the Torch Hub weight cache for later use. For an offline machine, populate that cache before training.

Obtain LEVIR-CD, WHU-CD or SYSU-CD from their maintainers. Provide co-registered RGB pairs and binary change masks in a JSON manifest. All paths are relative to the manifest:

```json
[
  {"before": "train/A/001.png", "after": "train/B/001.png", "mask": "train/label/001.png", "split": "train", "group": "scene001", "dataset": "LEVIR-CD"},
  {"before": "val/A/002.png", "after": "val/B/002.png", "mask": "val/label/002.png", "split": "dev", "group": "scene002", "dataset": "LEVIR-CD"},
  {"before": "test/A/003.png", "after": "test/B/003.png", "mask": "test/label/003.png", "split": "test", "group": "scene003", "dataset": "LEVIR-CD"}
]
```

Use complete official image partitions. Set `group` to the original scene identity when extracting tiles so related tiles cannot cross partitions. Positive mask pixels indicate physical change.

```bash
iffpcs prepare --manifest data/manifest.json --output data/prepared
iffpcs train --data data/prepared --output runs/train
iffpcs evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

Preparation resizes pairs and nearest-neighbor masks to `--size` (518 by default, divisible by the 14-pixel patch size). It defines fixed appearance-sensitive proxy masks using per-image standardized RGB discrepancy on unchanged pixels, excluding a two-pixel change-boundary band. `--percentile 0.85` controls that proxy. `--small-area 64` defines the small-change connected-component panel at evaluation resolution. Freeze these definitions before comparisons; proxies do not depend on a detector's predictions.

## Baselines, ablations and stress

```bash
iffpcs train --mode rgb --no-gate --output runs/rgb
iffpcs train --mode deep --no-gate --output runs/deep
iffpcs train --mode middle --no-gate --output runs/middle
iffpcs train --no-gate --output runs/concatenation
iffpcs train --no-pseudo --no-preserve --output runs/gate_only
iffpcs train --no-preserve --output runs/no_preserve
iffpcs train --no-boundary --output runs/no_boundary
iffpcs train --fraction 0.1 --output runs/labels10
iffpcs evaluate --permute-gate --output runs/permuted
iffpcs evaluate --stress brightness --output runs/brightness
iffpcs evaluate --stress noise --output runs/noise
iffpcs evaluate --stress translation --output runs/translation
```

Evaluate fitted baselines with their own `--checkpoint`. Layer-source controls change the decoder input; concatenation retains all layers without attenuation. Gate permutation uses a fixed circular spatial permutation, retaining its value histogram. Stress affects only the later image; the stored proxy and reference masks stay fixed. Translation inserts a five-pixel horizontal offset without circular wraparound. Noise has standard deviation 0.03 and a fixed seed; brightness multiplies RGB by 0.8.

Training flags include `--epochs`, `--batch-size`, `--lr`, `--seed`, `--pseudo-weight`, `--preserve-weight`, and `--device`. `--fraction` selects a seed-fixed fraction of training pairs for low-label analysis. The encoder remains frozen; only the selected projections, gate and decoder are trainable. A development F1 sweep over thresholds 0.2–0.6 selects and stores one operating threshold.

## Metrics and files

Training writes `model.pt`. Evaluation writes binary masks, `metrics.json`, detailed `pairs.json` and `summary.png`.

`metrics.json` reports per-dataset F1/IoU from pooled confusion counts, mean per-image boundary F1, proxy-region false-alarm rate, small-change recall, observed throughput, trainable parameter count and the frozen threshold. Ratios are fractions. Boundary matching uses a two-pixel tolerance. PFR divides false alarms by the fixed proxy support; small-change recall uses only the separately declared small-component support.

`pairs.json` retains TP/FP/FN and all targeted denominators, gate AUC between proxy pixels and the deep-change preservation set, mean gate values on both sets, and mean shallow/middle/deep distance profiles for appearance-only, small-change and ordinary unchanged strata. Empty diagnostic strata yield `null`. Preserve these per-pair outputs for common-panel comparisons or seed summaries; do not combine the proxy and global confusion denominators.

## Implementation choices

DINOv2 blocks 2, 5 and 11 supply the three frozen representations. Cosine distances drive a two-layer pointwise sigmoid gate; `(1-g)` multiplies projected signed feature differences before decoding. The decoder uses compact convolutional fusion. The default layer selection and decoder widths make the manuscript's unspecified encoder/decoder choices concrete; parameter counts are computed from this model.

The pseudo loss raises suppression on unchanged high-discrepancy pixels. The preservation loss lowers suppression on changed pixels above the median deep distance. A fixed Sobel boundary loss adds no parameters. The appearance proxy is generated before model fitting, while the preservation set uses only annotated training changes and frozen features. Throughput includes loading and preprocessing each prepared pair, device inference and synchronization, excludes output-file writes, and is specific to the selected device; it is not an edge-device benchmark.
