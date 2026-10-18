# IFF-PCS

Implementation of **Cross-Layer Foundation Representations for Pseudo-Change Suppression in Remote Sensing Change Detection**. A frozen encoder supplies shallow, middle and deep temporal distances. A learned attenuation gate suppresses appearance-sensitive unchanged pixels while a complementary loss preserves annotated deep-feature changes.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, torchvision 0.22.1, scikit-learn 1.6.1, matplotlib 3.10.0, pillow 12.3.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

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



Preparation resizes pairs and nearest-neighbor masks to `--size` (518 by default, divisible by the 14-pixel patch size). It defines fixed appearance-sensitive proxy masks using per-image standardized RGB discrepancy on unchanged pixels, excluding a two-pixel change-boundary band. `--percentile 0.85` controls that proxy. `--small-area 64` defines the small-change connected-component panel at evaluation resolution. Freeze these definitions before comparisons; proxies do not depend on a detector's predictions.

```bash
iffpcs prepare --manifest data/manifest.json --output data/prepared
```

## Training and inference

```bash
iffpcs train --data data/prepared --output runs/train --seed 42
```

## Evaluation

```bash
iffpcs evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

## Comparisons and ablations

```bash
iffpcs train --data data/prepared --seed 42 --mode rgb --no-gate --output runs/rgb
iffpcs train --data data/prepared --seed 42 --mode deep --no-gate --output runs/deep
iffpcs train --data data/prepared --seed 42 --mode middle --no-gate --output runs/middle
iffpcs train --data data/prepared --seed 42 --no-gate --output runs/concatenation
iffpcs train --data data/prepared --seed 42 --no-pseudo --no-preserve --output runs/gate_only
iffpcs train --data data/prepared --seed 42 --no-preserve --output runs/no_preserve
iffpcs train --data data/prepared --seed 42 --no-boundary --output runs/no_boundary
iffpcs train --data data/prepared --seed 42 --fraction 0.1 --output runs/labels10
iffpcs evaluate --data data/prepared --checkpoint runs/train/model.pt --permute-gate --output runs/permuted
iffpcs evaluate --data data/prepared --checkpoint runs/train/model.pt --stress brightness --output runs/brightness
iffpcs evaluate --data data/prepared --checkpoint runs/train/model.pt --stress noise --output runs/noise
iffpcs evaluate --data data/prepared --checkpoint runs/train/model.pt --stress translation --output runs/translation
```

Evaluate each fitted control with its own checkpoint; scoring/stress commands above use the full checkpoint.

## Outputs

Training writes `runs/train/model.pt`. Evaluation writes predicted masks, `metrics.json`, `pairs.json` and `summary.png`.
