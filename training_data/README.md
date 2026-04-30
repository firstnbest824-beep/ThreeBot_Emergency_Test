# Three-Bot Training Data

This directory is the project-local home for training inputs.

Large files in this directory are intentionally excluded from git. Keep this
README in the public repo so another machine knows where each dataset belongs.

## Layout

```text
training_data/
  raw_bags/              # Original ROS 2 bag episodes
  processed_episodes/    # Preprocessed .npy episodes used by the current pilot path
  rlds/                  # Optional RLDS/TFDS output generated from raw_bags
  legacy_vla_misc/       # Older VLA scripts/logs and legacy dataset artifacts
```

## Current Contents

- `raw_bags`: 83 ROS bag episodes, about 211G
- `processed_episodes`: 83 `.npy` episodes, about 35G
- `legacy_vla_misc`: older helper scripts plus `vla_dataset_final.npy`

`legacy_vla_misc/vla_dataset_final.npy` is preserved, but it raised an EOF error
when loaded with NumPy during reorganization. Treat `processed_episodes` as the
current valid training input.

Legacy logs, diagnostic images, and older checkpoints were moved under
`training_outputs/`.

## Training

The default training command now reads `training_data/processed_episodes/*.npy`
and writes checkpoints to `training_outputs/checkpoints`.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONPATH=쓰리봇/three_bot_octo \
python3 -m three_bot_octo.training
```

## Expected Processed Episode Schema

Each file in `processed_episodes/` is expected to contain step dictionaries with:

| Key | Meaning |
| --- | --- |
| `img_left` | Left gripper RGB frame, resized to `224x224` during preprocessing |
| `img_right` | Right gripper RGB frame |
| `img_global` | Global workspace RGB frame |
| `action` | 5-D control target: `[dX1, dY1, dX2, dY2, dZ]` |

The current processed files do not contain proprioception. The training loader
therefore provides a zero `[10]` state vector so the model contract stays
compatible with the RLDS path.

## RLDS Conversion

To build an RLDS/TFDS dataset from the raw ROS bags:

```bash
PYTHONPATH=쓰리봇/three_bot_octo \
python3 -m three_bot_octo.rlds_converter \
  --bag_dir training_data/raw_bags \
  --output_dir training_data/rlds \
  --dataset_name three_bot_rlds
```
