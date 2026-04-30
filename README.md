# ThreeBot Emergency Test

ThreeBot Emergency Test is the local workspace for the Three-Bot VLA/Octo training path. It keeps the robot source code, the current training pipeline, the reorganized training data, and the training outputs in one project folder.

The public repository should contain source code and documentation only. Raw bags, processed episodes, checkpoints, credentials, and recovery dumps stay local and are excluded by `.gitignore`.

## Current Layout

```text
ThreeBot_Emergency_Test/
  README.md                          # Project overview and operating guide
  .gitignore                         # Public-repo safety rules
  쓰리봇/three_bot_octo/              # Main JAX/Flax + ROS 2 policy package
  threebot_collected/code/           # ROS 2 control, logging, core, and support packages
  threebot_collected/firmware/       # STM32 firmware source for the robot side
  training_data/                     # Local-only training inputs
  training_outputs/                  # Local-only checkpoints and run outputs
  label_data/                        # Local-only small recovered label samples
```

## Data Status

```text
training_data/
  raw_bags/               # 83 original ROS 2 bag episodes, about 211G
  processed_episodes/     # 83 processed .npy episodes used by current training, about 35G
  rlds/                   # Optional generated RLDS/TFDS dataset output
  legacy_vla_misc/        # Old VLA scripts/artifacts; includes a corrupt legacy dataset file

training_outputs/
  checkpoints/                  # Default output path for new training runs
  pilot_processed_npy/          # First successful one-step pilot training run
  legacy_vla_checkpoints/       # Older VLA checkpoints
  legacy_vla_training_log.txt   # Older training log
  legacy_dataset_diagnostic.png # Older diagnostic plot
```

The current valid training input is `training_data/processed_episodes/*.npy`. The old `legacy_vla_misc/vla_dataset_final.npy` is preserved for reference, but it failed NumPy loading during inspection and should not be treated as the active dataset.

## Main Code Paths

### Policy and Training

```text
쓰리봇/three_bot_octo/three_bot_octo/
  training.py              # End-to-end JAX/Flax training loop
  octo_model.py            # Vision + proprioception + transformer policy model
  vision_encoder.py        # Three-camera ResNet encoder and cross-view fusion
  action_head.py           # 5-DoF action head: XY MSE + Z Gaussian NLL
  gaussian_action_head.py  # Uncertainty-aware Gaussian head for shared Z
  rlds_converter.py        # ROS 2 bag to RLDS/TFDS converter
  octo_inference_node.py   # ROS 2 real-time inference node
```

### ROS 2 Runtime Packages

```text
threebot_collected/code/
  three_bot_control/       # Teleop, mux, safety filter, motor bridge
  three_bot_core/          # Message, service, and launch definitions
  three_bot_logging/       # Episodic synchronized data logger
  three_bot_perception/    # Perception package skeleton
```

## Training Flow

1. Raw teleoperation episodes are stored as ROS 2 bags under `training_data/raw_bags/`.
2. The current working path uses processed `.npy` episodes under `training_data/processed_episodes/`.
3. `training.py` loads each step as three camera frames plus a 5-D action.
4. Because the processed `.npy` files do not include robot state, the loader fills proprioception with a zero `[10]` vector.
5. `octo_model.py` encodes camera frames and proprioception tokens, fuses them with transformer blocks, and predicts `[dX1, dY1, dX2, dY2, dZ]`.
6. New checkpoints are written to `training_outputs/checkpoints/`.

## Run A Pilot Training Check

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONPATH=쓰리봇/three_bot_octo \
python3 -m three_bot_octo.training \
  --dataset_format processed_npy \
  --dataset_path "training_data/processed_episodes/*.npy" \
  --checkpoint_dir training_outputs/checkpoints \
  --batch_size 1 \
  --num_epochs 1 \
  --steps_per_epoch 1 \
  --max_train_steps_per_epoch 1 \
  --max_val_steps 1
```

## Convert Raw Bags to RLDS

```bash
PYTHONPATH=쓰리봇/three_bot_octo \
python3 -m three_bot_octo.rlds_converter \
  --bag_dir training_data/raw_bags \
  --output_dir training_data/rlds \
  --dataset_name three_bot_rlds
```

## Public Repository Policy

Do commit:

- Source code under `쓰리봇/three_bot_octo/`.
- ROS 2 source packages under `threebot_collected/code/`.
- Firmware source under `threebot_collected/firmware/`.
- README files and lightweight documentation.

Do not commit:

- `training_data/raw_bags/`
- `training_data/processed_episodes/`
- `training_data/legacy_vla_misc/`
- `training_outputs/checkpoints/`
- `training_outputs/legacy_vla_checkpoints/`
- `training_outputs/pilot_processed_npy/checkpoints/`
- Any `credentials.json`, model checkpoint, `.npy`, `.db3`, or video file.

## GitHub Note

This workspace is connected to the GitHub repository `firstnbest824-beep/ThreeBot_Emergency_Test` on branch `main`.

Standard git push and pull workflows are sufficient for publishing source changes. GitHub CLI is optional, not required.
