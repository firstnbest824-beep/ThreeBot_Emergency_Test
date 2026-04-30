# Three-Bot Octo Package

ROS 2 + JAX/Flax package for training and running a Three-Bot visual-language-action policy. The model is adapted for a 5-DoF dual-winch control target:

```text
[dX1, dY1, dX2, dY2, dZ]
```

## Package Structure

```text
three_bot_octo/
  action_head.py           # 5-DoF policy head; deterministic XY + Gaussian Z
  gaussian_action_head.py  # Gaussian negative-log-likelihood head for Z
  octo_model.py            # Vision/proprioception fusion and transformer policy
  vision_encoder.py        # Three-camera ResNet encoder with cross-view attention
  training.py              # Dataset loading, training loop, evaluation, checkpointing
  rlds_converter.py        # ROS 2 bag to RLDS/TFDS conversion utility
  octo_inference_node.py   # ROS 2 inference node publishing policy actions
```

## Model Data Contract

| Tensor | Shape | Description |
| --- | --- | --- |
| `observations["gripper1"]` | `[B, T, 224, 224, 3]` | Left gripper RGB camera |
| `observations["gripper2"]` | `[B, T, 224, 224, 3]` | Right gripper RGB camera |
| `observations["global"]` | `[B, T, 224, 224, 3]` | Global workspace RGB camera |
| `proprioception` | `[B, 10]` | `[x1, y1, x2, y2, z, vx1, vy1, vx2, vy2, vz]` |
| `actions` | `[B, 5]` | `[dX1, dY1, dX2, dY2, dZ]` |

The current `processed_npy` training path does not contain proprioception, so
`training.py` fills it with zeros. The RLDS path preserves real robot state when
the ROS bag contains `/three_bot/control/robot_state`.

## Training

Run from the project root:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONPATH=쓰리봇/three_bot_octo \
python3 -m three_bot_octo.training \
  --dataset_format processed_npy \
  --dataset_path "training_data/processed_episodes/*.npy" \
  --checkpoint_dir training_outputs/checkpoints
```

For a fast smoke test:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONPATH=쓰리봇/three_bot_octo \
python3 -m three_bot_octo.training \
  --batch_size 1 \
  --num_epochs 1 \
  --steps_per_epoch 1 \
  --max_train_steps_per_epoch 1 \
  --max_val_steps 1
```

## RLDS Conversion

```bash
PYTHONPATH=쓰리봇/three_bot_octo \
python3 -m three_bot_octo.rlds_converter \
  --bag_dir training_data/raw_bags \
  --output_dir training_data/rlds \
  --dataset_name three_bot_rlds
```

## Inference

```bash
ros2 run three_bot_octo octo_inference_node \
  --ros-args \
  -p checkpoint_path:=training_outputs/checkpoints \
  -p control_rate:=20.0 \
  -p model_size:=small
```

## Core Flow

1. `vision_encoder.py` turns three synchronized cameras into visual tokens.
2. `octo_model.py` converts robot state into three proprioception tokens and concatenates them with the visual tokens.
3. Transformer blocks fuse the camera and state context.
4. `action_head.py` predicts four deterministic XY deltas and one uncertainty-aware Z delta.
5. `training.py` optimizes behavior-cloning loss and writes checkpoints.
