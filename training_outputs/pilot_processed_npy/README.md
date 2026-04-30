# Three-Bot Pilot Training Result

This folder contains the first successful pilot training output created from the
current project's preprocessed `.npy` data.

## Source Data

- Dataset format: `processed_npy`
- Dataset path: `training_data/processed_episodes/*.npy`
- Note: `processed_npy` samples do not include `robot_state`, so training used a
  zero-filled 10D proprioception vector.

## Run Settings

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true \
PYTHONPATH=쓰리봇/three_bot_octo \
python3 -m three_bot_octo.training \
  --dataset_format processed_npy \
  --dataset_path 'training_data/processed_episodes/*.npy' \
  --checkpoint_dir /tmp/threebot_pilot_ckpt \
  --batch_size 1 \
  --num_epochs 1 \
  --steps_per_epoch 1 \
  --max_train_steps_per_epoch 1 \
  --max_val_steps 1
```

## Metrics

- Train loss: `1.0656`
- Validation loss: `1.2203`
- XY loss: `0.0688`
- Z MSE: `0.0644`
- Z uncertainty: `1.1020`
- Gradient norm: `77.3916`

## Output

- Checkpoint directory: `checkpoints/checkpoint_0`
- Checkpoint size: about `344M`
