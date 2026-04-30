# Three-Bot Training Outputs

This directory stores training results and legacy output artifacts.

Large checkpoint files are intentionally excluded from git. Commit this README
only; keep actual run outputs local unless a separate model registry or release
artifact is used.

## Layout

```text
training_outputs/
  checkpoints/                 # Default output for new training runs
  pilot_processed_npy/          # First successful pilot run
  legacy_vla_checkpoints/       # Older VLA checkpoints moved from VLA_Data
  legacy_vla_training_log.txt   # Older training log
  legacy_dataset_diagnostic.png # Older dataset diagnostic image
```

## Current Pilot Run

`pilot_processed_npy/` records the first successful smoke test against the
processed `.npy` training path. The checkpoint itself is local-only and ignored
by git; the README inside that directory documents the command and metrics.
