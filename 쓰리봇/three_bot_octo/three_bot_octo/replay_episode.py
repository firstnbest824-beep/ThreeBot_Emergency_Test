"""Offline episode replay for comparing human actions with model actions.

The script loads one processed `.npy` episode, restores a trained Octo
checkpoint, predicts actions with the same forward path used for inference, and
writes a replay video plus action comparison artifacts.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from .octo_model import create_three_bot_octo_model


ACTION_NAMES = ("dX1", "dY1", "dX2", "dY2", "dZ")


def _resolve_episode(
    dataset_path: str,
    split: str,
    episode_index: int,
    episode_path: str | None,
) -> Path:
    if episode_path:
        path = Path(episode_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Episode not found: {path}")
        return path

    files = [Path(p).resolve() for p in sorted(glob.glob(dataset_path))]
    if not files:
        raise FileNotFoundError(f"No episode files matched: {dataset_path}")

    train_count = max(1, int(len(files) * 0.8))
    if split == "train":
        selected = files[:train_count]
    elif split == "val":
        selected = files[train_count:] or files[-1:]
    elif split == "all":
        selected = files
    else:
        raise ValueError(f"Unsupported split: {split}")

    if episode_index < 0 or episode_index >= len(selected):
        raise IndexError(
            f"episode_index={episode_index} outside {split} split size {len(selected)}"
        )
    return selected[episode_index]


def _load_episode(path: Path) -> np.ndarray:
    episode = np.load(path, allow_pickle=True)
    if len(episode) == 0:
        raise ValueError(f"Episode has no steps: {path}")
    required = {"img_left", "img_right", "img_global", "action"}
    missing = required.difference(episode[0].keys())
    if missing:
        raise ValueError(f"Episode missing fields {sorted(missing)}: {path}")
    return episode


def _iter_batches(episode: np.ndarray, batch_size: int) -> Iterable[Tuple[int, Dict[str, np.ndarray]]]:
    for start in range(0, len(episode), batch_size):
        steps = episode[start : start + batch_size]
        obs = {
            "gripper1": np.stack(
                [step["img_left"].astype(np.float32) / 255.0 for step in steps], axis=0
            )[:, None, ...],
            "gripper2": np.stack(
                [step["img_right"].astype(np.float32) / 255.0 for step in steps], axis=0
            )[:, None, ...],
            "global": np.stack(
                [step["img_global"].astype(np.float32) / 255.0 for step in steps], axis=0
            )[:, None, ...],
        }
        yield start, obs


def predict_episode(
    checkpoint_path: Path,
    episode: np.ndarray,
    model_size: str,
    batch_size: int,
) -> np.ndarray:
    model = create_three_bot_octo_model(model_size=model_size)
    checkpoint = checkpoints.restore_checkpoint(
        ckpt_dir=str(checkpoint_path.expanduser().resolve()),
        target=None,
    )
    if checkpoint is None or "params" not in checkpoint:
        raise RuntimeError(f"Failed to restore checkpoint params from {checkpoint_path}")

    variables = {"params": checkpoint["params"]}
    if checkpoint.get("batch_stats"):
        variables["batch_stats"] = checkpoint["batch_stats"]

    @jax.jit
    def _predict(observations, proprioception):
        return model.apply(
            variables,
            observations,
            proprioception,
            train=False,
            rngs={"sample": jax.random.PRNGKey(0)},
        )

    predictions: List[np.ndarray] = []
    for _, obs in _iter_batches(episode, batch_size):
        proprioception = np.zeros((obs["global"].shape[0], 10), dtype=np.float32)
        pred = _predict(
            jax.tree_util.tree_map(jnp.asarray, obs),
            jnp.asarray(proprioception),
        )
        predictions.append(np.asarray(pred))
    return np.concatenate(predictions, axis=0)


def _compute_metrics(gt: np.ndarray, pred: np.ndarray, sign_threshold: float) -> Dict[str, np.ndarray | float]:
    error = pred - gt
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    max_abs = np.max(np.abs(error), axis=0)

    active = np.abs(gt) > sign_threshold
    sign_acc = np.full(gt.shape[1], np.nan, dtype=np.float32)
    for i in range(gt.shape[1]):
        if np.any(active[:, i]):
            sign_acc[i] = np.mean(np.sign(pred[active[:, i], i]) == np.sign(gt[active[:, i], i]))

    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs_error": max_abs,
        "mean_abs_delta_pred": np.mean(np.abs(np.diff(pred, axis=0)), axis=0),
        "mean_abs_delta_gt": np.mean(np.abs(np.diff(gt, axis=0)), axis=0),
        "sign_accuracy": sign_acc,
        "overall_mae": float(np.mean(mae)),
        "overall_rmse": float(np.mean(rmse)),
    }


def _baseline_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, Tuple[float, float]]:
    baselines = {
        "model": pred,
        "zero": np.zeros_like(gt),
        "episode_mean": np.tile(np.mean(gt, axis=0), (len(gt), 1)),
        "previous_gt": np.vstack([gt[0], gt[:-1]]),
    }
    summary = {}
    for name, values in baselines.items():
        error = values - gt
        mae = float(np.mean(np.mean(np.abs(error), axis=0)))
        rmse = float(np.mean(np.sqrt(np.mean(error**2, axis=0))))
        summary[name] = (mae, rmse)
    return summary


def _write_csv(output_path: Path, gt: np.ndarray, pred: np.ndarray) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["step"]
        header.extend(f"gt_{name}" for name in ACTION_NAMES)
        header.extend(f"pred_{name}" for name in ACTION_NAMES)
        header.extend(f"err_{name}" for name in ACTION_NAMES)
        writer.writerow(header)
        for i, (gt_row, pred_row) in enumerate(zip(gt, pred)):
            writer.writerow([i, *gt_row.tolist(), *pred_row.tolist(), *(pred_row - gt_row).tolist()])


def _write_metrics(
    output_path: Path,
    episode_path: Path,
    checkpoint_path: Path,
    metrics: Dict[str, np.ndarray | float],
    baselines: Dict[str, Tuple[float, float]],
) -> None:
    lines = [
        f"episode: {episode_path}",
        f"checkpoint: {checkpoint_path}",
        f"overall_mae: {metrics['overall_mae']:.8f}",
        f"overall_rmse: {metrics['overall_rmse']:.8f}",
        "",
        "axis,mae,rmse,max_abs_error,sign_accuracy,mean_abs_delta_gt,mean_abs_delta_pred",
    ]
    for i, name in enumerate(ACTION_NAMES):
        lines.append(
            ",".join(
                [
                    name,
                    f"{metrics['mae'][i]:.8f}",
                    f"{metrics['rmse'][i]:.8f}",
                    f"{metrics['max_abs_error'][i]:.8f}",
                    f"{metrics['sign_accuracy'][i]:.8f}",
                    f"{metrics['mean_abs_delta_gt'][i]:.8f}",
                    f"{metrics['mean_abs_delta_pred'][i]:.8f}",
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n")

    baseline_lines = ["", "baseline,overall_mae,overall_rmse"]
    for name, (mae, rmse) in baselines.items():
        baseline_lines.append(f"{name},{mae:.8f},{rmse:.8f}")
    with output_path.open("a") as f:
        f.write("\n".join(baseline_lines) + "\n")


def _scale_series(values: np.ndarray, y_min: float, y_max: float, top: int, bottom: int) -> np.ndarray:
    if y_max <= y_min:
        return np.full_like(values, (top + bottom) // 2, dtype=np.int32)
    norm = (values - y_min) / (y_max - y_min)
    return (bottom - norm * (bottom - top)).astype(np.int32)


def write_action_plot(output_path: Path, gt: np.ndarray, pred: np.ndarray) -> None:
    width = 1600
    row_h = 180
    left = 105
    right = width - 30
    top_margin = 55
    height = top_margin + row_h * len(ACTION_NAMES) + 35
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    cv2.putText(
        canvas,
        "Episode action replay: GT (green) vs Pred (red)",
        (30, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    steps = np.arange(len(gt))
    x = (left + (steps / max(1, len(gt) - 1)) * (right - left)).astype(np.int32)

    for axis, name in enumerate(ACTION_NAMES):
        row_top = top_margin + axis * row_h
        row_bottom = row_top + row_h - 25
        y_min = float(min(np.min(gt[:, axis]), np.min(pred[:, axis])))
        y_max = float(max(np.max(gt[:, axis]), np.max(pred[:, axis])))
        pad = max(1e-6, (y_max - y_min) * 0.1)
        y_min -= pad
        y_max += pad

        cv2.rectangle(canvas, (left, row_top), (right, row_bottom), (225, 225, 225), 1)
        zero_y = int(_scale_series(np.array([0.0]), y_min, y_max, row_top, row_bottom)[0])
        if row_top <= zero_y <= row_bottom:
            cv2.line(canvas, (left, zero_y), (right, zero_y), (210, 210, 210), 1)

        gt_y = _scale_series(gt[:, axis], y_min, y_max, row_top, row_bottom)
        pred_y = _scale_series(pred[:, axis], y_min, y_max, row_top, row_bottom)
        gt_pts = np.stack([x, gt_y], axis=1).reshape((-1, 1, 2))
        pred_pts = np.stack([x, pred_y], axis=1).reshape((-1, 1, 2))
        cv2.polylines(canvas, [gt_pts], False, (30, 150, 30), 2, cv2.LINE_AA)
        cv2.polylines(canvas, [pred_pts], False, (30, 30, 220), 2, cv2.LINE_AA)

        cv2.putText(canvas, name, (25, row_top + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        cv2.putText(
            canvas,
            f"{y_max:.3f}",
            (left - 78, row_top + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 80, 80),
            1,
        )
        cv2.putText(
            canvas,
            f"{y_min:.3f}",
            (left - 78, row_bottom),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 80, 80),
            1,
        )

    cv2.imwrite(str(output_path), canvas)


def _draw_bar_pair(
    frame: np.ndarray,
    label: str,
    gt_value: float,
    pred_value: float,
    x: int,
    y: int,
    width: int,
    scale: float,
) -> None:
    center = x + width // 2
    cv2.putText(frame, label, (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.line(frame, (center, y + 28), (center, y + 78), (120, 120, 120), 1)
    for offset, value, color, text_label in [
        (36, gt_value, (70, 220, 70), "GT"),
        (62, pred_value, (70, 70, 255), "PR"),
    ]:
        bar_len = int(np.clip(value / scale, -1.0, 1.0) * (width // 2 - 8))
        x0, x1 = sorted((center, center + bar_len))
        cv2.rectangle(frame, (x0, y + offset), (x1, y + offset + 14), color, -1)
        cv2.putText(
            frame,
            f"{text_label} {value:+.3f}",
            (x + width + 12, y + offset + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def write_replay_video(
    output_path: Path,
    episode: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    fps: float,
    stride: int,
) -> None:
    first = episode[0]
    cam_h, cam_w = first["img_left"].shape[:2]
    panel_h = 260
    frame_size = (cam_w * 3, cam_h + panel_h)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    scale = float(np.max(np.abs(np.concatenate([gt, pred], axis=0))))
    scale = max(scale, 1e-6)

    for step_idx in range(0, len(episode), stride):
        step = episode[step_idx]
        cams = [
            cv2.cvtColor(step["img_left"], cv2.COLOR_RGB2BGR),
            cv2.cvtColor(step["img_right"], cv2.COLOR_RGB2BGR),
            cv2.cvtColor(step["img_global"], cv2.COLOR_RGB2BGR),
        ]
        top = np.concatenate(cams, axis=1)
        panel = np.zeros((panel_h, frame_size[0], 3), dtype=np.uint8)
        cv2.putText(
            panel,
            f"step {step_idx + 1}/{len(episode)} | GT green, Pred red",
            (22, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(panel, "left gripper", (22, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(panel, "right gripper", (cam_w + 22, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(panel, "global", (cam_w * 2 + 22, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        bar_w = 170
        start_x = 22
        for axis, name in enumerate(ACTION_NAMES):
            _draw_bar_pair(
                panel,
                name,
                float(gt[step_idx, axis]),
                float(pred[step_idx, axis]),
                start_x + axis * 300,
                92,
                bar_w,
                scale,
            )

        frame = np.concatenate([top, panel], axis=0)
        writer.write(frame)
    writer.release()


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Replay one episode and compare GT actions with model predictions.")
    parser.add_argument("--checkpoint", default="training_outputs/octo_full_processed_npy_20260430_115347/best_available_checkpoint_epoch98")
    parser.add_argument("--dataset_path", default="training_data/processed_episodes/*.npy")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--episode", default=None, help="Explicit .npy episode path. Overrides split/index.")
    parser.add_argument("--model_size", default="small", choices=["small", "base"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=None, help="Optional step limit for quick previews.")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sign_threshold", type=float, default=1e-6)
    parser.add_argument("--output_dir", default="training_outputs/replays")
    parser.add_argument("--no_video", action="store_true")
    parsed = parser.parse_args(args=args)

    episode_path = _resolve_episode(parsed.dataset_path, parsed.split, parsed.episode_index, parsed.episode)
    checkpoint_path = Path(parsed.checkpoint).expanduser().resolve()
    output_root = Path(parsed.output_dir).expanduser().resolve()
    output_dir = output_root / f"{episode_path.stem}_{checkpoint_path.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Episode: {episode_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")

    episode = _load_episode(episode_path)
    if parsed.max_steps is not None:
        episode = episode[: parsed.max_steps]
        if len(episode) == 0:
            raise ValueError("--max_steps produced an empty episode")
    gt = np.stack([step["action"].astype(np.float32) for step in episode], axis=0)
    pred = predict_episode(checkpoint_path, episode, parsed.model_size, parsed.batch_size)

    metrics = _compute_metrics(gt, pred, parsed.sign_threshold)
    baselines = _baseline_metrics(gt, pred)
    _write_csv(output_dir / "actions.csv", gt, pred)
    _write_metrics(output_dir / "metrics.txt", episode_path, checkpoint_path, metrics, baselines)
    write_action_plot(output_dir / "action_plot.png", gt, pred)
    if not parsed.no_video:
        write_replay_video(output_dir / "replay.mp4", episode, gt, pred, parsed.fps, parsed.stride)

    print("\nSummary")
    print(f"  steps: {len(episode)}")
    print(f"  overall_mae: {metrics['overall_mae']:.8f}")
    print(f"  overall_rmse: {metrics['overall_rmse']:.8f}")
    print("  baselines:")
    for name, (mae, rmse) in baselines.items():
        print(f"    {name}: overall_mae={mae:.8f}, overall_rmse={rmse:.8f}")
    for i, name in enumerate(ACTION_NAMES):
        print(
            f"  {name}: mae={metrics['mae'][i]:.8f}, "
            f"rmse={metrics['rmse'][i]:.8f}, "
            f"sign_acc={metrics['sign_accuracy'][i]:.4f}"
        )


if __name__ == "__main__":
    main()
