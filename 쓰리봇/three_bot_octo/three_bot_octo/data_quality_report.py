"""Visual data-quality report for processed Three-Bot episodes."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np


ACTION_NAMES = ("dX1", "dY1", "dX2", "dY2", "dZ")
CAMERA_KEYS = ("img_left", "img_right", "img_global")


def _load_episode(path: Path) -> np.ndarray:
    episode = np.load(path, allow_pickle=True)
    if len(episode) == 0:
        raise ValueError(f"Episode has no steps: {path}")
    missing = {"action", *CAMERA_KEYS}.difference(episode[0].keys())
    if missing:
        raise ValueError(f"Episode missing fields {sorted(missing)}: {path}")
    return episode


def _episode_actions(episode: np.ndarray) -> np.ndarray:
    return np.stack([step["action"].astype(np.float32) for step in episode], axis=0)


def _collect_episode_stats(files: Sequence[Path], threshold: float) -> tuple[List[Dict], np.ndarray]:
    rows: List[Dict] = []
    all_actions: List[np.ndarray] = []

    for path in files:
        episode = _load_episode(path)
        actions = _episode_actions(episode)
        active = np.abs(actions) > threshold
        active_any = np.any(active, axis=1)

        same_frames = {key: 0 for key in CAMERA_KEYS}
        dark_frames = {key: 0 for key in CAMERA_KEYS}
        bright_frames = {key: 0 for key in CAMERA_KEYS}
        bad_shapes = 0
        previous = None

        for step in episode:
            images = {key: step[key] for key in CAMERA_KEYS}
            for key, image in images.items():
                if image.ndim != 3 or image.shape[-1] not in (3, 4):
                    bad_shapes += 1
                mean = float(np.mean(image))
                dark_frames[key] += int(mean < 5.0)
                bright_frames[key] += int(mean > 250.0)
                if previous is not None and np.array_equal(image, previous[key]):
                    same_frames[key] += 1
            previous = images

        row = {
            "episode": path.name,
            "path": str(path),
            "steps": len(episode),
            "active_any": int(np.sum(active_any)),
            "active_pct": float(np.mean(active_any) * 100.0),
            "bad_shapes": bad_shapes,
            "same_frames": same_frames,
            "dark_frames": dark_frames,
            "bright_frames": bright_frames,
        }
        for i, name in enumerate(ACTION_NAMES):
            row[f"{name}_active"] = int(np.sum(active[:, i]))
            row[f"{name}_active_pct"] = float(np.mean(active[:, i]) * 100.0)
            row[f"{name}_mean_abs"] = float(np.mean(np.abs(actions[:, i])))
            row[f"{name}_pos"] = int(np.sum(actions[:, i] > threshold))
            row[f"{name}_neg"] = int(np.sum(actions[:, i] < -threshold))
        rows.append(row)
        all_actions.append(actions)

    return rows, np.concatenate(all_actions, axis=0)


def _put_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.62,
    color: tuple[int, int, int] = (35, 35, 35),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_stacked_bar(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    values: Sequence[float],
    colors: Sequence[tuple[int, int, int]],
    labels: Sequence[str],
) -> None:
    total = max(float(sum(values)), 1e-9)
    cursor = x
    for value, color, label in zip(values, colors, labels):
        w = int(round(width * float(value) / total))
        cv2.rectangle(image, (cursor, y), (cursor + w, y + height), color, -1)
        if w > 95:
            _put_text(image, label, cursor + 8, y + height // 2 + 6, 0.48, (255, 255, 255), 1)
        cursor += w
    cv2.rectangle(image, (x, y), (x + width, y + height), (80, 80, 80), 1)


def _draw_axis_bars(
    image: np.ndarray,
    actions: np.ndarray,
    threshold: float,
    x: int,
    y: int,
    width: int,
    row_h: int,
) -> None:
    _put_text(image, "Axis action balance: negative / zero / positive", x, y - 15, 0.62, (20, 20, 20), 2)
    for i, name in enumerate(ACTION_NAMES):
        vals = actions[:, i]
        neg = np.sum(vals < -threshold)
        pos = np.sum(vals > threshold)
        zero = len(vals) - neg - pos
        yy = y + i * row_h
        _put_text(image, name, x, yy + 25, 0.55, (25, 25, 25), 2)
        _draw_stacked_bar(
            image,
            x + 70,
            yy + 3,
            width - 70,
            28,
            [neg, zero, pos],
            [(70, 95, 210), (190, 190, 190), (45, 155, 80)],
            [
                f"- {neg / len(vals) * 100:.1f}%",
                f"0 {zero / len(vals) * 100:.1f}%",
                f"+ {pos / len(vals) * 100:.1f}%",
            ],
        )


def _draw_episode_activity(
    image: np.ndarray,
    rows: Sequence[Dict],
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    _put_text(image, "Episode active-step ratio, sorted low to high", x, y - 15, 0.62, (20, 20, 20), 2)
    sorted_rows = sorted(rows, key=lambda item: item["active_pct"])
    n = max(len(sorted_rows), 1)
    bar_w = max(2, width // n)
    chart_bottom = y + height
    cv2.rectangle(image, (x, y), (x + width, chart_bottom), (220, 220, 220), 1)
    for pct in (25, 50, 75):
        yy = int(chart_bottom - height * pct / 100.0)
        cv2.line(image, (x, yy), (x + width, yy), (232, 232, 232), 1)
        _put_text(image, f"{pct}%", x + width + 8, yy + 5, 0.42, (90, 90, 90), 1)

    for i, row in enumerate(sorted_rows):
        pct = row["active_pct"]
        bar_h = int(height * pct / 100.0)
        color = (80, 150, 80) if pct >= 35.0 else (70, 105, 210)
        xx = x + i * bar_w
        cv2.rectangle(image, (xx, chart_bottom - bar_h), (xx + bar_w - 1, chart_bottom), color, -1)

    low = sorted_rows[0]
    high = sorted_rows[-1]
    median = float(np.median([row["active_pct"] for row in sorted_rows]))
    _put_text(image, f"lowest: {low['episode']} ({low['active_pct']:.1f}%)", x, chart_bottom + 25, 0.48)
    _put_text(image, f"median: {median:.1f}%", x + 520, chart_bottom + 25, 0.48)
    _put_text(image, f"highest: {high['episode']} ({high['active_pct']:.1f}%)", x + 760, chart_bottom + 25, 0.48)


def _draw_histograms(
    image: np.ndarray,
    actions: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    _put_text(image, "Action-value histograms", x, y - 15, 0.62, (20, 20, 20), 2)
    panel_w = width // len(ACTION_NAMES)
    for i, name in enumerate(ACTION_NAMES):
        vals = actions[:, i]
        vmax = float(max(np.max(np.abs(vals)), 1e-6))
        hist, edges = np.histogram(vals, bins=41, range=(-vmax, vmax))
        hist = hist.astype(np.float32) / max(float(np.max(hist)), 1.0)
        xx = x + i * panel_w
        cv2.rectangle(image, (xx, y), (xx + panel_w - 14, y + height), (225, 225, 225), 1)
        zero_x = xx + int((panel_w - 14) * (-edges[0]) / (edges[-1] - edges[0]))
        cv2.line(image, (zero_x, y), (zero_x, y + height), (200, 200, 200), 1)
        bin_w = max(1, (panel_w - 14) // len(hist))
        for j, h in enumerate(hist):
            bar_h = int(h * (height - 25))
            bx = xx + j * bin_w
            cv2.rectangle(image, (bx, y + height - bar_h), (bx + bin_w, y + height), (75, 130, 190), -1)
        _put_text(image, name, xx + 8, y + 22, 0.52, (30, 30, 30), 2)
        _put_text(image, f"range +/-{vmax:.3f}", xx + 8, y + height + 22, 0.42, (80, 80, 80), 1)


def write_dashboard(output_path: Path, rows: Sequence[Dict], actions: np.ndarray, threshold: float) -> None:
    width, height = 1800, 1450
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    active = np.abs(actions) > threshold
    active_any = np.any(active, axis=1)
    total_steps = len(actions)
    active_steps = int(np.sum(active_any))
    zero_steps = total_steps - active_steps

    _put_text(canvas, "Three-Bot Dataset Quality Dashboard", 40, 50, 1.05, (20, 20, 20), 2)
    _put_text(canvas, f"episodes: {len(rows)}", 40, 95, 0.65)
    _put_text(canvas, f"steps: {total_steps}", 250, 95, 0.65)
    _put_text(canvas, f"active steps: {active_steps} ({active_steps / total_steps * 100:.1f}%)", 470, 95, 0.65)
    _put_text(canvas, f"all-zero steps: {zero_steps} ({zero_steps / total_steps * 100:.1f}%)", 820, 95, 0.65)
    _put_text(canvas, f"threshold: {threshold:g}", 1210, 95, 0.65)

    _draw_stacked_bar(
        canvas,
        40,
        130,
        1180,
        44,
        [zero_steps, active_steps],
        [(180, 180, 180), (45, 155, 80)],
        [f"zero {zero_steps / total_steps * 100:.1f}%", f"active {active_steps / total_steps * 100:.1f}%"],
    )

    _draw_axis_bars(canvas, actions, threshold, 40, 245, 820, 52)
    _draw_episode_activity(canvas, rows, 40, 610, 1660, 330)
    _draw_histograms(canvas, actions, 40, 1060, 1660, 250)

    quality_notes = [
        "Read this as a data-health view, not model accuracy.",
        "Too many gray zero regions mean the model can learn to stay still.",
        "Blue/green imbalance means one direction has fewer examples.",
        "Low bars in the episode chart are weak demonstrations.",
    ]
    for i, line in enumerate(quality_notes):
        _put_text(canvas, line, 40, 1370 + i * 23, 0.5, (70, 70, 70), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _scale_series(values: np.ndarray, y_min: float, y_max: float, top: int, bottom: int) -> np.ndarray:
    if y_max <= y_min:
        return np.full_like(values, (top + bottom) // 2, dtype=np.int32)
    norm = (values - y_min) / (y_max - y_min)
    return (bottom - norm * (bottom - top)).astype(np.int32)


def write_episode_timeline(output_path: Path, episode_path: Path, threshold: float) -> None:
    episode = _load_episode(episode_path)
    actions = _episode_actions(episode)
    active_any = np.any(np.abs(actions) > threshold, axis=1)
    n = len(episode)

    width = 1800
    thumb_h, thumb_w = 170, 170
    timeline_top = 300
    row_h = 150
    height = timeline_top + row_h * len(ACTION_NAMES) + 80
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    _put_text(canvas, f"Episode Timeline: {episode_path.name}", 40, 45, 0.95, (20, 20, 20), 2)
    _put_text(canvas, f"steps: {n}", 40, 82, 0.58)
    _put_text(canvas, f"active steps: {int(np.sum(active_any))} ({np.mean(active_any) * 100:.1f}%)", 220, 82, 0.58)

    sample_indices = np.linspace(0, n - 1, 9).astype(int)
    for i, idx in enumerate(sample_indices):
        step = episode[idx]
        image = step["img_global"]
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        x = 40 + i * 195
        y = 105
        cv2.rectangle(canvas, (x - 2, y - 2), (x + thumb_w + 2, y + thumb_h + 2), (80, 80, 80), 1)
        canvas[y : y + thumb_h, x : x + thumb_w] = image
        color = (45, 155, 80) if active_any[idx] else (120, 120, 120)
        _put_text(canvas, f"step {idx}", x, y + thumb_h + 22, 0.43, color, 1)

    left, right = 90, width - 50
    steps = np.arange(n)
    x_vals = (left + steps / max(1, n - 1) * (right - left)).astype(np.int32)
    for i, name in enumerate(ACTION_NAMES):
        row_top = timeline_top + i * row_h
        row_bottom = row_top + row_h - 25
        vals = actions[:, i]
        y_min = float(np.min(vals))
        y_max = float(np.max(vals))
        pad = max(1e-6, (y_max - y_min) * 0.1)
        y_min -= pad
        y_max += pad

        cv2.rectangle(canvas, (left, row_top), (right, row_bottom), (225, 225, 225), 1)
        active_x = x_vals[active_any]
        for xx in active_x[:: max(1, len(active_x) // 450)]:
            cv2.line(canvas, (int(xx), row_top), (int(xx), row_bottom), (230, 245, 230), 1)

        zero_y = int(_scale_series(np.array([0.0]), y_min, y_max, row_top, row_bottom)[0])
        if row_top <= zero_y <= row_bottom:
            cv2.line(canvas, (left, zero_y), (right, zero_y), (205, 205, 205), 1)

        y_vals = _scale_series(vals, y_min, y_max, row_top, row_bottom)
        points = np.stack([x_vals, y_vals], axis=1).reshape((-1, 1, 2))
        cv2.polylines(canvas, [points], False, (50, 90, 205), 2, cv2.LINE_AA)
        _put_text(canvas, name, 25, row_top + 70, 0.66, (30, 30, 30), 2)
        _put_text(canvas, f"{y_max:.3f}", left - 70, row_top + 16, 0.42, (90, 90, 90), 1)
        _put_text(canvas, f"{y_min:.3f}", left - 70, row_bottom, 0.42, (90, 90, 90), 1)

    cv2.imwrite(str(output_path), canvas)


def write_episode_csv(output_path: Path, rows: Sequence[Dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "steps",
        "active_any",
        "active_pct",
        "bad_shapes",
        "left_same_frames",
        "right_same_frames",
        "global_same_frames",
        "left_dark_frames",
        "right_dark_frames",
        "global_dark_frames",
        "left_bright_frames",
        "right_bright_frames",
        "global_bright_frames",
    ]
    for name in ACTION_NAMES:
        fieldnames.extend(
            [
                f"{name}_active",
                f"{name}_active_pct",
                f"{name}_mean_abs",
                f"{name}_pos",
                f"{name}_neg",
            ]
        )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {
                "episode": row["episode"],
                "steps": row["steps"],
                "active_any": row["active_any"],
                "active_pct": row["active_pct"],
                "bad_shapes": row["bad_shapes"],
                "left_same_frames": row["same_frames"]["img_left"],
                "right_same_frames": row["same_frames"]["img_right"],
                "global_same_frames": row["same_frames"]["img_global"],
                "left_dark_frames": row["dark_frames"]["img_left"],
                "right_dark_frames": row["dark_frames"]["img_right"],
                "global_dark_frames": row["dark_frames"]["img_global"],
                "left_bright_frames": row["bright_frames"]["img_left"],
                "right_bright_frames": row["bright_frames"]["img_right"],
                "global_bright_frames": row["bright_frames"]["img_global"],
            }
            for name in ACTION_NAMES:
                flat[f"{name}_active"] = row[f"{name}_active"]
                flat[f"{name}_active_pct"] = row[f"{name}_active_pct"]
                flat[f"{name}_mean_abs"] = row[f"{name}_mean_abs"]
                flat[f"{name}_pos"] = row[f"{name}_pos"]
                flat[f"{name}_neg"] = row[f"{name}_neg"]
            writer.writerow(flat)


def _resolve_episode(files: Sequence[Path], episode: str | None, episode_index: int | None, rows: Sequence[Dict]) -> Path:
    if episode:
        path = Path(episode).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Episode not found: {path}")
        return path
    if episode_index is not None:
        if episode_index < 0 or episode_index >= len(files):
            raise IndexError(f"episode_index={episode_index} outside dataset size {len(files)}")
        return files[episode_index]

    weakest = min(rows, key=lambda row: row["active_pct"])
    return Path(weakest["path"])


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Create visual quality reports for processed Three-Bot episodes.")
    parser.add_argument("--dataset_path", default="training_data/processed_episodes/*.npy")
    parser.add_argument("--output_dir", default="training_outputs/data_quality")
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--episode", default=None, help="Explicit episode path for timeline visualization.")
    parser.add_argument("--episode_index", type=int, default=None, help="Episode index for timeline visualization.")
    parsed = parser.parse_args(args=args)

    files = [Path(path).resolve() for path in sorted(glob.glob(parsed.dataset_path))]
    if not files:
        raise FileNotFoundError(f"No episode files matched: {parsed.dataset_path}")

    output_dir = Path(parsed.output_dir).expanduser().resolve()
    rows, actions = _collect_episode_stats(files, parsed.threshold)
    episode_path = _resolve_episode(files, parsed.episode, parsed.episode_index, rows)

    write_episode_csv(output_dir / "episodes.csv", rows)
    write_dashboard(output_dir / "dataset_dashboard.png", rows, actions, parsed.threshold)
    write_episode_timeline(output_dir / f"{episode_path.stem}_timeline.png", episode_path, parsed.threshold)

    active_any = np.any(np.abs(actions) > parsed.threshold, axis=1)
    print(f"Episodes: {len(rows)}")
    print(f"Steps: {len(actions)}")
    print(f"Active steps: {int(np.sum(active_any))} ({np.mean(active_any) * 100:.2f}%)")
    print(f"All-zero steps: {int(np.sum(~active_any))} ({np.mean(~active_any) * 100:.2f}%)")
    print(f"Dashboard: {output_dir / 'dataset_dashboard.png'}")
    print(f"Episode CSV: {output_dir / 'episodes.csv'}")
    print(f"Timeline: {output_dir / f'{episode_path.stem}_timeline.png'}")


if __name__ == "__main__":
    main()
