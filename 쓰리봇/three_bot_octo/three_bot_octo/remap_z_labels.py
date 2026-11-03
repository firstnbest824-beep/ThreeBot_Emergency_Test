"""Remap Three-Bot Z action labels from STM status codes to signed actions."""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import numpy as np


def _format_pct(count: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{count / total * 100.0:.2f}%"


def _load_episode(path: Path) -> np.ndarray:
    episode = np.load(path, allow_pickle=True)
    if len(episode) == 0:
        raise ValueError(f"empty episode: {path}")
    if "action" not in episode[0]:
        raise ValueError(f"episode has no action field: {path}")
    return episode


def _remap_episode(
    episode: np.ndarray,
    *,
    zero_code: float,
    up_code: float,
    down_code: float,
    up_value: float,
    down_value: float,
    atol: float,
) -> tuple[np.ndarray, dict[str, int]]:
    remapped_steps = []
    counts = {
        "zero": 0,
        "up": 0,
        "down": 0,
        "unknown": 0,
    }

    for step in episode:
        new_step = dict(step)
        action = np.array(step["action"], dtype=np.float32, copy=True)
        z = float(action[4])

        if np.isclose(z, zero_code, atol=atol):
            action[4] = 0.0
            counts["zero"] += 1
        elif np.isclose(z, up_code, atol=atol):
            action[4] = up_value
            counts["up"] += 1
        elif np.isclose(z, down_code, atol=atol):
            action[4] = down_value
            counts["down"] += 1
        else:
            counts["unknown"] += 1

        new_step["action"] = action
        remapped_steps.append(new_step)

    return np.array(remapped_steps, dtype=object), counts


def remap_dataset(args: argparse.Namespace) -> None:
    files = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not files:
        raise FileNotFoundError(f"no input files matched: {args.input_glob}")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite and not args.dry_run:
            raise FileExistsError(
                f"output dir is not empty: {output_dir}. "
                "Use --overwrite or choose a new --output-dir."
            )
    elif not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    total = {"steps": 0, "zero": 0, "up": 0, "down": 0, "unknown": 0}

    for src in files:
        episode = _load_episode(src)
        remapped, counts = _remap_episode(
            episode,
            zero_code=args.zero_code,
            up_code=args.up_code,
            down_code=args.down_code,
            up_value=args.up_value,
            down_value=args.down_value,
            atol=args.atol,
        )

        steps = len(episode)
        total["steps"] += steps
        for key in ("zero", "up", "down", "unknown"):
            total[key] += counts[key]

        if not args.dry_run:
            dst = output_dir / src.name
            if dst.exists() and args.overwrite:
                dst.unlink()
            np.save(dst, remapped, allow_pickle=True)

    if not args.dry_run:
        readme = output_dir / "Z_LABEL_REMAP.txt"
        readme.write_text(
            "\n".join(
                [
                    "Z label remap generated from processed_episodes.",
                    f"input_glob: {args.input_glob}",
                    f"zero_code {args.zero_code} -> 0.0",
                    f"up_code {args.up_code} -> {args.up_value}",
                    f"down_code {args.down_code} -> {args.down_value}",
                    f"atol: {args.atol}",
                    "",
                    "Counts:",
                    f"steps: {total['steps']}",
                    f"zero: {total['zero']}",
                    f"up: {total['up']}",
                    f"down: {total['down']}",
                    f"unknown: {total['unknown']}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    print(f"episodes: {len(files)}")
    print(f"steps: {total['steps']}")
    print(f"zero {args.zero_code} -> 0.0: {total['zero']} ({_format_pct(total['zero'], total['steps'])})")
    print(f"up {args.up_code} -> {args.up_value}: {total['up']} ({_format_pct(total['up'], total['steps'])})")
    print(
        f"down {args.down_code} -> {args.down_value}: "
        f"{total['down']} ({_format_pct(total['down'], total['steps'])})"
    )
    print(f"unknown z values left unchanged: {total['unknown']}")
    if args.dry_run:
        print("dry run only: no files written")
    else:
        print(f"wrote remapped dataset: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remap Z labels 0.00/0.01/0.02 into signed Z actions."
    )
    parser.add_argument(
        "--input-glob",
        default="training_data/processed_episodes/*.npy",
        help="Input processed episode glob.",
    )
    parser.add_argument(
        "--output-dir",
        default="training_data/processed_episodes_z_signed",
        help="Output directory for remapped episodes.",
    )
    parser.add_argument("--zero-code", type=float, default=0.0)
    parser.add_argument("--up-code", type=float, default=0.01)
    parser.add_argument("--down-code", type=float, default=0.02)
    parser.add_argument("--up-value", type=float, default=1.0)
    parser.add_argument("--down-value", type=float, default=-1.0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    remap_dataset(parser.parse_args())


if __name__ == "__main__":
    main()
