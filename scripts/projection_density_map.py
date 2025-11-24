#!/usr/bin/env python3
"""
Aggregate multiple projection renders into a single density heatmap.

By averaging binarized silhouettes we obtain a coarse occupancy prior
that highlights the areas where most building projections contain mass.
This helps select canonical camera poses for future dataset filtering.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory that contains projection .png files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Glob pattern to match inside --images-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Number of files to aggregate (0 = use all).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=768,
        help="Resize longer edge to this size before averaging.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/final_submission/figs/projection_density.png"),
        help="Path to save the generated heatmap.",
    )
    return parser.parse_args()


def _load_grayscale(path: Path) -> np.ndarray:
    img = mpimg.imread(path)
    if img.ndim == 3:
        img = img[:, :, 0] if img.shape[2] == 1 else np.mean(img[:, :, :3], axis=-1)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return img


def load_images(paths: Sequence[Path], target_size: int) -> np.ndarray:
    """Load projections, resize with nearest-neighbor, convert to inverted grayscale."""
    tensors = []
    for path in paths:
        img = _load_grayscale(path)
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        y_idx = np.linspace(0, h - 1, new_h).astype(np.int64)
        x_idx = np.linspace(0, w - 1, new_w).astype(np.int64)
        resized = img[y_idx][:, x_idx]
        canvas = np.zeros((target_size, target_size), dtype=np.float32)
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
        tensors.append(1.0 - canvas / 255.0)  # invert so ink=1, background=0
    return np.stack(tensors, axis=0)


def main() -> None:
    args = parse_args()
    paths = sorted(args.images_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {args.pattern} in {args.images_dir}")
    if args.limit > 0:
        paths = paths[: args.limit]
    volume = load_images(paths, args.size)
    density = volume.mean(axis=0)

    plt.figure(figsize=(6, 6))
    plt.title(f"Projection density from {len(paths)} samples")
    plt.imshow(density, cmap="inferno", origin="upper")
    plt.colorbar(label="Relative occupancy")
    plt.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"Saved density heatmap to {args.output.resolve()}")


if __name__ == "__main__":
    main()

