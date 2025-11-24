#!/usr/bin/env python3
"""
Compose a quick-look gallery of multi-view CAD projections.

The script scans a folder with already rendered XY/XZ/YZ composites
and arranges a fixed number of samples into a grid. This is handy for
spot-checking silhouettes and sharing qualitative progress in reports.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Sequence

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
        "--rows",
        type=int,
        default=2,
        help="Rows in the resulting gallery.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Columns in the resulting gallery.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Square size (in px) for each tile after resizing.",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=12,
        help="Padding (in px) between tiles.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to make selection reproducible.",
    )
    parser.add_argument(
        "--output",
       type=Path,
        default=Path("docs/final_submission/figs/projection_gallery.png"),
        help="Where to save the gallery image.",
    )
    return parser.parse_args()


def pick_images(all_paths: Sequence[Path], rows: int, cols: int, seed: int) -> List[Path]:
    """Return a deterministic subset of images."""
    needed = rows * cols
    if len(all_paths) < needed:
        raise ValueError(f"Need at least {needed} files, but found {len(all_paths)}")
    random.Random(seed).shuffle(all_paths)
    return list(all_paths[:needed])


def _load_image(path: Path) -> np.ndarray:
    img = mpimg.imread(path)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def _resize_keep_aspect(img: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = tile_size / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    y_idx = np.linspace(0, h - 1, new_h).astype(np.int64)
    x_idx = np.linspace(0, w - 1, new_w).astype(np.int64)
    resized = img[y_idx][:, x_idx]
    canvas = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
    offset_x = (tile_size - new_w) // 2
    offset_y = (tile_size - new_h) // 2
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
    return canvas


def load_and_prepare(path: Path, tile_size: int) -> np.ndarray:
    """Load an image, resize with nearest-neighbor, and letterbox to a square tile."""
    img = _load_image(path)
    return _resize_keep_aspect(img, tile_size)


def compose_grid(images: Sequence[np.ndarray], rows: int, cols: int, pad: int) -> np.ndarray:
    tile_h, tile_w = images[0].shape[:2]
    height = rows * tile_h + (rows - 1) * pad
    width = cols * tile_w + (cols - 1) * pad
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            y = r * (tile_h + pad)
            x = c * (tile_w + pad)
            canvas[y : y + tile_h, x : x + tile_w] = images[idx]
            idx += 1
    return canvas


def main() -> None:
    args = parse_args()
    paths = sorted(args.images_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {args.pattern} in {args.images_dir}")
    chosen = pick_images(paths, args.rows, args.cols, args.seed)
    tiles = [load_and_prepare(p, args.tile_size) for p in chosen]
    gallery = compose_grid(tiles, args.rows, args.cols, args.pad)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(args.output, gallery)
    print(f"Saved gallery to {args.output.resolve()} from {[p.name for p in chosen]}")


if __name__ == "__main__":
    main()

