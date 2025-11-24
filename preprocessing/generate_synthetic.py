import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class Primitive:
    type: str
    bbox: Tuple[int, int, int, int]
    value: str = ""

    def to_dict(self) -> Dict:
        data = {"type": self.type, "bbox": list(map(int, self.bbox))}
        if self.value:
            data["value"] = self.value
        return data


def clamp_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(x))))


def draw_rectangle(img: np.ndarray, rng: random.Random) -> Primitive:
    h, w = img.shape[:2]
    x1 = rng.randint(int(w * 0.05), int(w * 0.6))
    y1 = rng.randint(int(h * 0.05), int(h * 0.6))
    x2 = rng.randint(x1 + int(w * 0.1), min(w - 5, x1 + int(w * 0.6)))
    y2 = rng.randint(y1 + int(h * 0.1), min(h - 5, y1 + int(h * 0.6)))
    color = (0, 0, 0)
    thickness = rng.choice([1, 2, 3])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    bbox = (x1, y1, x2 - x1, y2 - y1)

    width_val = x2 - x1
    mid_y = y2 + 10
    cv2.arrowedLine(img, (x1, mid_y), (x2, mid_y), color, 1, tipLength=0.02)
    cv2.putText(
        img,
        f"{width_val}",
        (x1 + (width_val // 2) - 10, mid_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )

    height_val = y2 - y1
    mid_x = x2 + 10
    cv2.arrowedLine(img, (mid_x, y1), (mid_x, y2), color, 1, tipLength=0.02)
    cv2.putText(
        img, f"{height_val}", (mid_x + 3, y1 + (height_val // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
    )

    return Primitive("rectangle", bbox)


def draw_circle(img: np.ndarray, rng: random.Random) -> Primitive:
    h, w = img.shape[:2]
    r = rng.randint(int(min(w, h) * 0.05), int(min(w, h) * 0.15))
    cx = rng.randint(r + 5, w - r - 5)
    cy = rng.randint(r + 5, h - r - 5)
    color = (0, 0, 0)
    thickness = rng.choice([1, 2, 3])
    cv2.circle(img, (cx, cy), r, color, thickness)

    bbox = (cx - r, cy - r, 2 * r, 2 * r)

    # Radius annotation
    angle = rng.uniform(0, 2 * math.pi)
    ex = clamp_int(cx + r * math.cos(angle), 0, w - 1)
    ey = clamp_int(cy + r * math.sin(angle), 0, h - 1)
    cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 1, tipLength=0.02)
    label_pos = (ex + 5, ey + 5)
    cv2.putText(img, f"R={r}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return Primitive("circle", bbox, value=str(r))


def draw_line(img: np.ndarray, rng: random.Random) -> Primitive:
    h, w = img.shape[:2]
    x1 = rng.randint(5, w - 20)
    y1 = rng.randint(5, h - 20)
    x2 = clamp_int(x1 + rng.randint(-w // 3, w // 3), 5, w - 5)
    y2 = clamp_int(y1 + rng.randint(-h // 3, h // 3), 5, h - 5)
    color = (0, 0, 0)
    thickness = rng.choice([1, 2])
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    # Length annotation
    length = int(math.hypot(x2 - x1, y2 - y1))
    label_x = (x1 + x2) // 2
    label_y = (y1 + y2) // 2
    cv2.putText(img, f"L={length}", (label_x + 5, label_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return Primitive("line", bbox, value=str(length))


def draw_arrow(img: np.ndarray, rng: random.Random) -> Primitive:
    h, w = img.shape[:2]
    x1 = rng.randint(10, w - 40)
    y1 = rng.randint(10, h - 40)
    x2 = clamp_int(x1 + rng.randint(-w // 4, w // 4), 10, w - 10)
    y2 = clamp_int(y1 + rng.randint(-h // 4, h // 4), 10, h - 10)
    color = (0, 0, 0)
    thickness = rng.choice([1, 2])
    cv2.arrowedLine(img, (x1, y1), (x2, y2), color, thickness, tipLength=0.15)

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    # Store coarse direction as value
    dx, dy = x2 - x1, y2 - y1
    direction = "right" if abs(dx) >= abs(dy) and dx >= 0 else (
        "left" if abs(dx) >= abs(dy) else ("down" if dy >= 0 else "up")
    )
    return Primitive("arrow", bbox, value=direction)


def draw_text_token(img: np.ndarray, rng: random.Random) -> Primitive:
    h, w = img.shape[:2]
    value = str(rng.randint(1, 999))
    x = rng.randint(5, max(5, w - 60))
    y = rng.randint(15, max(15, h - 5))
    color = (0, 0, 0)
    font_scale = rng.choice([0.5, 0.6, 0.7])
    thickness = rng.choice([1, 2])
    cv2.putText(img, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    # Estimate bbox via text size
    (tw, th), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    bbox = (x, y - th, tw, th)
    return Primitive("text", bbox, value=value)


PRIMITIVE_DRAWERS = [draw_rectangle, draw_circle, draw_line, draw_arrow, draw_text_token]


def generate_sample(img_size: int, rng: random.Random) -> Tuple[np.ndarray, List[Primitive]]:
    img = np.full((img_size, img_size, 3), 255, np.uint8)
    primitives: List[Primitive] = []

    noise_strength = 6
    noise = rng.random() < 0.7
    if noise:
        n = np.random.default_rng(rng.randint(0, 1 << 30)).integers(0, noise_strength, size=img.shape, dtype=np.uint8)
        img = cv2.subtract(img, n)

    count = rng.randint(3, 7)
    for _ in range(count):
        drawer = rng.choice(PRIMITIVE_DRAWERS)
        primitives.append(drawer(img, rng))

    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), sigmaX=0.8)

    return img, primitives


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _serialize_program(primitives: List[Primitive]) -> str:
    tokens: List[str] = []
    for p in primitives:
        x, y, bw, bh = p.bbox
        if p.type == "rectangle":
            tokens.append(f"RECT {x} {y} {bw} {bh}")
        elif p.type == "circle":
            tokens.append(f"CIRC {x} {y} {bw} {bh} R={p.value}")
        elif p.type == "line":
            tokens.append(f"LINE {x} {y} {bw} {bh} L={p.value}")
        elif p.type == "arrow":
            tokens.append(f"ARROW {x} {y} {bw} {bh} DIR={p.value}")
        elif p.type == "text":
            tokens.append(f"TEXT '{p.value}' {x} {y} {bw} {bh}")
    return "; ".join(tokens)


def write_annotation(json_path: str, image_name: str, w: int, h: int, primitives: List[Primitive]) -> None:
    data = {
        "image": image_name,
        "width": int(w),
        "height": int(h),
        "primitives": [p.to_dict() for p in primitives],
        "program": _serialize_program(primitives),
        "ocr_gt": [p.value for p in primitives if p.type == "text" and p.value],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_splits(
    all_names: List[str], out_splits_dir: str, rng: random.Random, train_ratio: float, val_ratio: float
) -> None:
    ensure_dir(out_splits_dir)
    names = all_names[:]
    rng.shuffle(names)
    n = len(names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = names[:n_train]
    val = names[n_train : n_train + n_val]
    test = names[n_train + n_val :]

    def _dump(lst: List[str], name: str) -> None:
        with open(os.path.join(out_splits_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            for s in lst:
                f.write(s + "\n")

    _dump(train, "train")
    _dump(val, "val")
    _dump(test, "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic sketch dataset with simple primitives and annotations."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("..", "dataset", "synthetic"),
        help="Output root directory for images/annotations/splits",
    )
    parser.add_argument("--num-samples", type=int, default=200, help="Number of images to generate")
    parser.add_argument("--image-size", type=int, default=512, help="Square output image size in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    images_dir = os.path.join(args.output_dir, "images")
    ann_dir = os.path.join(args.output_dir, "annotations")
    splits_dir = os.path.join(args.output_dir, "splits")
    ensure_dir(images_dir)
    ensure_dir(ann_dir)

    names: List[str] = []
    for idx in range(args.num_samples):
        img, primitives = generate_sample(args.image_size, rng)
        img_name = f"sketch_{idx:05d}.png"
        ann_name = f"sketch_{idx:05d}.json"
        cv2.imwrite(os.path.join(images_dir, img_name), img)
        write_annotation(os.path.join(ann_dir, ann_name), img_name, args.image_size, args.image_size, primitives)
        names.append(os.path.splitext(img_name)[0])

    write_splits(names, splits_dir, rng, args.train_ratio, args.val_ratio)

    print(f"Generated {args.num_samples} samples to {args.output_dir}")
    print(f"- images:       {images_dir}")
    print(f"- annotations:  {ann_dir}")
    print(f"- splits:       {splits_dir}")


if __name__ == "__main__":
    main()
