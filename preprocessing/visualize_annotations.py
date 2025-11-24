import argparse
import json
import os
from typing import Dict, List

import cv2


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def visualize(image_path: str, ann_path: str, out_path: str) -> None:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    ann = load_json(ann_path)
    primitives: List[Dict] = ann.get("primitives", [])

    for p in primitives:
        t = p.get("type", "?")
        bbox = p.get("bbox", [0, 0, 0, 0])
        val = p.get("value", "")
        x, y, w, h = map(int, bbox)
        if t == "rectangle":
            color = (0, 128, 255)
        elif t == "circle":
            color = (0, 180, 0)
        elif t == "line":
            color = (180, 0, 0)
        elif t == "arrow":
            color = (120, 0, 200)
        elif t == "text":
            color = (0, 0, 0)
        else:
            color = (100, 100, 100)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = t if not val else f"{t}:{val}"
        cv2.putText(img, label, (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize annotations on top of images.")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--annotations-dir", type=str, required=True, help="Directory with JSON annotations")
    parser.add_argument("--name", type=str, default="", help="Name without extension (e.g., sketch_00012)")
    parser.add_argument("--output", type=str, default=os.path.join("..", "dataset", "synthetic", "vis"),
                        help="Output directory for visualizations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.name:
        raise SystemExit("--name is required (e.g., sketch_00012)")

    img_path = os.path.join(args.images_dir, f"{args.name}.png")
    ann_path = os.path.join(args.annotations_dir, f"{args.name}.json")
    out_path = os.path.join(args.output, f"{args.name}_vis.png")

    visualize(img_path, ann_path, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


