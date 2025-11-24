import argparse
import json
import os
from typing import Dict, List

import cv2

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ocr_image(image_path: str) -> List[str]:
    if pytesseract is None:
        return []
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    text = pytesseract.image_to_string(img)
    tokens = [t for t in text.replace("\n", " ").split(" ") if t.strip()]
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text tokens from images via OCR and store alongside annotations.")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--annotations-dir", type=str, required=True, help="Directory with JSON annotations")
    parser.add_argument("--output-dir", type=str, default="", help="Optional separate output directory for augmented annotations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [os.path.splitext(f)[0] for f in os.listdir(args.images_dir) if f.endswith(".png") or f.endswith(".jpg")]
    out_dir = args.output_dir if args.output_dir else args.annotations_dir
    os.makedirs(out_dir, exist_ok=True)

    for name in names:
        img_path = os.path.join(args.images_dir, f"{name}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(args.images_dir, f"{name}.jpg")
        ann_path = os.path.join(args.annotations_dir, f"{name}.json")
        if not os.path.exists(ann_path):
            continue
        ann = load_json(ann_path)
        pred_tokens = ocr_image(img_path)
        ann["ocr_pred"] = pred_tokens
        save_json(os.path.join(out_dir, f"{name}.json"), ann)
        print(f"OCR tokens for {name}: {len(pred_tokens)}")


if __name__ == "__main__":
    main()


