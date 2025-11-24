import argparse
import json
import os
from collections import Counter
from typing import Dict


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple dataset sanity/evaluation for synthetic primitives.")
    parser.add_argument("--annotations-dir", type=str, required=True, help="Directory with JSON annotations")
    parser.add_argument("--splits", type=str, required=True, help="Path to a split file (train/val/test .txt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.splits, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]

    type_counter: Counter = Counter()
    total = 0
    for name in names:
        ann_path = os.path.join(args.annotations_dir, f"{name}.json")
        ann = load_json(ann_path)
        for p in ann.get("primitives", []):
            type_counter[p.get("type", "?")] += 1
            total += 1

    print("Samples in split:", len(names))
    print("Total primitives:", total)
    print("By type:")
    for t, c in type_counter.most_common():
        print(f"  {t:10s}: {c}")


if __name__ == "__main__":
    main()


