import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

# Robust import whether run as a module or as a script
try:
    from .metrics import exact_match, char_accuracy  # type: ignore
except Exception:  # pragma: no cover
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from metrics import exact_match, char_accuracy  # type: ignore


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_predictions(path: str) -> Dict[str, str]:
    """Read TSV/CSV with name<TAB>program (auto detects comma or tab)."""
    preds: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(",", 1)
            if len(parts) < 2:
                continue
            name, prog = parts[0], parts[1]
            preds[name] = prog
    return preds


def evaluate_sequence_accuracy(annotations_dir: str, predictions_path: str, names: List[str]) -> Tuple[float, float]:
    preds = read_predictions(predictions_path)
    em_total = 0.0
    ca_total = 0.0
    n = 0
    for name in names:
        ann_path = os.path.join(annotations_dir, f"{name}.json")
        if not os.path.exists(ann_path) or name not in preds:
            continue
        ann = load_json(ann_path)
        ref = ann.get("program", "")
        hyp = preds.get(name, "")
        em_total += exact_match(ref, hyp)
        ca_total += char_accuracy(ref, hyp)
        n += 1
    em = em_total / n if n > 0 else 0.0
    ca = ca_total / n if n > 0 else 0.0
    return em, ca


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sequence-level program predictions against ground truth.")
    parser.add_argument("--annotations-dir", type=str, required=True)
    parser.add_argument("--splits", type=str, required=True, help="Path to split file (e.g., test.txt)")
    parser.add_argument("--predictions", type=str, required=True, help="TSV/CSV with name and predicted program")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.splits, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    em, ca = evaluate_sequence_accuracy(args.annotations_dir, args.predictions, names)
    print(f"Sequence Exact Match: {em:.4f}")
    print(f"Char-level Accuracy: {ca:.4f}")


if __name__ == "__main__":
    main()


