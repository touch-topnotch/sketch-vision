from typing import Iterable, Tuple


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two axis-aligned boxes (x, y, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h

    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)

    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter) / float(denom)



def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def exact_match(ref: str, hyp: str) -> float:
    return 1.0 if (ref or "") == (hyp or "") else 0.0


def char_accuracy(ref: str, hyp: str) -> float:
    ref = ref or ""
    hyp = hyp or ""
    if not ref and not hyp:
        return 1.0
    n = max(len(ref), len(hyp))
    correct = sum(1 for a, b in zip(ref, hyp) if a == b)
    return correct / n if n > 0 else 0.0

