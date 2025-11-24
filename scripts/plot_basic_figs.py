import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_split(split_path: str) -> List[str]:
    with open(split_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def draw_title(draw: ImageDraw.ImageDraw, title: str, width: int) -> None:
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    w, h = draw.textlength(title, font=font), 12
    draw.text(((width - w) / 2, 10), title, fill=(0, 0, 0), font=font)


def save_bar_chart(labels: List[str], values: List[int], title: str, out_path: str, x_label: str = "", y_label: str = "") -> None:
    width, height = 800, 500
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_title(draw, title, width)
    margin = 60
    top = 50
    bottom = height - margin
    left = margin
    right = width - margin
    # axes
    draw.line((left, bottom, right, bottom), fill=(0, 0, 0), width=2)
    draw.line((left, bottom, left, top), fill=(0, 0, 0), width=2)
    if y_label:
        draw.text((10, (top + bottom) / 2), y_label, fill=(0, 0, 0))
    if x_label:
        xl_w = draw.textlength(x_label)
        draw.text(((left + right - xl_w) / 2, bottom + 25), x_label, fill=(0, 0, 0))
    n = max(1, len(values))
    max_val = max(1, max(values) if values else 1)
    bar_w = (right - left) / (n * 1.5)
    gap = bar_w / 2
    for i, v in enumerate(values):
        x0 = left + i * (bar_w + gap) + gap
        x1 = x0 + bar_w
        h = (v / max_val) * (bottom - top)
        y0 = bottom - h
        color = (76, 120, 168)
        draw.rectangle((x0, y0, x1, bottom - 1), fill=color, outline=(0, 0, 0))
        # label
        lbl = labels[i]
        if len(lbl) > 10:
            lbl = lbl[:9] + "…"
        draw.text((x0, bottom + 5), lbl, fill=(0, 0, 0))
        draw.text((x0, y0 - 12), str(v), fill=(0, 0, 0))
    img.save(out_path)


def save_histogram(values: List[int], bins: int, title: str, out_path: str, x_label: str = "", y_label: str = "") -> None:
    width, height = 800, 500
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_title(draw, title, width)
    margin = 60
    top = 50
    bottom = height - margin
    left = margin
    right = width - margin
    draw.line((left, bottom, right, bottom), fill=(0, 0, 0), width=2)
    draw.line((left, bottom, left, top), fill=(0, 0, 0), width=2)
    if y_label:
        draw.text((10, (top + bottom) / 2), y_label, fill=(0, 0, 0))
    if x_label:
        xl_w = draw.textlength(x_label)
        draw.text(((left + right - xl_w) / 2, bottom + 25), x_label, fill=(0, 0, 0))
    if not values:
        values = [0]
    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        vmax = vmin + 1
    bin_edges = [vmin + (vmax - vmin) * i / bins for i in range(bins + 1)]
    counts = [0 for _ in range(bins)]
    for v in values:
        idx = min(bins - 1, int((v - vmin) / (vmax - vmin) * bins))
        counts[idx] += 1
    max_count = max(1, max(counts))
    bar_w = (right - left) / bins
    for i, c in enumerate(counts):
        x0 = left + i * bar_w + 1
        x1 = left + (i + 1) * bar_w - 1
        h = (c / max_count) * (bottom - top)
        y0 = int(bottom - h)
        y0 = max(top, min(y0, bottom - 1))
        color = (229, 87, 86)
        draw.rectangle((x0, y0, x1, bottom - 1), fill=color, outline=(0, 0, 0))

    # X-axis ticks and labels
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # Show up to 10 x-ticks
    num_xticks = min(10, bins)
    if num_xticks > 1:
        step = max(1, bins // num_xticks)
        for i in range(0, bins, step):
            # tick position at bin center
            cx = left + (i + 0.5) * bar_w
            draw.line((cx, bottom, cx, bottom + 5), fill=(0, 0, 0), width=1)
            # label value ~ midpoint of bin
            edge_l = bin_edges[i]
            edge_r = bin_edges[i + 1]
            val = (edge_l + edge_r) / 2
            # if range is small integers, round to int
            lbl = str(int(round(val))) if (vmax - vmin) <= 50 else f"{val:.1f}"
            tw = draw.textlength(lbl, font=font)
            draw.text((cx - tw / 2, bottom + 8), lbl, fill=(0, 0, 0), font=font)

    # Y-axis ticks and labels (4 ticks)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = bottom - frac * (bottom - top)
        draw.line((left - 5, y, left, y), fill=(0, 0, 0), width=1)
        lbl = str(int(round(frac * max_count)))
        draw.text((left - 40, y - 6), lbl, fill=(0, 0, 0), font=font)
    img.save(out_path)


def gather_stats(annotations_dir: str, names: List[str]) -> Tuple[List[str], List[int], List[int], List[float], List[int], List[int], List[float]]:
    type_counter: Counter = Counter()
    counts_per_image: List[int] = []
    bbox_area_norm: List[float] = []
    ocr_token_lengths: List[int] = []
    program_statement_counts: List[int] = []
    aspect_ratios: List[float] = []
    for n in names:
        data = load_json(os.path.join(annotations_dir, f"{n}.json"))
        prims = data.get("primitives", [])
        counts_per_image.append(len(prims))
        w = max(1, int(data.get("width", 1)))
        h = max(1, int(data.get("height", 1)))
        img_area = float(w * h)
        
        # OCR tokens
        ocr_gt = data.get("ocr_gt", [])
        for token in ocr_gt:
            ocr_token_lengths.append(len(str(token)))
        
        # Program statements
        program = str(data.get("program", ""))
        stmts = [s for s in program.split(";") if s.strip()]
        program_statement_counts.append(len(stmts))
        
        for p in prims:
            t = p.get("type", "?")
            type_counter[t] += 1
            x, y, bw, bh = p.get("bbox", [0, 0, 0, 0])
            bbox_area_norm.append(max(0.0, float(bw * bh) / img_area))
            # Aspect ratio
            bw = max(1, int(bw))
            bh = max(1, int(bh))
            aspect_ratios.append(float(bw) / float(bh))
    
    labels = list(type_counter.keys())
    values = [type_counter[k] for k in labels]
    counts_int = [int(c) for c in counts_per_image]
    area_scaled = [int(a * 1000) for a in bbox_area_norm]  # scale for integer hist
    aspect_scaled = [int(a * 100) for a in aspect_ratios]  # scale for integer hist
    return labels, values, counts_int, area_scaled, ocr_token_lengths, program_statement_counts, aspect_scaled


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate basic figures without matplotlib.")
    p.add_argument("--annotations-dir", required=True, type=str)
    p.add_argument("--split", required=True, type=str)
    p.add_argument("--out-dir", default=os.path.join("docs", "figs"), type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    names = read_split(args.split)
    ensure_dir(args.out_dir)
    labels, values, counts, area_norm, ocr_lengths, prog_stmts, aspect_ratios = gather_stats(args.annotations_dir, names)
    
    save_bar_chart(labels, values, "Primitive Types Distribution", os.path.join(args.out_dir, "primitive_types.png"), x_label="type", y_label="count")
    save_histogram(counts, bins=max(5, min(20, max(counts) + 1 if counts else 5)), title="# Primitives per Image", out_path=os.path.join(args.out_dir, "primitives_per_image.png"), x_label="# primitives", y_label="# images")
    save_histogram(area_norm, bins=20, title="BBox Area Distribution (normalized x1000)", out_path=os.path.join(args.out_dir, "bbox_area_hist.png"), x_label="bbox_area / image_area (x1e-3)", y_label="count")
    
    if ocr_lengths:
        save_histogram(ocr_lengths, bins=max(5, min(20, max(ocr_lengths) + 1 if ocr_lengths else 5)), title="OCR Token Lengths", out_path=os.path.join(args.out_dir, "ocr_token_lengths.png"), x_label="characters per token", y_label="count")
    
    if prog_stmts:
        save_histogram(prog_stmts, bins=max(5, min(20, max(prog_stmts) + 1 if prog_stmts else 5)), title="Program Statements per Image", out_path=os.path.join(args.out_dir, "program_statements.png"), x_label="# statements", y_label="# images")
    
    if aspect_ratios:
        save_histogram(aspect_ratios, bins=20, title="BBox Aspect Ratio", out_path=os.path.join(args.out_dir, "bbox_aspect_ratio.png"), x_label="w/h (x100)", y_label="count")
    
    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()


