import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    import torchvision
except Exception:  # pragma: no cover
    fasterrcnn_resnet50_fpn = None
    torchvision = None


CLASS_TO_ID = {
    "rectangle": 1,
    "circle": 2,
    "line": 3,
    "arrow": 4,
    "text": 5,
}


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class SketchDataset(Dataset):
    def __init__(self, images_dir: str, annotations_dir: str, split_file: str):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        with open(split_file, "r", encoding="utf-8") as f:
            self.names = [line.strip() for line in f.readlines() if line.strip()]

        self.transforms = None
        if torchvision is not None:
            self.transforms = torchvision.transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img_path_png = os.path.join(self.images_dir, f"{name}.png")
        img_path_jpg = os.path.join(self.images_dir, f"{name}.jpg")
        img_path = img_path_png if os.path.exists(img_path_png) else img_path_jpg
        ann_path = os.path.join(self.annotations_dir, f"{name}.json")
        ann = load_json(ann_path)

        import cv2
        import numpy as np
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        boxes: List[List[float]] = []
        labels: List[int] = []
        for p in ann.get("primitives", []):
            x, y, w, h = [float(v) for v in p.get("bbox", [0, 0, 0, 0])]
            boxes.append([x, y, x + w, y + h])
            labels.append(CLASS_TO_ID.get(p.get("type", ""), 0))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple detector baseline on the synthetic dataset.")
    parser.add_argument("--data-root", type=str, default=os.path.join("..", "dataset", "synthetic"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = os.path.join(args.data_root, "images")
    ann_dir = os.path.join(args.data_root, "annotations")
    split_file = os.path.join(args.data_root, "splits", f"{args.split}.txt")

    if fasterrcnn_resnet50_fpn is None:
        raise SystemExit("torchvision detection models are unavailable. Please install torchvision.")

    dataset = SketchDataset(images_dir, ann_dir, split_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    num_classes = max(CLASS_TO_ID.values()) + 1
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    model.to(args.device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for images, targets in loader:
            images = [img.to(args.device) for img in images]
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += float(losses.detach().cpu())
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {total_loss/ max(1, len(loader)):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", "fasterrcnn_synthetic.pt"))
    print("Saved checkpoint to checkpoints/fasterrcnn_synthetic.pt")


if __name__ == "__main__":
    main()


