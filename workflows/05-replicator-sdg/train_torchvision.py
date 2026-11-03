"""Open alternative to the TAO step: fine-tune a torchvision Faster R-CNN on the Replicator KITTI data.

Same data, same 14% per-set hold-out (``kitti_tools.split_by_group``), same metric (single-class AP at IoU 0.5,
``kitti_tools.average_precision``), so the result can sit next to the TAO DetectNet_v2 number. Uses only pip packages,
no NGC account.

    python train_torchvision.py --root data --epochs 10 --out runs/frcnn_mbv3
    python train_torchvision.py --root data --eval-only --checkpoint runs/frcnn_mbv3/model.pt

Pinned: Python 3.11, torch 2.7.0 + torchvision 0.22.0 (CUDA 12.8 wheels), numpy, Pillow.
not executed: needs GPU (CPU works but one epoch over 6,000 frames at 960x544 takes hours).
The pure helpers ``objects_to_arrays`` and ``collate`` are tested on CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import kitti_tools as kt
import numpy as np

CLASSES = ("__background__", "palletjack")  # torchvision reserves label 0 for background


def objects_to_arrays(
    objects: Sequence[kt.KittiObject], class_to_id: Mapping[str, int], min_height: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """KITTI objects -> (boxes float32 (N, 4) as x1, y1, x2, y2; labels int64 (N,)), dropping unknown classes,
    degenerate boxes, and boxes shorter than ``min_height`` pixels."""
    boxes, labels = [], []
    for o in objects:
        if o.cls not in class_to_id or o.width <= 0 or o.height <= 0 or o.height < min_height:
            continue
        boxes.append(o.bbox)
        labels.append(class_to_id[o.cls])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(labels, dtype=np.int64)


def collate(batch):
    """Detection batches are lists (images differ in box count); torchvision's detection models expect that."""
    return tuple(zip(*batch, strict=True))


def make_dataset(samples: Sequence[kt.Sample], class_to_id: Mapping[str, int], train: bool):
    """A torch Dataset over ``samples``. Imports torch lazily so this module loads without it."""
    import torch
    from PIL import Image
    from torchvision.transforms.functional import hflip, to_tensor

    class KittiDetection(torch.utils.data.Dataset):
        def __init__(self):
            self.samples = [s for s in samples if s.image_path is not None]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            s = self.samples[i]
            img = to_tensor(Image.open(s.image_path).convert("RGB"))
            boxes, labels = objects_to_arrays(kt.read_labels(s.label_path), class_to_id)
            boxes_t = torch.from_numpy(boxes)
            if train and torch.rand(()) < 0.5:  # horizontal flip, as in the TAO spec (hflip_probability 0.5)
                img = hflip(img)
                w = img.shape[-1]
                boxes_t = torch.stack([w - boxes_t[:, 2], boxes_t[:, 1], w - boxes_t[:, 0], boxes_t[:, 3]], dim=1)
            return img, {"boxes": boxes_t, "labels": torch.from_numpy(labels), "frame": f"{s.group}/{s.stem}"}

    return KittiDetection()


def build_model(num_classes: int):
    """Faster R-CNN with a MobileNetV3-Large FPN backbone, COCO-pretrained, new box head for ``num_classes``."""
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def evaluate(model, loader, device) -> float:
    """Single-class AP@0.5 over a loader, with the same function used for the dataset tools."""
    import torch

    model.eval()
    dets, gts = {}, {}
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([im.to(device) for im in images])
            for out, tgt in zip(outputs, targets, strict=True):
                dets[tgt["frame"]] = (out["boxes"].cpu().numpy(), out["scores"].cpu().numpy())
                gts[tgt["frame"]] = tgt["boxes"].numpy()
    return kt.average_precision(dets, gts, iou_threshold=0.5)


def main(argv: Sequence[str] | None = None) -> int:  # not executed: needs GPU
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--sets", nargs="+", default=list(kt.DEFAULT_SETS))
    ap.add_argument("--val", type=float, default=0.14)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/frcnn_mbv3"))
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=None)
    args = ap.parse_args(argv)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_id = {name: i for i, name in enumerate(CLASSES) if i > 0}
    parts = kt.split_by_group(kt.pair_dataset(args.root, args.sets), args.val)
    train_ds = make_dataset(parts["train"], class_to_id, train=True)
    val_ds = make_dataset(parts["val"], class_to_id, train=False)
    loader_kw = {"batch_size": args.batch, "num_workers": args.workers, "collate_fn": collate}
    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kw)
    val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kw)

    model = build_model(num_classes=len(CLASSES)).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    if not args.eval_only:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        for epoch in range(args.epochs):
            model.train()
            t0, total = time.time(), 0.0
            for batch_images, batch_targets in train_dl:
                images = [im.to(device) for im in batch_images]
                targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in batch_targets]
                loss = sum(model(images, targets).values())
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss)
            sched.step()
            ap50 = evaluate(model, val_dl, device)
            history.append({"epoch": epoch, "loss": total / max(len(train_dl), 1), "val_ap50": ap50})
            print(f"epoch {epoch}: loss {history[-1]['loss']:.4f}  val AP@0.5 {ap50:.4f}  ({time.time() - t0:.0f} s)")
            torch.save(model.state_dict(), args.out / "model.pt")
    ap50 = evaluate(model, val_dl, device)
    metrics = {"val_ap50": ap50, "num_train": len(train_ds), "num_val": len(val_ds), "history": history}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
