"""KITTI label tools for the Replicator pallet-jack dataset: parse, check, split, report, score.

Omniverse Basic Training 05 writes detection labels with Replicator's ``KittiWriter``; Basic Training 06 trains TAO
DetectNet_v2 on them and evaluates "on 14% held-out samples" (transcript 40:04-40:09). This module does the dataset
bookkeeping around that, on a CPU, with no NVIDIA software:

- ``parse_line`` / ``read_labels``: the 15-field KITTI object format (16 with a detector score).
- ``pair_dataset``: match ``rgb/*.png`` with ``object_detection/*.txt`` per distractor set.
- ``dataset_stats`` / ``class_balance_report``: counts, empty frames, box sizes, degenerate and out-of-frame boxes,
  boxes below TAO's minimum evaluation height.
- ``train_val_split`` / ``split_by_group``: deterministic 14% hold-out, applied per distractor set so validation
  contains warehouse, additional, and no-distractor frames in proportion.
- ``iou`` / ``average_precision``: single-class AP at IoU 0.5 for the torchvision trainer.

CLI:
    python kitti_tools.py stats --root data --sets distractors_warehouse distractors_additional no_distractors
    python kitti_tools.py split --root data --sets ... --val 0.14 --out splits/

Executed here (tests/test_kitti_tools.py, synthetic label files).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KITTI_FIELDS = (
    "type",
    "truncated",
    "occluded",
    "alpha",
    "bbox_left",
    "bbox_top",
    "bbox_right",
    "bbox_bottom",
    "dim_height",
    "dim_width",
    "dim_length",
    "loc_x",
    "loc_y",
    "loc_z",
    "rotation_y",
)
DEFAULT_SETS = ("distractors_warehouse", "distractors_additional", "no_distractors")
TAO_MIN_EVAL_HEIGHT = 20  # px; evaluation_box_config minimum_height in tao/detectnet_v2_train_resnet18_kitti.txt


@dataclass(frozen=True)
class KittiObject:
    cls: str
    truncated: float
    occluded: int
    alpha: float
    bbox: tuple[float, float, float, float]  # left, top, right, bottom in pixels
    dimensions: tuple[float, float, float]  # height, width, length in metres (0 for 2D-only labels)
    location: tuple[float, float, float]
    rotation_y: float
    score: float | None = None

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def to_line(self) -> str:
        vals = [
            self.cls,
            f"{self.truncated:.2f}",
            str(self.occluded),
            f"{self.alpha:.2f}",
            *(f"{v:.2f}" for v in self.bbox),
            *(f"{v:.2f}" for v in self.dimensions),
            *(f"{v:.2f}" for v in self.location),
            f"{self.rotation_y:.2f}",
        ]
        if self.score is not None:
            vals.append(f"{self.score:.4f}")
        return " ".join(vals)


def parse_line(line: str) -> KittiObject:
    """Parse one KITTI object line. Class names may not contain spaces (KITTI convention)."""
    parts = line.split()
    if len(parts) not in (15, 16):
        raise ValueError(f"expected 15 or 16 fields, got {len(parts)}: {line!r}")
    try:
        nums = [float(p) for p in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"non-numeric field in {line!r}") from exc
    return KittiObject(
        cls=parts[0],
        truncated=nums[0],
        occluded=round(nums[1]),
        alpha=nums[2],
        bbox=(nums[3], nums[4], nums[5], nums[6]),
        dimensions=(nums[7], nums[8], nums[9]),
        location=(nums[10], nums[11], nums[12]),
        rotation_y=nums[13],
        score=nums[14] if len(nums) == 15 else None,
    )


def read_labels(path: Path | str) -> list[KittiObject]:
    """All objects in one label file; blank lines are ignored. Errors name the file and line."""
    path = Path(path)
    objects = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            objects.append(parse_line(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return objects


@dataclass(frozen=True)
class Sample:
    group: str  # distractor set
    stem: str
    label_path: Path
    image_path: Path | None


def pair_dataset(
    root: Path | str,
    sets: Sequence[str] = DEFAULT_SETS,
    *,
    camera: str = "Camera",
    image_dir: str = "rgb",
    label_dir: str = "object_detection",
    image_ext: str = ".png",
) -> list[Sample]:
    """Samples under ``root/<set>/<camera>/{rgb,object_detection}``, the layout Basic Training 05 shows (33:07-34:28).

    A label without an image keeps ``image_path=None`` so ``dataset_stats`` can report it.
    """
    root = Path(root)
    samples = []
    for group in sets:
        lbl_dir = root / group / camera / label_dir
        img_dir = root / group / camera / image_dir
        if not lbl_dir.is_dir():
            raise FileNotFoundError(f"missing label folder {lbl_dir}")
        for label in sorted(lbl_dir.glob("*.txt")):
            img = img_dir / f"{label.stem}{image_ext}"
            samples.append(Sample(group, label.stem, label, img if img.exists() else None))
    return samples


def dataset_stats(
    labels: Mapping[str, Sequence[KittiObject]],
    image_size: tuple[int, int] | None = None,
    min_height: float = TAO_MIN_EVAL_HEIGHT,
) -> dict:
    """Summary statistics over ``{frame_id: objects}``. ``image_size`` is (width, height) for bounds checks."""
    per_class: Counter[str] = Counter()
    widths, heights, per_image = [], [], []
    degenerate = out_of_bounds = small = 0
    for objs in labels.values():
        per_image.append(len(objs))
        for o in objs:
            per_class[o.cls] += 1
            widths.append(o.width)
            heights.append(o.height)
            if o.width <= 0 or o.height <= 0:
                degenerate += 1
            elif o.height < min_height:
                small += 1
            if image_size is not None:
                w, h = image_size
                left, top, right, bottom = o.bbox
                if left < 0 or top < 0 or right > w or bottom > h:
                    out_of_bounds += 1

    def pct(values: list[float]) -> dict:
        if not values:
            return {}
        arr = np.asarray(values, dtype=float)
        return {k: round(float(np.percentile(arr, q)), 2) for k, q in (("p5", 5), ("p50", 50), ("p95", 95))}

    n_img = len(per_image)
    return {
        "num_images": n_img,
        "num_empty_images": sum(1 for n in per_image if n == 0),
        "num_objects": int(sum(per_image)),
        "objects_per_image_mean": round(float(np.mean(per_image)), 3) if n_img else 0.0,
        "objects_per_image_max": int(max(per_image)) if n_img else 0,
        "per_class": dict(sorted(per_class.items())),
        "bbox_width_px": pct(widths),
        "bbox_height_px": pct(heights),
        "num_degenerate_boxes": degenerate,
        "num_out_of_bounds_boxes": out_of_bounds,
        f"num_boxes_below_{int(min_height)}px": small,
    }


def class_balance_report(per_class: Mapping[str, int]) -> str:
    """Plain-text table of class counts and shares, plus the max/min imbalance ratio."""
    total = sum(per_class.values())
    if total == 0:
        return "no objects\n"
    width = max(len(c) for c in per_class) if per_class else 5
    lines = [f"{'class':<{width}}  {'count':>7}  {'share':>6}"]
    for cls, n in sorted(per_class.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"{cls:<{width}}  {n:>7}  {100.0 * n / total:>5.1f}%")
    counts = [n for n in per_class.values() if n > 0]
    ratio = max(counts) / min(counts)
    lines.append(f"imbalance ratio (max/min): {ratio:.2f}" + ("  (single class)" if len(counts) == 1 else ""))
    return "\n".join(lines) + "\n"


def train_val_split(stems: Iterable[str], val_fraction: float = 0.14, seed: int = 42) -> tuple[list[str], list[str]]:
    """Deterministic split. Sorting first makes the result independent of directory listing order.

    The validation count is ``round(n * val_fraction)``, at least 1 when there are two or more items.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    items = sorted(set(stems))
    n_val = round(len(items) * val_fraction)
    n_val = min(max(n_val, 1), len(items) - 1) if len(items) >= 2 else 0
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


def split_by_group(samples: Sequence[Sample], val_fraction: float = 0.14, seed: int = 42) -> dict[str, list[Sample]]:
    """Apply ``train_val_split`` inside each distractor set; returns {"train": [...], "val": [...]}."""
    out: dict[str, list[Sample]] = {"train": [], "val": []}
    groups = sorted({s.group for s in samples})
    for g in groups:
        members = {s.stem: s for s in samples if s.group == g}
        tr, va = train_val_split(members, val_fraction, seed)
        out["train"] += [members[k] for k in tr]
        out["val"] += [members[k] for k in va]
    return out


def tao_resolution_problems(width: int, height: int) -> list[str]:
    """DetectNet_v2 input sides must be multiples of 16; 960x544 (the tutorial's resolution) passes."""
    return [f"{name} {v} is not a multiple of 16" for name, v in (("width", width), ("height", height)) if v % 16]


def iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, boxes as (left, top, right, bottom). Returns shape (len(a), len(b))."""
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def average_precision(
    detections: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ground_truth: Mapping[str, np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """Single-class AP with all-point interpolation (VOC 2010+ style).

    Args:
        detections: frame_id -> (boxes (N, 4), scores (N,)).
        ground_truth: frame_id -> boxes (M, 4).
    With one class, AP equals mAP, as Basic Training 06 notes for the pallet-jack model (37:35-37:46).
    """
    n_gt = sum(len(np.asarray(g).reshape(-1, 4)) for g in ground_truth.values())
    if n_gt == 0:
        raise ValueError("no ground-truth boxes")
    records = []  # (score, frame, box)
    for frame, (det_boxes, scores) in detections.items():
        boxes = np.asarray(det_boxes, dtype=float).reshape(-1, 4)
        for box, score in zip(boxes, np.asarray(scores, dtype=float).reshape(-1), strict=True):
            records.append((float(score), frame, box))
    records.sort(key=lambda r: -r[0])
    matched = {f: np.zeros(len(np.asarray(g).reshape(-1, 4)), dtype=bool) for f, g in ground_truth.items()}
    tp = np.zeros(len(records))
    for i, (_, frame, box) in enumerate(records):
        gts = np.asarray(ground_truth.get(frame, np.zeros((0, 4))), dtype=float).reshape(-1, 4)
        if len(gts) == 0:
            continue
        overlaps = iou(box[None], gts)[0]
        j = int(np.argmax(overlaps))
        if overlaps[j] >= iou_threshold and not matched[frame][j]:
            matched[frame][j] = True
            tp[i] = 1.0
    cum_tp = np.cumsum(tp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.arange(1, len(records) + 1) if records else np.zeros(0)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _labels_for(samples: Sequence[Sample]) -> dict[str, list[KittiObject]]:
    return {f"{s.group}/{s.stem}": read_labels(s.label_path) for s in samples}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="KITTI dataset statistics and train/val split")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stats", "split"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS))
        p.add_argument("--width", type=int, default=960)
        p.add_argument("--height", type=int, default=544)
    sub.choices["split"].add_argument("--val", type=float, default=0.14)
    sub.choices["split"].add_argument("--seed", type=int, default=42)
    sub.choices["split"].add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    samples = pair_dataset(args.root, args.sets)
    if args.cmd == "stats":
        stats = dataset_stats(_labels_for(samples), image_size=(args.width, args.height))
        stats["labels_without_image"] = sum(1 for s in samples if s.image_path is None)
        stats["tao_resolution_problems"] = tao_resolution_problems(args.width, args.height)
        print(json.dumps(stats, indent=2))
        print(class_balance_report(stats["per_class"]), end="")
        return 0
    parts = split_by_group(samples, args.val, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, members in parts.items():
        (args.out / f"{name}.txt").write_text("".join(f"{s.group}/{s.stem}\n" for s in members), encoding="utf-8")
    n_tr, n_va = len(parts["train"]), len(parts["val"])
    print(f"train {n_tr}  val {n_va}  ({100.0 * n_va / max(n_tr + n_va, 1):.1f}% held out) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
