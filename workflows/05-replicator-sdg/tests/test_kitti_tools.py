"""CPU tests for kitti_tools on synthetic KITTI label files generated here (no Replicator output needed)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

FOLDER = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kitti_tools", FOLDER / "kitti_tools.py")
kt = importlib.util.module_from_spec(spec)
sys.modules["kitti_tools"] = kt  # train_torchvision imports it by name
spec.loader.exec_module(kt)

LINE = "palletjack 0.00 0 0.00 100.00 200.00 300.00 260.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00"


def _obj(left, top, right, bottom, *, cls="palletjack", score=None):
    return kt.KittiObject(cls, 0.0, 0, 0.0, (left, top, right, bottom), (0, 0, 0), (0, 0, 0), 0.0, score)


def _make_dataset(root: Path, rng: np.random.Generator, n_per_set=(50, 30, 20), with_images=True):
    """Write a Replicator-like tree: root/<set>/Camera/{rgb,object_detection}/<stem>.{png,txt}."""
    for group, n in zip(kt.DEFAULT_SETS, n_per_set, strict=True):
        lbl = root / group / "Camera" / "object_detection"
        img = root / group / "Camera" / "rgb"
        lbl.mkdir(parents=True)
        img.mkdir(parents=True)
        for i in range(n):
            k = int(rng.integers(0, 4))  # 0-3 pallet jacks per frame; some frames empty
            objs = []
            for _ in range(k):
                x, y = rng.uniform(0, 900), rng.uniform(0, 500)
                w, h = rng.uniform(8, 200), rng.uniform(8, 150)
                objs.append(_obj(x, y, x + w, y + h))
            (lbl / f"{i:06d}.txt").write_text("".join(o.to_line() + "\n" for o in objs), encoding="utf-8")
            if with_images:
                (img / f"{i:06d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # placeholder; stats never decode images


def test_parse_line_fields():
    o = kt.parse_line(LINE)
    assert o.cls == "palletjack"
    assert o.bbox == (100.0, 200.0, 300.0, 260.0)
    assert (o.width, o.height, o.area) == (200.0, 60.0, 12000.0)
    assert o.score is None
    assert kt.parse_line(LINE + " 0.87").score == pytest.approx(0.87)


def test_parse_line_round_trip():
    o = kt.parse_line(LINE + " 0.5")
    assert kt.parse_line(o.to_line()) == o


@pytest.mark.parametrize("bad", ["palletjack 0 0 0", LINE + " 1 2", LINE.replace("100.00", "abc")])
def test_parse_line_rejects_malformed(bad):
    with pytest.raises(ValueError):
        kt.parse_line(bad)


def test_read_labels_reports_file_and_line(tmp_path):
    f = tmp_path / "000001.txt"
    f.write_text(LINE + "\n\nbroken line\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"000001\.txt:3"):
        kt.read_labels(f)


def test_pair_dataset_and_missing_images(tmp_path):
    _make_dataset(tmp_path, np.random.default_rng(0), n_per_set=(3, 2, 1))
    (tmp_path / "no_distractors/Camera/rgb/000000.png").unlink()
    samples = kt.pair_dataset(tmp_path)
    assert len(samples) == 6
    assert {s.group for s in samples} == set(kt.DEFAULT_SETS)
    missing = [s for s in samples if s.image_path is None]
    assert [(s.group, s.stem) for s in missing] == [("no_distractors", "000000")]
    with pytest.raises(FileNotFoundError):
        kt.pair_dataset(tmp_path, sets=["not_a_set"])


def test_dataset_stats_counts_and_checks():
    labels = {
        "a": [_obj(10, 10, 110, 60), _obj(900, 500, 980, 560)],  # second box leaves a 960x544 frame
        "b": [],
        "c": [_obj(5, 5, 50, 15), _obj(20, 20, 20, 40)],  # 10 px tall (below 20) and zero width
        "d": [_obj(0, 0, 10, 30, cls="person")],
    }
    s = kt.dataset_stats(labels, image_size=(960, 544))
    assert s["num_images"] == 4
    assert s["num_empty_images"] == 1
    assert s["num_objects"] == 5
    assert s["objects_per_image_max"] == 2
    assert s["per_class"] == {"palletjack": 4, "person": 1}
    assert s["num_degenerate_boxes"] == 1
    assert s["num_out_of_bounds_boxes"] == 1
    assert s["num_boxes_below_20px"] == 1
    assert s["bbox_height_px"]["p50"] == pytest.approx(30.0)


def test_class_balance_report():
    rep = kt.class_balance_report({"palletjack": 300, "person": 100})
    assert "palletjack" in rep and "75.0%" in rep and "imbalance ratio (max/min): 3.00" in rep
    assert "(single class)" in kt.class_balance_report({"palletjack": 10})
    assert kt.class_balance_report({}) == "no objects\n"


def test_train_val_split_holds_out_14_percent_deterministically():
    stems = [f"{i:06d}" for i in range(2000)]
    train, val = kt.train_val_split(stems, 0.14, seed=7)
    assert len(val) == 280 and len(train) == 1720
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(stems)
    # order of the input does not matter; the seed does
    assert kt.train_val_split(reversed(stems), 0.14, seed=7) == (train, val)
    assert kt.train_val_split(stems, 0.14, seed=8)[1] != val


def test_train_val_split_edge_cases():
    assert kt.train_val_split(["a"], 0.14) == (["a"], [])
    assert kt.train_val_split(["a", "b"], 0.14) == (["a"], ["b"]) or kt.train_val_split(["a", "b"], 0.14) == (
        ["b"],
        ["a"],
    )
    with pytest.raises(ValueError):
        kt.train_val_split(["a", "b"], 1.0)


def test_split_by_group_keeps_each_set_at_14_percent(tmp_path):
    _make_dataset(tmp_path, np.random.default_rng(1), n_per_set=(100, 50, 50))
    parts = kt.split_by_group(kt.pair_dataset(tmp_path), 0.14, seed=3)
    by_group = {g: sum(1 for s in parts["val"] if s.group == g) for g in kt.DEFAULT_SETS}
    assert by_group == {"distractors_warehouse": 14, "distractors_additional": 7, "no_distractors": 7}
    assert len(parts["train"]) + len(parts["val"]) == 200


def test_tao_resolution():
    assert kt.tao_resolution_problems(960, 544) == []
    assert kt.tao_resolution_problems(1920, 1080) == ["height 1080 is not a multiple of 16"]


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 0, 0]])
    np.testing.assert_allclose(kt.iou(a, b), [[1.0, 50 / 150, 0.0, 0.0]])


def test_average_precision_perfect_partial_and_ranking():
    gts = {"f1": np.array([[0, 0, 10, 10]]), "f2": np.array([[0, 0, 10, 10], [20, 20, 40, 40]])}
    perfect = {"f1": (gts["f1"], np.array([0.9])), "f2": (gts["f2"], np.array([0.8, 0.7]))}
    assert kt.average_precision(perfect, gts) == pytest.approx(1.0)
    # one of three objects missed -> AP 2/3
    partial = {"f1": (gts["f1"], np.array([0.9])), "f2": (gts["f2"][:1], np.array([0.8]))}
    assert kt.average_precision(partial, gts) == pytest.approx(2 / 3)
    # a high-scoring false positive ranked first: precision at the first TP is 1/2
    fp_first = {"f1": (np.array([[100, 100, 120, 120], [0, 0, 10, 10]]), np.array([0.99, 0.5]))}
    assert kt.average_precision(fp_first, {"f1": gts["f1"]}) == pytest.approx(0.5)
    # duplicate detections of one object count once
    dup = {"f1": (np.array([[0, 0, 10, 10], [0, 0, 10, 10]]), np.array([0.9, 0.8]))}
    assert kt.average_precision(dup, {"f1": gts["f1"]}) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        kt.average_precision({}, {"f": np.zeros((0, 4))})


def test_cli_stats_and_split(tmp_path, capsys):
    _make_dataset(tmp_path, np.random.default_rng(2), n_per_set=(100, 50, 50))
    assert kt.main(["stats", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    stats = json.loads(out[: out.rindex("}") + 1])  # JSON first, then the plain-text balance table
    assert stats["num_images"] == 200
    assert stats["tao_resolution_problems"] == []
    assert kt.main(["split", "--root", str(tmp_path), "--out", str(tmp_path / "splits")]) == 0
    val = (tmp_path / "splits/val.txt").read_text(encoding="utf-8").split()
    train = (tmp_path / "splits/train.txt").read_text(encoding="utf-8").split()
    assert len(val) == 28 and len(train) == 172
    assert val[0].split("/")[0] in kt.DEFAULT_SETS
