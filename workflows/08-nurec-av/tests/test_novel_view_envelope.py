"""Novel-view visibility proxy: ray casting, occlusion, the trend, and the committed numbers."""

import json
from pathlib import Path

import novel_view_envelope as nv
import numpy as np
import pytest

FOLDER = Path(__file__).resolve().parents[1]


def test_occlusion_by_a_wall():
    wall = np.array([[0.0, 0.0, 0.0, 1.0, 10.0, 5.0]])
    cams = np.array([[-5.0, 5.0, 1.6], [5.0, 5.0, 1.6], [-5.0, 5.0, 20.0]])
    vis = nv.visible_from(np.array([[3.0, 5.0, 1.0]]), cams, wall)[0]
    assert vis.tolist() == [False, True, True]  # behind the wall; same side; high enough to see over it


def test_cast_hits_ground_and_boxes():
    boxes = np.array([[10.0, -1.0, 0.0, 11.0, 1.0, 3.0]])
    o = np.array([[0.0, 0.0, 1.6], [0.0, 0.0, 1.6], [0.0, 0.0, 1.6]])
    d = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    pts, hit = nv.cast(o, d, boxes)
    assert hit.tolist() == [True, True, False]
    assert np.allclose(pts[0], [10.0, 0.0, 1.6]) and np.allclose(pts[1], [0.0, 0.0, 0.0])


def test_support_falls_with_offset_and_matches_committed():
    rows = nv.envelope(offsets=(0.0, 3.5))
    assert rows[0]["well_supported"] == 1.0 and rows[0]["median_min_angle_deg"] == 0.0
    assert rows[1]["well_supported"] < 0.6 and rows[1]["ground_well_supported"] < 0.25
    committed = json.loads((FOLDER / "results" / "novel_view_envelope.json").read_text())["rows"]
    by_key = {(r["case"], r["offset_m"]): r for r in committed}
    for r in rows:
        assert by_key[(r["case"], r["offset_m"])] == r
    lateral = [r for r in committed if r["case"] == "lateral"]
    angles = [r["median_min_angle_deg"] for r in lateral]
    assert angles == sorted(angles)


def test_committed_turn_case_is_worst():
    committed = json.loads((FOLDER / "results" / "novel_view_envelope.json").read_text())["rows"]
    turn = next(r for r in committed if r["case"] != "lateral")
    assert turn["median_min_angle_deg"] > max(r["median_min_angle_deg"] for r in committed if r["case"] == "lateral")


@pytest.mark.parametrize("off", [0.0])
def test_zero_offset_has_full_ground_support(off):
    xs = np.array([50.0])
    novel = np.column_stack([xs, np.full_like(xs, nv.EGO_Y + off), np.full_like(xs, nv.CAM_H)])
    assert nv.support(novel)["ground_well_supported"] == 1.0
