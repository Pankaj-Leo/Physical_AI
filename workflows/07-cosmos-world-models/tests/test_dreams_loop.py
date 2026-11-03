"""Toy closed loop: dynamics, rendering, perception, the history cache, Kendall tau, and the ranking study."""

import json
import math
from pathlib import Path

import dreams_loop as dl
import numpy as np
import pytest

FOLDER = Path(__file__).resolve().parents[1]
STRAIGHT = dl.Scenario(0, dl.RoadMap(0.0, 100.0), 20.0, 10.0, 99.0, 0.0, 0.0, 500.0, 10.0)


def test_bicycle_model():
    rt = dl.ToyRuntime(STRAIGHT, ego_speed=10.0)
    states = rt.step(np.zeros((dl.FRAMES_PER_CHUNK, 2)))
    assert len(states) == dl.FRAMES_PER_CHUNK
    assert states[-1].ego.x == pytest.approx(8.0)  # 8 frames x 0.1 s x 10 m/s
    rt.reset()
    rt.step(np.array([[0.0, 0.1]]))
    assert rt.ego.yaw == pytest.approx(10.0 / dl.WHEELBASE * math.tan(0.1) * dl.DT)


def test_obb_overlap():
    a = dl.Box("a", 0.0, 0.0, 0.0)
    assert dl.obb_overlap(a, dl.Box("b", 4.0, 0.0, 0.0))
    assert not dl.obb_overlap(a, dl.Box("b", 5.0, 0.0, 0.0))
    assert not dl.obb_overlap(a, dl.Box("b", 0.0, 2.0, 0.0))
    assert dl.obb_overlap(a, dl.Box("b", 0.0, 2.0, math.pi / 2))


def test_rasterize_and_perception_on_straight_road():
    state = dl.ToyRuntime(STRAIGHT).reset()
    frame = dl.rasterize(state)
    assert frame.shape == (2, dl.BEV_H, dl.BEV_W)
    lead_row = dl.EGO_ROW - round(20.0 / dl.BEV_RES)
    assert frame[1, lead_row, dl.EGO_COL] == 1.0
    lane = dl.perceive_lane(frame)
    assert abs(np.polyval(lane, 10.0)) < 0.3  # ego sits on its lane center
    gap = dl.perceive_gap(frame, lane)
    assert gap == pytest.approx(20.0 - dl.CAR_L, abs=0.6)


def test_policy_chunk_shape_and_limits():
    state = dl.ToyRuntime(STRAIGHT).reset()
    acts = dl.FramePolicy(dl.POLICIES[1]).act(dl.rasterize(state), 10.0)
    assert acts.shape == (dl.FRAMES_PER_CHUNK, 2)
    assert np.all(np.abs(acts[:, 1]) <= 0.5)


def test_world_model_cache_is_static_shape():
    rt = dl.ToyRuntime(STRAIGHT)
    wm = dl.RasterWorldModel(dl.SEVERE, seed=1)
    first = dl.rasterize(rt.reset())
    wm.reset(first, dl.PROMPT)
    assert np.array_equal(wm.cache[0], first)
    shape = wm.cache.shape
    for _ in range(5):
        frames = wm.generate(rt.step(np.zeros((8, 2))))
        assert frames.shape == (8, 2, dl.BEV_H, dl.BEV_W)
        assert wm.cache.shape == shape
    assert wm.ptr == 40


def test_exact_loop_drives_safely():
    m = dl.run_episode(dl.FramePolicy(dl.POLICIES[1]), dl.ToyRuntime(dl.Scenario.sample(0)), dl.RasterWorldModel())
    assert not m.collided and m.departure_s == 0.0 and m.progress_m > 50.0


def test_kendall_tau():
    assert dl.kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert dl.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    scipy_stats = pytest.importorskip("scipy.stats")
    r = np.random.default_rng(0)
    for _ in range(5):
        a, b = r.integers(0, 4, 7), r.integers(0, 4, 7)
        assert dl.kendall_tau(a, b) == pytest.approx(scipy_stats.kendalltau(a, b).statistic)


def test_ranking_study_reproduces_committed_result():
    summ = dl.ranking_study(range(10)).summary()
    committed = json.loads((FOLDER / "results" / "dreams_ranking.json").read_text())
    assert summ == committed
    assert summ["kendall_tau_vs_exact"]["exact"] == 1.0
    assert summ["kendall_tau_vs_exact"]["mild"] >= summ["kendall_tau_vs_exact"]["severe"]


def test_wsm_like_render_and_export(tmp_path):
    state = dl.ToyRuntime(dl.Scenario.sample(0)).reset()
    img = dl.render_wsm_like(state, 320, 180)
    assert img.shape == (180, 320, 3) and img.any()
    assert ((img == (255, 210, 0)).all(axis=-1)).any()  # yellow center line is drawn
    dl.export_wsm(tmp_path, frames=5)
    assert len(list(tmp_path.glob("frame_*.png"))) == 5
