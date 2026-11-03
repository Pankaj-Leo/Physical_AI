"""gsplat trainer: the CPU half (dataset loading, conventions, plan) and the guard on the GPU half."""

import importlib.util
import json

import numpy as np
import prepare_capture as pc
import pytest
import train_gsplat as tg


@pytest.fixture
def prepared(tmp_path):
    pc.prepare(pc.make_synthetic_capture(tmp_path / "cap"), tmp_path / "out")
    return tmp_path / "out"


def test_load_dataset_conventions(prepared):
    views, pts = tg.load_dataset(prepared)
    assert len(views) == 120 and pts.shape == (200, 6)
    tf = json.loads((prepared / "transforms.json").read_text())["frames"]
    for v, f in zip(views[:5], tf[:5], strict=True):
        c2w_cv = np.array(f["transform_matrix"]) @ tg.GL_FROM_CV
        assert np.allclose(np.linalg.inv(v.viewmat), c2w_cv)
        # a point 10 m along the camera's optical axis projects to the principal point with positive depth
        p_world = c2w_cv @ np.array([0.0, 0.0, 10.0, 1.0])
        p_cam = v.viewmat @ p_world
        uv = v.K @ (p_cam[:3] / p_cam[2])
        assert p_cam[2] == pytest.approx(10.0) and np.allclose(uv[:2], [v.K[0, 2], v.K[1, 2]])


def test_plan_on_cpu(prepared, capsys):
    p = tg.plan(prepared, tg.TrainConfig())
    assert p["views"] == 120 and p["masked_views"] == 40 and p["init_points"] == 200
    assert p["missing_images"] == [] and "not executed: needs GPU" in p["status"]
    assert tg.main(["--data", str(prepared), "--dry-run"]) == 0
    assert '"views": 120' in capsys.readouterr().out


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="torch installed; guard not exercised")
def test_training_is_guarded_without_gpu_stack(prepared, tmp_path):
    with pytest.raises(SystemExit, match="gsplat"):
        tg.train(prepared, tmp_path / "gs", tg.TrainConfig(steps=1))
