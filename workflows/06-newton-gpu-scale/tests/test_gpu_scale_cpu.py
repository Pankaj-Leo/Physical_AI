"""Tiny CPU runs of the GPU-scale workflow: replicate + step, graph replay correctness, tiled-camera clips."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
pytest.importorskip("newton")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"wf06_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sb = _load("scale_benchmark")
rc = _load("render_clips")


def test_replicate_and_step_no_graph():
    cfg = sb.BenchConfig(frames=3, warmup_frames=1, graph=False)
    r1 = sb.run_one(1, cfg)
    r2 = sb.run_one(2, cfg)
    assert r1["finite"] and r2["finite"]
    assert not r1["graph"]
    assert r2["env_steps_per_s"] > 0
    assert r2["sim_array_mb"] > r1["sim_array_mb"]  # arrays grow with world count


def test_graph_replay_matches_plain_stepping():
    """A captured frame replayed on CPU must reproduce plain stepping bit for bit."""
    cfg_a = sb.BenchConfig(graph=False)
    a = sb.ParallelSim(sb.build_model(2, cfg_a), cfg_a)
    for _ in range(4):
        a.simulate()
    cfg_b = sb.BenchConfig(graph=True)
    b = sb.ParallelSim(sb.build_model(2, cfg_b), cfg_b)
    b.capture()  # one warm-up frame executes; the captured frame is recorded, not executed, on CPU
    for _ in range(3):
        b.step()
    assert b.graph is not None
    np.testing.assert_array_equal(a.state_0.body_q.numpy(), b.state_0.body_q.numpy())


def test_gpu_paths_are_marked():
    assert "not executed: needs GPU" in inspect.getdoc(sb.gpu_sweep)
    assert "not executed: needs GPU" in inspect.getdoc(rc.gpu_preset)
    assert sb.GPU_WORLD_COUNTS[-1] == 16384


def test_render_tiny_clips(tmp_path):
    cfg = rc.ClipConfig(worlds=2, width=32, height=18, frames=4, write_mp4=False)
    summary = rc.render(cfg, tmp_path)
    assert summary["device"] == "cpu"
    for w in range(2):
        for kind in ("color", "depth", "depth_mm", "seg"):
            assert len(list((tmp_path / f"world_{w:03d}" / kind).glob("frame_*.png"))) == 4
        gt = json.loads((tmp_path / f"world_{w:03d}.json").read_text())
        assert len(gt["base_pose_xyz_qxyzw"]) == 4
        assert set(gt) >= {"fell", "forward_distance_m", "camera", "semantic_classes", "fall_rule"}
    manifest = [json.loads(line) for line in (tmp_path / "clips.jsonl").read_text().splitlines()]
    assert {"clip", "material", "seed", "fell", "forward_distance_m"} <= set(manifest[0])

    from PIL import Image

    seg = np.asarray(Image.open(tmp_path / "world_000" / "seg" / "frame_0003.png"))
    robot_colors = {tuple(rc.CLASSES[i][1]) for i in (2, 3, 4, 5)}
    present = {tuple(px) for px in seg.reshape(-1, 3)}
    assert present & robot_colors, "robot not visible in the semantic image"
    depth_mm = np.asarray(Image.open(tmp_path / "world_000" / "depth_mm" / "frame_0003.png"))
    assert depth_mm.max() > 500  # ground and robot are metres away from the camera
