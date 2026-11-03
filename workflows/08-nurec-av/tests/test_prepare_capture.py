"""Capture preparation on synthetic drives: geometry conventions, outputs, and fault detection (CPU)."""

import json
import math

import numpy as np
import prepare_capture as pc
import pytest


@pytest.fixture
def clean(tmp_path):
    cap = pc.make_synthetic_capture(tmp_path / "cap")
    rep = pc.prepare(cap, tmp_path / "out")
    return cap, tmp_path / "out", rep


def test_quaternion_helpers():
    r = np.random.default_rng(1)
    for _ in range(20):
        q = r.normal(size=4)
        q /= np.linalg.norm(q)
        q = -q if q[3] < 0 else q
        R = pc.quat_to_matrix(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.allclose(pc.matrix_to_quat(R), q, atol=1e-9)
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])  # 90 deg yaw
    assert pc.yaw_of(pc.slerp(q0, q1, 0.5)) == pytest.approx(math.pi / 4)
    assert pc.yaw_of(pc.slerp(q0, -q1, 0.5)) == pytest.approx(math.pi / 4)  # sign-flipped input, same rotation


def test_clean_capture_passes(clean):
    _, out, rep = clean
    assert rep["ok"] and rep["errors"] == []
    assert rep["sync"]["max_group_skew_ms"] < 1e-6
    assert rep["poses"]["rate_hz"] == pytest.approx(100.0)
    cov = rep["coverage"]
    assert cov["frames_used"] == 120 and cov["path_length_m"] == pytest.approx(31.3, abs=0.5)
    assert cov["camera_lateral_spread_m"] == pytest.approx(1.6, abs=0.01)  # left/right cameras at +-0.8 m
    for name in ("transforms.json", "sparse/0/cameras.txt", "sparse/0/images.txt", "sparse/0/points3D.txt"):
        assert (out / name).is_file()
    assert json.loads((out / "capture_report.json").read_text())["ok"]


def test_front_camera_looks_along_heading(clean):
    _, out, _ = clean
    tf = json.loads((out / "transforms.json").read_text())
    assert tf["camera_model"] == "OPENCV" and len(tf["frames"]) == 120
    f0 = next(f for f in tf["frames"] if f["camera"] == "front")
    c2w_gl = np.array(f0["transform_matrix"])
    view_dir = -c2w_gl[:3, 2]  # OpenGL cameras look down -z
    t = 0.05  # first frame time in the synthetic drive
    heading = np.array([8.0, math.cos(0.5 * t), 0.0])
    assert np.dot(view_dir, heading / np.linalg.norm(heading)) > 0.999
    assert c2w_gl[2, 1] > 0.99  # camera up (+y in OpenGL) is world up
    assert f0["mask_path"] == "masks/front.png" and (out / f0["mask_path"]).is_file()
    assert (out / f0["file_path"]).is_file()


def test_colmap_images_agree_with_transforms(clean):
    _, out, _ = clean
    tf = {f["file_path"][len("images/") :]: f for f in json.loads((out / "transforms.json").read_text())["frames"]}
    lines = [ln for ln in (out / "sparse/0/images.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 120
    for ln in lines[:10]:
        _, qw, qx, qy, qz, tx, ty, tz, _, name = ln.split()
        R_cw = pc.quat_to_matrix([float(qx), float(qy), float(qz), float(qw)])
        centre = -R_cw.T @ np.array([float(tx), float(ty), float(tz)])
        assert np.allclose(centre, np.array(tf[name]["transform_matrix"])[:3, 3], atol=1e-5)
    cams = (out / "sparse/0/cameras.txt").read_text().splitlines()[1:]
    assert [c.split()[1] for c in cams] == ["OPENCV"] * 3
    assert len((out / "sparse/0/points3D.txt").read_text().splitlines()) == 201


def test_detects_skew_drops_and_pose_jumps(tmp_path):
    cap = pc.make_synthetic_capture(tmp_path / "bad", skew_ms={"left": 3.0}, drop={"right": [5, 6]}, pose_jump_at=200)
    rep = pc.prepare(cap, tmp_path / "out")
    assert not rep["ok"]
    assert rep["sync"]["per_camera_offset_ms"]["left"]["median"] == pytest.approx(3.0, abs=1e-3)
    assert rep["sync"]["groups_over_limit"] > 0
    assert rep["frame_intervals"]["right"]["dropped_frames_est"] == 2
    assert rep["poses"]["jumps"] >= 1
    assert any("PTP" in e for e in rep["errors"]) and any("pose jumps" in e for e in rep["errors"])


def test_interpolation_outside_range_is_refused(tmp_path):
    cap = pc.load_capture(pc.make_synthetic_capture(tmp_path / "c", with_points=False, with_mask=False))
    assert pc.interpolate_pose(cap, int(cap.pose_t[0]) - 1) is None
    mid = pc.interpolate_pose(cap, int((cap.pose_t[0] + cap.pose_t[1]) // 2))
    assert np.allclose(mid[:3, 3], 0.5 * (cap.pose_p[0] + cap.pose_p[1]))


def test_cli_synthetic_then_prepare(tmp_path, capsys):
    assert pc.main(["--synthetic", str(tmp_path / "s")]) == 0
    assert pc.main(["--capture", str(tmp_path / "s"), "--out", str(tmp_path / "o"), "--copy"]) == 0
    assert "ok: True" in capsys.readouterr().out
    assert not (tmp_path / "o/images/front").is_symlink()
