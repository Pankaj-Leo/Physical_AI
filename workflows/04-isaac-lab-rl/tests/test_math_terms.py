"""CPU tests for the push-task reward, termination, and frame math (numpy path of the array-API functions)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MDP = Path(__file__).resolve().parents[1] / "source/atlas_rl_tasks/atlas_rl_tasks/tasks/push_object/mdp"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mt = _load("atlas_math_terms", MDP / "math_terms.py")


def test_namespace_is_numpy_for_numpy_inputs():
    assert mt.namespace(np.zeros(3)) is np


def test_distance_full_and_planar():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    b = np.array([[3.0, 4.0, 12.0], [1.0, 2.0, 0.0]])
    np.testing.assert_allclose(mt.distance(a, b), [13.0, 2.0])
    np.testing.assert_allclose(mt.distance(a, b, planar=True), [5.0, 0.0])


def test_reach_reward_is_one_at_contact_and_decreases_with_distance():
    obj = np.zeros((4, 3))
    wrist = np.array([[0.0, 0, 0], [0.025, 0, 0], [0.1, 0, 0], [1.0, 0, 0]])
    r = mt.reach_reward(wrist, obj, std=0.1)
    assert r.shape == (4,)
    np.testing.assert_allclose(r[0], 1.0)
    np.testing.assert_allclose(r[1], 1 - np.tanh(0.25), rtol=1e-12)
    np.testing.assert_allclose(r[2], 1 - np.tanh(1.0), rtol=1e-12)  # ~0.238
    assert np.all(np.diff(r) < 0)
    assert 0 < r[-1] < 1e-3


def test_reach_reward_rejects_bad_std():
    with pytest.raises(ValueError, match="std"):
        mt.reach_reward(np.zeros((1, 3)), np.zeros((1, 3)), std=0.0)


def test_progress_rewards_motion_toward_target_and_penalizes_motion_away():
    prev = np.array([0.20, 0.20, 0.20])
    curr = np.array([0.19, 0.21, 0.20])
    fresh = np.array([False, False, False])
    r, new_prev = mt.progress_step(prev, curr, fresh, max_step=0.02)
    np.testing.assert_allclose(r, [0.01, -0.01, 0.0])
    np.testing.assert_array_equal(new_prev, curr)


def test_progress_is_zero_for_fresh_envs_and_clipped_for_teleports():
    prev = np.array([0.50, 0.50])
    curr = np.array([0.10, 0.10])  # a reset moved the object 0.4 m in one step
    r, _ = mt.progress_step(prev, curr, np.array([True, False]), max_step=0.02)
    np.testing.assert_allclose(r, [0.0, 0.02])


def test_progress_sums_to_net_distance_over_an_episode():
    # Object pushed from 0.15 m to 0.01 m from the target in 20 steps, with a 2-step wobble backwards.
    d = np.concatenate([np.linspace(0.15, 0.05, 11), [0.06, 0.07], np.linspace(0.06, 0.01, 6)])
    prev = np.array([d[0]])
    fresh = np.array([True])
    total = 0.0
    for x in d:
        r, prev = mt.progress_step(prev, np.array([x]), fresh)
        fresh = np.array([False])
        total += float(r[0])
    np.testing.assert_allclose(total, d[0] - d[-1], atol=1e-12)


def test_success_bonus_and_at_goal_use_planar_distance():
    obj = np.array([[0.50, 0.20, 0.72], [0.50, 0.20, 0.90], [0.55, 0.20, 0.72]])
    tgt = np.array([[0.51, 0.21, 0.72]] * 3)
    np.testing.assert_array_equal(mt.at_goal(obj, tgt, 0.03), [True, True, False])
    np.testing.assert_array_equal(mt.at_goal(obj, tgt, 0.03, planar=False), [True, False, False])
    bonus = mt.success_bonus(obj, tgt, 0.03)
    assert bonus.dtype.kind == "f"
    np.testing.assert_array_equal(bonus, [1.0, 1.0, 0.0])


def test_terminations():
    np.testing.assert_array_equal(mt.below_height(np.array([0.72, 0.59]), 0.60), [False, True])
    pos = np.array([[0.3, 0.2, 0.0], [0.7, 0.4, 0.0]])
    np.testing.assert_array_equal(mt.beyond_radius(pos, np.zeros_like(pos), 0.75), [False, True])


def _quat_wxyz_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def _rotmat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_quat_apply_inverse_matches_rotation_matrix_transpose():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(64, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    v = rng.normal(size=(64, 3))
    got = mt.quat_apply_inverse_wxyz(q, v)
    want = np.stack([_rotmat(qi).T @ vi for qi, vi in zip(q, v, strict=True)])
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_pos_in_frame_yaw_90():
    # Robot at (1, 2, 0.74) yawed +90 deg: a point 0.3 m along world +y is 0.3 m straight ahead (+x) in the root frame.
    frame_pos = np.array([[1.0, 2.0, 0.74]])
    q = _quat_wxyz_from_axis_angle([0, 0, 1], np.pi / 2)[None]
    p = np.array([[1.0, 2.3, 0.74]])
    np.testing.assert_allclose(mt.pos_in_frame(frame_pos, q, p), [[0.3, 0.0, 0.0]], atol=1e-12)
