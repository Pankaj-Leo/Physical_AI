"""CPU tests for the Isaac Lab -> Newton sim-to-sim mapping.

Cross-checks the name-derived index maps against the arrays hard-coded in Newton's
``example_robot_anymal_c_walk.py`` (read with ``ast``, so warp and warp_nn are not imported) and against the joint
order in the cached ANYmal C URDF that ``ssw.terrain`` downloads. Both external files are optional: the test skips if
they are absent.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
FOLDER = HERE.parents[1]
REPO = HERE.parents[4]
NEWTON_EXAMPLE = REPO / "newton-main/newton/examples/robot/example_robot_anymal_c_walk.py"
URDF_GLOB = "newton-assets_anybotics_anymal_c_*/anybotics_anymal_c/urdf/anymal.urdf"

spec = importlib.util.spec_from_file_location("atlas_sim2sim_map", FOLDER / "scripts/sim2sim_map.py")
s2s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2s)


def _example_arrays() -> dict[str, list[int]]:
    if not NEWTON_EXAMPLE.exists():
        pytest.skip("newton-main snapshot not present")
    tree = ast.parse(NEWTON_EXAMPLE.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) else None
        if isinstance(target, ast.Name) and target.id in {"lab_to_mujoco", "mujoco_to_lab"}:
            out[target.id] = ast.literal_eval(node.value)
    return out


def test_reorder_indices_round_trip():
    lab, urdf = s2s.ANYMAL_C_ISAACLAB_ORDER, s2s.ANYMAL_C_URDF_ORDER
    n2l = s2s.reorder_indices(urdf, lab)
    l2n = s2s.reorder_indices(lab, urdf)
    assert [urdf[i] for i in n2l] == lab
    assert [lab[i] for i in l2n] == urdf
    np.testing.assert_array_equal(n2l[l2n], np.arange(12))


def test_reorder_indices_rejects_mismatched_names():
    with pytest.raises(ValueError, match="differ"):
        s2s.reorder_indices(["a", "b"], ["a", "c"])
    with pytest.raises(ValueError, match="unique"):
        s2s.reorder_indices(["a", "a"], ["a", "a"])


def test_maps_match_newton_example_hardcoded_arrays():
    arrays = _example_arrays()
    lab, urdf = s2s.ANYMAL_C_ISAACLAB_ORDER, s2s.ANYMAL_C_URDF_ORDER
    # The example's lab_to_mujoco indexes Newton arrays to build policy-ordered observations.
    assert s2s.reorder_indices(urdf, lab).tolist() == arrays["lab_to_mujoco"]
    # Its mujoco_to_lab indexes policy actions to build Newton-ordered joint targets.
    assert s2s.reorder_indices(lab, urdf).tolist() == arrays["mujoco_to_lab"]


def test_urdf_order_matches_cached_asset():
    candidates = sorted((Path.home() / ".cache/newton").glob(URDF_GLOB))
    if not candidates:
        pytest.skip("ANYmal C asset not cached (run python -m ssw.terrain run --material rigid once)")
    text = candidates[0].read_text(encoding="utf-8")
    movable = re.findall(r'<joint name="([^"]+)" type="(?:revolute|continuous|prismatic)"', text)
    assert movable == s2s.ANYMAL_C_URDF_ORDER


def test_obs_layout_is_48_wide_with_the_diagram_blocks():
    sl = s2s.obs_slices()
    assert sum(w for _, w in s2s.VELOCITY_OBS_LAYOUT) == 48
    assert sl["velocity_command"] == slice(9, 12)
    assert sl["last_action"] == slice(36, 48)


def test_build_obs_and_targets_for_a_standing_robot():
    lab, urdf = s2s.ANYMAL_C_ISAACLAB_ORDER, s2s.ANYMAL_C_URDF_ORDER
    n2l, l2n = s2s.reorder_indices(urdf, lab), s2s.reorder_indices(lab, urdf)
    q0 = np.arange(12, dtype=float) / 10.0  # distinct defaults expose any ordering slip
    joint_q = np.concatenate([[0.0, 0.0, 0.6], [0.0, 0.0, 0.0, 1.0], q0 + 0.01 * np.arange(12)])  # identity XYZW
    joint_qd = np.concatenate([[1.0, 0.0, 0.0], [0.0, 0.0, 0.5], np.arange(12, dtype=float)])
    last = np.arange(12, dtype=float)
    obs = s2s.build_velocity_obs(
        joint_q, joint_qd, q0, command=np.array([1.0, 0.0, 0.0]), last_action=last, newton_to_lab=n2l
    )
    sl = s2s.obs_slices()
    np.testing.assert_allclose(obs[sl["base_lin_vel_b"]], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(obs[sl["projected_gravity_b"]], [0.0, 0.0, -1.0])
    # joint i of the policy order is joint n2l[i] of Newton order
    np.testing.assert_allclose(obs[sl["joint_pos_rel"]], 0.01 * n2l)
    np.testing.assert_allclose(obs[sl["joint_vel_rel"]], n2l.astype(float))
    # zero action -> default pose; a one-hot action on policy joint k moves Newton joint l2n^-1(k) only
    np.testing.assert_allclose(s2s.action_to_targets(np.zeros(12), q0, l2n), q0)
    k = lab.index("RF_HFE")
    tgt = s2s.action_to_targets(np.eye(12)[k], q0, l2n)
    moved = np.flatnonzero(~np.isclose(tgt, q0))
    assert [urdf[i] for i in moved] == ["RF_HFE"]
    np.testing.assert_allclose(tgt[moved] - q0[moved], 0.5)


def test_projected_gravity_for_a_tilted_base():
    # Base pitched +90 deg about y (XYZW): body x axis points down, so gravity appears along body +x.
    q = np.array([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
    np.testing.assert_allclose(s2s.quat_rotate_inverse_xyzw(q, [0, 0, -1.0]), [1.0, 0.0, 0.0], atol=1e-12)
