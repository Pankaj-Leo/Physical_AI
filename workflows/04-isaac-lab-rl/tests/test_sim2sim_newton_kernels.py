"""Cross-check the numpy sim-to-sim reference against Newton's own Warp kernels, on the Warp CPU device.

Runs ``_compute_obs_kernel`` and ``_build_joint_target_q_kernel`` from ``example_robot_anymal_c_walk.py`` on random
floating-base states and compares with ``scripts/sim2sim_map.py``. Then feeds one observation through the shipped
PhysX-trained ANYmal policy (``anymal_walking_policy_physx.onnx``) with warp_nn, the same runtime the example uses.

Skips cleanly when Newton, warp_nn, or the cached asset are missing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("warp_nn")
example = pytest.importorskip("newton.examples.robot.example_robot_anymal_c_walk")

FOLDER = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("atlas_sim2sim_map_k", FOLDER / "scripts/sim2sim_map.py")
s2s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2s)

LAB, URDF = s2s.ANYMAL_C_ISAACLAB_ORDER, s2s.ANYMAL_C_URDF_ORDER
N2L, L2N = s2s.reorder_indices(URDF, LAB), s2s.reorder_indices(LAB, URDF)


def _random_state(rng):
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)  # XYZW
    joint_q = np.concatenate([rng.normal(size=3), quat, rng.normal(scale=0.5, size=12)]).astype(np.float32)
    joint_qd = rng.normal(size=18).astype(np.float32)
    q0 = rng.normal(scale=0.5, size=12).astype(np.float32)
    command = rng.normal(size=3).astype(np.float32)
    last = rng.normal(size=12).astype(np.float32)
    return joint_q, joint_qd, q0, command, last


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_numpy_obs_matches_newton_obs_kernel(seed):
    rng = np.random.default_rng(seed)
    joint_q, joint_qd, q0, command, last = _random_state(rng)
    dev = "cpu"
    obs = wp.zeros((1, 48), dtype=wp.float32, device=dev)
    wp.launch(
        example._compute_obs_kernel,
        dim=1,
        inputs=[
            wp.array(joint_q, dtype=wp.float32, device=dev),
            wp.array(joint_qd, dtype=wp.float32, device=dev),
            wp.array(q0, dtype=wp.float32, device=dev),
            wp.array(np.asarray(example.lab_to_mujoco, dtype=np.int32), dtype=wp.int32, device=dev),
            wp.vec3(0.0, 0.0, -1.0),
            wp.vec3(*command.tolist()),
            wp.array(last[None, :], dtype=wp.float32, device=dev),
            obs,
        ],
        device=dev,
    )
    want = obs.numpy()[0]
    got = s2s.build_velocity_obs(
        joint_q.astype(float), joint_qd.astype(float), q0, command=command, last_action=last, newton_to_lab=N2L
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_numpy_targets_match_newton_target_kernel():
    rng = np.random.default_rng(3)
    act = rng.normal(size=12).astype(np.float32)
    q0 = rng.normal(scale=0.5, size=12).astype(np.float32)
    dev = "cpu"
    out = wp.zeros(19, dtype=wp.float32, device=dev)
    wp.launch(
        example._build_joint_target_q_kernel,
        dim=19,
        inputs=[
            wp.array(act[None, :], dtype=wp.float32, device=dev),
            wp.array(q0, dtype=wp.float32, device=dev),
            wp.array(np.asarray(example.mujoco_to_lab, dtype=np.int32), dtype=wp.int32, device=dev),
            0.5,
            7,
            out,
        ],
        device=dev,
    )
    np.testing.assert_allclose(s2s.action_to_targets(act, q0, L2N), out.numpy()[7:], rtol=1e-6, atol=1e-6)


def test_shipped_physx_policy_runs_on_cpu_with_48_in_12_out():
    candidates = sorted(
        (Path.home() / ".cache/newton").glob(
            "newton-assets_anybotics_anymal_c_*/anybotics_anymal_c/rl_policies/anymal_walking_policy_physx.onnx"
        )
    )
    if not candidates:
        pytest.skip("ANYmal C policy not cached")
    from warp_nn.runtime import OnnxRuntime

    policy = OnnxRuntime(str(candidates[0]), device="cpu")
    name_in, name_out = policy.input_names[0], policy.output_names[0]
    # Standing still, upright, zero command: the observation is all zeros except projected gravity.
    obs_np = np.zeros((1, 48), dtype=np.float32)
    obs_np[0, s2s.obs_slices()["projected_gravity_b"]] = [0.0, 0.0, -1.0]
    act = policy({name_in: wp.array(obs_np, dtype=wp.float32, device="cpu")})[name_out].numpy()
    assert act.shape == (1, 12)
    assert np.all(np.isfinite(act))
