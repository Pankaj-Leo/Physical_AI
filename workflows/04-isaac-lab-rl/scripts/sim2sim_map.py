"""Joint-order maps and the observation layout needed to run an Isaac Lab locomotion policy in Newton.

An Isaac Lab policy reads and writes joints in the order the PhysX articulation reports them (for ANYmal C, grouped by
joint type: all HAA, then all HFE, then all KFE). Newton, like MuJoCo, keeps the URDF's depth-first order (one leg at
a time). Feeding one order into a policy trained on the other produces a robot that twitches and falls. Newton's
``example_robot_anymal_c_walk.py`` hard-codes the two index arrays; this module derives them from joint names so the
same step works for any robot, and ``tests/test_sim2sim_map.py`` checks the derivation against the upstream arrays and
the cached ANYmal URDF.

It also documents the 48-wide observation of Isaac Lab's flat-terrain velocity task, the same seven blocks shown in the
NVIDIA Developer Blog diagram this folder uses as evidence (base linear and angular velocity, projected gravity,
velocity command, joint positions, joint velocities, last action).

Pure numpy; executed here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Isaac Lab (PhysX) order for ANYmal C, as implied by Newton's lab_to_mujoco array; grouped by joint type.
ANYMAL_C_ISAACLAB_ORDER = [f"{leg}_{j}" for j in ("HAA", "HFE", "KFE") for leg in ("LF", "LH", "RF", "RH")]
# URDF / Newton / MuJoCo order: depth-first, one leg at a time.
ANYMAL_C_URDF_ORDER = [f"{leg}_{j}" for leg in ("LF", "RF", "LH", "RH") for j in ("HAA", "HFE", "KFE")]

# Isaac-Velocity-Flat-Anymal-C-v0 policy observation, in order. Widths sum to 48.
VELOCITY_OBS_LAYOUT: list[tuple[str, int]] = [
    ("base_lin_vel_b", 3),
    ("base_ang_vel_b", 3),
    ("projected_gravity_b", 3),
    ("velocity_command", 3),  # (vx, vy, yaw rate), the "Sampled Command" box in the diagram
    ("joint_pos_rel", 12),  # q - q_default, policy (Isaac Lab) order
    ("joint_vel_rel", 12),
    ("last_action", 12),
]


def reorder_indices(src: Sequence[str], dst: Sequence[str]) -> np.ndarray:
    """Index array ``idx`` with ``dst[i] == src[idx[i]]``, so ``x_dst = x_src[idx]``.

    Raises ValueError if the two lists do not hold the same joint names exactly once each.
    """
    if len(set(src)) != len(src) or len(set(dst)) != len(dst):
        raise ValueError("joint names must be unique")
    if set(src) != set(dst):
        missing = sorted(set(src) ^ set(dst))
        raise ValueError(f"joint sets differ: {missing}")
    pos = {name: i for i, name in enumerate(src)}
    return np.array([pos[name] for name in dst], dtype=np.int64)


def obs_slices(layout: Sequence[tuple[str, int]] = VELOCITY_OBS_LAYOUT) -> dict[str, slice]:
    """Name -> slice into the flat observation vector."""
    out: dict[str, slice] = {}
    start = 0
    for name, width in layout:
        out[name] = slice(start, start + width)
        start += width
    return out


def quat_rotate_inverse_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the inverse of unit quaternion ``q`` given in Newton's (x, y, z, w) order."""
    u = -np.asarray(q[:3], dtype=float)
    w = float(q[3])
    v = np.asarray(v, dtype=float)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def build_velocity_obs(
    joint_q: np.ndarray,
    joint_qd: np.ndarray,
    q_default_newton: np.ndarray,
    *,
    command: np.ndarray,
    last_action: np.ndarray,
    newton_to_lab: np.ndarray,
) -> np.ndarray:
    """The 48-wide policy observation from a Newton floating-base state (numpy reference for the Warp kernel).

    Args:
        joint_q: Newton generalized coordinates, ``[px, py, pz, qx, qy, qz, qw, q_0 .. q_11]`` (quaternion XYZW).
        joint_qd: ``[vx, vy, vz, wx, wy, wz, qd_0 .. qd_11]``, base velocities in the world frame.
        q_default_newton: default joint angles in Newton order.
        command: (vx, vy, yaw rate) in the body frame.
        last_action: previous policy output, policy (Isaac Lab) order.
        newton_to_lab: ``reorder_indices(newton_names, lab_names)``; maps Newton-ordered arrays to policy order.
            Newton's example hard-codes this array as ``lab_to_mujoco`` (named for the lookup, not the direction).
    """
    quat = joint_q[3:7]
    lin_b = quat_rotate_inverse_xyzw(quat, joint_qd[0:3])
    ang_b = quat_rotate_inverse_xyzw(quat, joint_qd[3:6])
    grav_b = quat_rotate_inverse_xyzw(quat, np.array([0.0, 0.0, -1.0]))
    q_rel = (joint_q[7:] - q_default_newton)[newton_to_lab]
    qd = joint_qd[6:][newton_to_lab]
    obs = np.concatenate([lin_b, ang_b, grav_b, np.asarray(command, float), q_rel, qd, np.asarray(last_action, float)])
    if obs.shape != (48,):
        raise ValueError(f"expected 48 observations, got {obs.shape}")
    return obs


def action_to_targets(
    action: np.ndarray, q_default_newton: np.ndarray, lab_to_newton: np.ndarray, scale: float = 0.5
) -> np.ndarray:
    """Joint position targets in Newton order: ``q_default + scale * action`` reordered from policy order.

    ``scale = 0.5`` is the action scale used by Isaac Lab's ANYmal velocity tasks and by Newton's example.
    ``lab_to_newton = reorder_indices(lab_names, newton_names)``; Newton's example calls it ``mujoco_to_lab``.
    """
    return q_default_newton + scale * np.asarray(action, float)[lab_to_newton]
