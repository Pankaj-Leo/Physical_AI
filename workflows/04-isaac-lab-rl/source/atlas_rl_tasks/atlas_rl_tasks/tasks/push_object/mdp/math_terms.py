"""Pure reward, termination, and frame math for the push-object task.

Tutorial #14 (Isaac Lab custom humanoid task) describes the push task's reward in three parts: "we reward the robot for
moving its wrist closer to the object", "we reward every small movement that pushes the object toward the target", and
"we provide a larger reward when the object successfully reaches the target position" (transcript 09:31-09:47). The
functions below are that math, kept free of Isaac Lab so it can be unit-tested on a CPU.

Array-API style: every function accepts array-likes whose last axis is xyz (shape ``(num_envs, 3)``) and returns an
array of shape ``(num_envs,)``. Inside Isaac Lab the inputs are torch tensors on the GPU; in this repo's tests they are
numpy arrays. Only operations that numpy and torch share are used, plus a small namespace shim for ``tanh``, ``sqrt``,
``clip`` and ``stack``.

Executed here: numpy path (tests/test_math_terms.py). Not executed: torch path (needs Isaac Lab on a GPU box).

Quaternions follow Isaac Lab's convention, (w, x, y, z). Newton uses (x, y, z, w); see ../../../../../sim_to_sim.md.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def namespace(*arrays: Any) -> Any:
    """Return an array namespace for the inputs: array_api_compat if installed, else torch or numpy."""
    try:
        import array_api_compat

        return array_api_compat.array_namespace(*arrays)
    except (ImportError, TypeError):
        pass
    if any(type(a).__module__.startswith("torch") for a in arrays):
        import torch

        return SimpleNamespace(
            sqrt=torch.sqrt,
            tanh=torch.tanh,
            clip=torch.clamp,
            stack=lambda xs, axis=0: torch.stack(list(xs), dim=axis),
        )
    import numpy as np

    return np


def distance(a: Any, b: Any, planar: bool = False) -> Any:
    """Euclidean distance between point sets along the last axis; ``planar`` drops z (table-top pushing)."""
    d = a - b
    if planar:
        d = d[..., :2]
    xp = namespace(d)
    return xp.sqrt((d * d).sum(-1))


def reach_reward(wrist_pos: Any, object_pos: Any, std: float = 0.1) -> Any:
    """Dense "wrist closer to object" term: 1 - tanh(d / std), in (0, 1], 1 at contact.

    Same kernel shape as Isaac Lab's ``object_ee_distance`` in the Lift task. ``std`` sets the length scale where the
    gradient is steepest; 0.1 m means the term is about 0.24 at 10 cm and 0.76 at 2.5 cm.
    """
    if std <= 0:
        raise ValueError("std must be positive")
    d = distance(wrist_pos, object_pos)
    xp = namespace(d)
    return 1.0 - xp.tanh(d / std)


def progress_step(prev_dist: Any, curr_dist: Any, fresh: Any, max_step: float = 0.02) -> tuple[Any, Any]:
    """Dense "object toward target" term: the per-step decrease of object-to-target distance.

    Args:
        prev_dist: object-to-target distance at the previous policy step, shape (N,).
        curr_dist: distance now, shape (N,).
        fresh: boolean mask (N,), True for environments that were reset since the last call; their previous distance
            is stale, so they get zero progress this step.
        max_step: clip in metres per step. A reset teleports the object; without the clip a stale distance would pay
            or charge a spike. 0.02 m per 20 ms step is 1 m/s, faster than a push should be.

    Returns:
        (reward, new_prev): reward is positive when the object moved toward the target; new_prev is ``curr_dist``.
    """
    if max_step <= 0:
        raise ValueError("max_step must be positive")
    xp = namespace(prev_dist, curr_dist)
    delta = xp.clip(prev_dist - curr_dist, -max_step, max_step)
    reward = delta * (1.0 - fresh * 1.0)
    return reward, curr_dist


def at_goal(object_pos: Any, target_pos: Any, threshold: float = 0.03, planar: bool = True) -> Any:
    """Boolean (N,): object within ``threshold`` metres of the target (planar by default, the object slides)."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return distance(object_pos, target_pos, planar=planar) < threshold


def success_bonus(object_pos: Any, target_pos: Any, threshold: float = 0.03, planar: bool = True) -> Any:
    """Sparse success term as float (N,): 1.0 while the object sits at the target, else 0.0."""
    return at_goal(object_pos, target_pos, threshold, planar) * 1.0


def below_height(z: Any, min_z: float) -> Any:
    """Termination (N,): the object fell below ``min_z`` (off the table)."""
    return z < min_z


def beyond_radius(pos: Any, center: Any, max_distance: float) -> Any:
    """Termination (N,): planar distance from ``center`` exceeds ``max_distance`` (object pushed out of reach)."""
    return distance(pos, center, planar=True) > max_distance


def quat_apply_inverse_wxyz(q: Any, v: Any) -> Any:
    """Rotate vectors ``v`` (N, 3) by the inverse of unit quaternions ``q`` (N, 4) in (w, x, y, z) order.

    Uses v' = v + 2 w (u x v) + 2 u x (u x v) with u = -q_xyz (the conjugate), which avoids building a matrix.
    """
    w = q[..., 0:1]
    u = -q[..., 1:4]
    t = 2.0 * _cross(u, v)
    return v + w * t + _cross(u, t)


def pos_in_frame(frame_pos: Any, frame_quat_wxyz: Any, p: Any) -> Any:
    """Express world points ``p`` in a frame given by its world position and (w, x, y, z) orientation."""
    return quat_apply_inverse_wxyz(frame_quat_wxyz, p - frame_pos)


def _cross(a: Any, b: Any) -> Any:
    xp = namespace(a, b)
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return xp.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)
