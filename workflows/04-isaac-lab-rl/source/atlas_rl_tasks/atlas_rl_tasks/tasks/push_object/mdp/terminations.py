"""Termination terms for the push-object task.

``object_out_of_reach`` ends hopeless episodes early; ``object_reached_goal`` exists for evaluation scripts but is not
enabled during training (see the README gotcha on success terminations). Falling off the table uses Isaac Lab's
built-in ``root_height_below_minimum``.

not executed: needs GPU (Isaac Lab, Isaac Sim 5.0). Math in ``math_terms`` is CPU-tested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from . import math_terms
from .observations import OBJECT, ROBOT, object_pos_in_root_frame

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["object_out_of_reach", "object_reached_goal"]


def object_out_of_reach(
    env: ManagerBasedRLEnv,
    max_distance: float,
    robot_cfg: SceneEntityCfg = ROBOT,
    object_cfg: SceneEntityCfg = OBJECT,
) -> torch.Tensor:
    """True where the object is farther than ``max_distance`` (planar) from the robot root."""
    obj_b = object_pos_in_root_frame(env, robot_cfg, object_cfg)
    return math_terms.beyond_radius(obj_b, obj_b * 0.0, max_distance)


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    robot_cfg: SceneEntityCfg = ROBOT,
    object_cfg: SceneEntityCfg = OBJECT,
) -> torch.Tensor:
    """True where the object is within ``threshold`` of the target; for success-rate evaluation."""
    obj_b = object_pos_in_root_frame(env, robot_cfg, object_cfg)
    target_b = env.command_manager.get_command(command_name)[:, :3]
    return math_terms.at_goal(obj_b, target_b, threshold, planar=True)
