"""Observation terms: wrist and object positions in the robot root frame (tutorial #14: "wrist position, and object
position", transcript 09:03-09:16).

not executed: needs GPU (Isaac Lab, Isaac Sim 5.0). The frame math is ``math_terms.pos_in_frame``, tested on CPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from . import math_terms

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["object_pos_in_root_frame", "wrist_pos_in_root_frame"]

# Module-level defaults (Isaac Lab terms usually write SceneEntityCfg("robot") inline as a default argument).
ROBOT = SceneEntityCfg("robot")
OBJECT = SceneEntityCfg("object")


def wrist_pos_in_root_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """(N, 3) position of the first body in ``asset_cfg.body_names`` (the wrist) in the robot root frame."""
    robot = env.scene[asset_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :]  # verify: body_ids is a list after resolve()
    return math_terms.pos_in_frame(robot.data.root_pos_w, robot.data.root_quat_w, body_pos_w)


def object_pos_in_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = ROBOT,
    object_cfg: SceneEntityCfg = OBJECT,
) -> torch.Tensor:
    """(N, 3) object position in the robot root frame (the frame the target command is expressed in)."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    return math_terms.pos_in_frame(robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w)
