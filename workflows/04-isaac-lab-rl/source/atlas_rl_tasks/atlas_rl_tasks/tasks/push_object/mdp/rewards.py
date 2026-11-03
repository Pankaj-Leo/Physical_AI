"""Reward terms for the push-object task, after tutorial #14's push-donut reward (transcript 09:16-09:56):

1. ``reach_object``     dense: wrist closer to the object.
2. ``ObjectGoalProgress`` dense: every small movement of the object toward the target.
3. ``object_at_goal``   sparse: a larger reward while the object is at the target.

Isaac Lab multiplies each term by ``weight * step_dt`` (RewardManager.compute), so weights in the env cfg are per
second, not per step. The math lives in ``math_terms`` (CPU-tested); these wrappers only fetch tensors.

not executed: needs GPU (Isaac Lab, Isaac Sim 5.0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import math_terms
from .observations import OBJECT, ROBOT, object_pos_in_root_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["ObjectGoalProgress", "object_at_goal", "reach_object"]


def reach_object(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = OBJECT,
) -> torch.Tensor:
    """1 - tanh(|wrist - object| / std); ``robot_cfg.body_names`` names the wrist body."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    wrist_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    return math_terms.reach_reward(wrist_w, obj.data.root_pos_w, std)


class ObjectGoalProgress(ManagerTermBase):
    """Stateful term: decrease of planar object-to-target distance since the previous policy step.

    A class term because progress needs the previous distance. ``reset`` marks environments as fresh instead of
    reading poses, because at reset time the event manager has written the new object pose but the scene buffers may
    not reflect it until the next physics step.  # verify: ordering of reset vs. buffer refresh in Isaac Lab 2.2
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev = torch.zeros(env.num_envs, device=env.device)
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._fresh[:] = True
        else:
            self._fresh[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        max_step: float = 0.02,
        robot_cfg: SceneEntityCfg = ROBOT,
        object_cfg: SceneEntityCfg = OBJECT,
    ) -> torch.Tensor:
        obj_b = object_pos_in_root_frame(env, robot_cfg, object_cfg)
        target_b = env.command_manager.get_command(command_name)[:, :3]
        dist = math_terms.distance(obj_b, target_b, planar=True)
        reward, self._prev = math_terms.progress_step(self._prev, dist, self._fresh, max_step)
        self._fresh[:] = False
        return reward


def object_at_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    robot_cfg: SceneEntityCfg = ROBOT,
    object_cfg: SceneEntityCfg = OBJECT,
) -> torch.Tensor:
    """1.0 while the object is within ``threshold`` (planar) of the commanded target, else 0.0."""
    obj_b = object_pos_in_root_frame(env, robot_cfg, object_cfg)
    target_b = env.command_manager.get_command(command_name)[:, :3]
    return math_terms.success_bonus(obj_b, target_b, threshold, planar=True)
