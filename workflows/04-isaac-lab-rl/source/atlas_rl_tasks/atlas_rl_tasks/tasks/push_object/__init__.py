"""Gym registration for the push-object task.

Tutorial #14 (08:15-08:26): "open the init.py file. Here we register the task and connect Isaac Lab to our environment
configuration and PPO configuration." Entry points are strings, so registering imports neither Isaac Lab nor torch;
the cfg modules load only when ``gym.make`` / Isaac Lab's ``parse_env_cfg`` resolves them on the GPU box.

The gymnasium import is guarded so the package imports on a CPU machine without Isaac Lab's environment.
"""

TASK_ID = "Atlas-PushObject-G1-v0"
PLAY_TASK_ID = "Atlas-PushObject-G1-Play-v0"

_ENV_CFG = f"{__name__}.push_object_env_cfg"
_AGENT_CFG = f"{__name__}.agents.rsl_rl_ppo_cfg:PushObjectPPORunnerCfg"

try:
    import gymnasium as gym
except ImportError:  # CPU container: nothing to register
    gym = None

if gym is not None:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{_ENV_CFG}:PushObjectEnvCfg", "rsl_rl_cfg_entry_point": _AGENT_CFG},
    )
    gym.register(
        id=PLAY_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{_ENV_CFG}:PushObjectEnvCfg_PLAY", "rsl_rl_cfg_entry_point": _AGENT_CFG},
    )
