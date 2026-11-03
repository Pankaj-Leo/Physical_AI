"""rsl_rl PPO runner cfg for the push-object task.

Tutorials #12 and #14 train with rsl_rl (captioned "RSL_RL" in #12 and "Isaac RL" in #14). #12 shows a checkpoint
"after every 50 iterations" (04:41), hence ``save_interval = 50``. Hyperparameters start from Isaac Lab's Franka Lift
PPO cfg, a manipulation task of similar size; they were never tuned for this task.

Pinned: Isaac Lab v2.2.x with the rsl-rl-lib version its installer pins.
not executed: needs GPU (Isaac Lab).
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PushObjectPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50  # model_50.pt, model_100.pt, ... under logs/rsl_rl/<experiment_name>/<timestamp>/
    experiment_name = "atlas_push_object_g1"
    # verify: rsl-rl-lib >= 3.0 (Isaac Lab 2.3+) replaces this flag with policy.actor_obs_normalization /
    # critic_obs_normalization; on Isaac Lab 2.2 it is a runner field.
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
