"""MDP terms for the push-object task: Isaac Lab's built-ins plus the task-specific terms in this package.

Same layout as the tasks shipped in ``isaaclab_tasks`` and as the ``isaaclab.sh --new`` template (tutorial #14):
``mdp`` re-exports ``isaaclab.envs.mdp`` so the env cfg can write ``mdp.joint_pos_rel`` and ``mdp.reach_object`` alike.

The Isaac Lab import is guarded so that ``math_terms`` stays importable on a CPU machine without Isaac Lab.
"""

try:
    from isaaclab.envs.mdp import *  # noqa: F403

    from .observations import *  # noqa: F403
    from .rewards import *  # noqa: F403
    from .terminations import *  # noqa: F403
except ImportError:  # CPU container: no Isaac Lab; only math_terms is usable
    ISAACLAB_AVAILABLE = False
else:
    ISAACLAB_AVAILABLE = True

from . import math_terms as math_terms
