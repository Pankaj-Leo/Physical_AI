"""Run Isaac Lab's own rsl_rl ``train.py`` / ``play.py`` with this folder's task registered.

Isaac Lab's scripts import ``isaaclab_tasks`` (the built-in tasks) but not external packages. Tutorial #14 solves this
by shipping copies of train/play inside the task repo; the ``isaaclab.sh --new`` template does the same. This wrapper
avoids the copy: it imports ``atlas_rl_tasks`` (which only calls ``gym.register`` with string entry points, so no
Isaac Sim import happens before the app launches), then executes the unmodified Isaac Lab script as ``__main__`` with
the same command-line flags.

    python scripts/run_custom_task.py train --task Atlas-PushObject-G1-v0 --num_envs 4096 --headless
    python scripts/run_custom_task.py play  --task Atlas-PushObject-G1-Play-v0 --num_envs 5 --checkpoint <model.pt>

Requires ``ISAACLAB_PATH`` (default ``~/IsaacLab``) and the Isaac Lab conda env (Python 3.11).
not executed: needs GPU (Isaac Sim 5.0, Isaac Lab v2.2.x).
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_SRC = HERE.parent / "source" / "atlas_rl_tasks"


def isaaclab_script(isaaclab_path: Path, mode: str) -> Path:
    """Path of Isaac Lab's rsl_rl script for ``mode`` ("train" or "play"), as in tutorial #12's folder tour."""
    if mode not in {"train", "play"}:
        raise ValueError(f"mode must be 'train' or 'play', got {mode!r}")
    return isaaclab_path / "scripts" / "reinforcement_learning" / "rsl_rl" / f"{mode}.py"


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in {"train", "play"}:
        print(__doc__)
        return 2
    mode, rest = argv[0], argv[1:]
    if mode == "play" and "--checkpoint" not in rest:
        # Tutorial #14 (12:10): play without --checkpoint loaded, "but the robot is not moving".
        print("play needs --checkpoint <logs/rsl_rl/<experiment>/<run>/model_<iter>.pt> for a custom task")
        return 2
    isaaclab_path = Path(os.environ.get("ISAACLAB_PATH", Path.home() / "IsaacLab")).expanduser()
    script = isaaclab_script(isaaclab_path, mode)
    if not script.exists():
        print(f"not found: {script} (set ISAACLAB_PATH)")
        return 2
    sys.path.insert(0, str(PKG_SRC))  # works without pip install -e as well
    import atlas_rl_tasks  # noqa: F401  (registers the gym IDs)

    # Isaac Lab writes logs/ relative to the working directory; run from the Isaac Lab root like the tutorials.
    os.chdir(isaaclab_path)
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name="__main__")  # verify: train.py parses args at module level (v2.2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
