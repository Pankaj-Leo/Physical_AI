"""Static checks on the GPU-only task package: what can be verified without Isaac Lab.

Parses the env cfg, agent cfg and registration with ``ast`` and checks that the tutorial #14 structure is present
(scene, actions, observations, rewards, terminations, events, commands), that the gym entry points resolve to classes
that exist, and that the package imports on a CPU machine (guarded imports).
"""

from __future__ import annotations

import ast
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import tomllib

FOLDER = Path(__file__).resolve().parents[1]
SRC = FOLDER / "source/atlas_rl_tasks"
TASK = SRC / "atlas_rl_tasks/tasks/push_object"


def _classes(path: Path) -> dict[str, ast.ClassDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, ast.ClassDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            found[node.name] = node
    return found


def _fields(cls: ast.ClassDef) -> set[str]:
    names = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            names |= {t.id for t in stmt.targets if isinstance(t, ast.Name)}
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
    return names


def test_env_cfg_has_every_manager_from_tutorial_14():
    cls = _classes(TASK / "push_object_env_cfg.py")
    env = _fields(cls["PushObjectEnvCfg"])
    assert {"scene", "observations", "actions", "commands", "rewards", "terminations", "events"} <= env
    assert {"robot", "table", "object", "ground", "light"} <= _fields(cls["PushObjectSceneCfg"])
    assert _fields(cls["ActionsCfg"]) == {"left_arm"}
    assert {"arm_joint_pos", "arm_joint_vel", "wrist_pos", "object_pos", "target", "last_action"} <= _fields(
        cls["PolicyCfg"]
    )
    assert {"reach", "progress", "success"} <= _fields(cls["RewardsCfg"])
    assert {"time_out", "object_dropped"} <= _fields(cls["TerminationsCfg"])
    # domain randomization: startup mass/friction, reset pose/joints/gains
    assert {"object_mass", "object_friction", "reset_object", "reset_arm", "arm_gains"} <= _fields(cls["EventCfg"])
    assert "object_pose" in _fields(cls["CommandsCfg"])


def test_reward_terms_reference_functions_that_exist():
    src = (TASK / "push_object_env_cfg.py").read_text(encoding="utf-8")
    rewards_py = (TASK / "mdp/rewards.py").read_text(encoding="utf-8")
    for name in ("reach_object", "ObjectGoalProgress", "object_at_goal"):
        assert f"mdp.{name}" in src
        assert f"def {name}" in rewards_py or f"class {name}" in rewards_py


def test_registration_entry_points_resolve_to_defined_classes():
    tree = ast.parse((TASK / "__init__.py").read_text(encoding="utf-8"))
    consts = {
        n.targets[0].id: n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.targets[0], ast.Name)
    }
    assert consts["TASK_ID"] == "Atlas-PushObject-G1-v0"
    assert consts["PLAY_TASK_ID"] == "Atlas-PushObject-G1-Play-v0"
    env_classes = _classes(TASK / "push_object_env_cfg.py")
    assert {"PushObjectEnvCfg", "PushObjectEnvCfg_PLAY"} <= set(env_classes)
    agent_classes = _classes(TASK / "agents/rsl_rl_ppo_cfg.py")
    assert "PushObjectPPORunnerCfg" in agent_classes
    src = (TASK / "__init__.py").read_text(encoding="utf-8")
    assert "isaaclab.envs:ManagerBasedRLEnv" in src
    assert "rsl_rl_cfg_entry_point" in src


def test_ppo_cfg_saves_every_50_iterations():
    cls = _classes(TASK / "agents/rsl_rl_ppo_cfg.py")["PushObjectPPORunnerCfg"]
    values = {
        s.targets[0].id: s.value.value
        for s in cls.body
        if isinstance(s, ast.Assign) and isinstance(s.value, ast.Constant) and isinstance(s.targets[0], ast.Name)
    }
    assert values["save_interval"] == 50  # tutorial #12, 04:41
    assert values["experiment_name"] == "atlas_push_object_g1"


def test_package_imports_on_cpu_without_isaaclab():
    sys.path.insert(0, str(SRC))
    try:
        for mod in [m for m in sys.modules if m.startswith("atlas_rl_tasks")]:
            del sys.modules[mod]
        pkg = importlib.import_module("atlas_rl_tasks")
        mdp = importlib.import_module("atlas_rl_tasks.tasks.push_object.mdp")
    finally:
        sys.path.remove(str(SRC))
    assert pkg.tasks.push_object.TASK_ID == "Atlas-PushObject-G1-v0"
    assert hasattr(mdp, "math_terms")
    assert mdp.ISAACLAB_AVAILABLE in {True, False}


def test_pyproject_and_extension_toml_parse():
    py = tomllib.loads((SRC / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["name"] == "atlas_rl_tasks"
    ext = tomllib.loads((SRC / "config/extension.toml").read_text(encoding="utf-8"))
    assert ext["python"]["module"][0]["name"] == "atlas_rl_tasks"


def test_shell_scripts_pass_bash_syntax_check():
    bash = shutil.which("bash")
    assert bash is not None
    scripts = sorted(FOLDER.glob("*.sh"))
    assert {p.name for p in scripts} == {"export_onnx.sh", "install_isaaclab.sh", "play.sh", "train.sh"}
    for script in scripts:
        subprocess.run([bash, "-n", str(script)], check=True)
