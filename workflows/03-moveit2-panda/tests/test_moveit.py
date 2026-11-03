"""CPU checks: the Isaac graph, the MoveIt config fixes, and that both sides agree on topics and joints."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import make_patches  # noqa: E402
import moveit_graph_spec as spec  # noqa: E402
import ogspec  # noqa: E402

CONFIG = HERE / "moveit_config" / "config"


def test_graph_is_consistent():
    assert ogspec.validate(spec.joint_states_graph()) == []


def test_graph_topics_and_targets_match_tutorial_13():
    g = spec.joint_states_graph()
    assert g.node("publish_joint_state").values["topicName"] == "isaac_joint_states"
    assert g.node("subscribe_joint_state").values["topicName"] == "isaac_joint_commands"
    for name in ("publish_joint_state", "articulation_controller"):
        assert g.node(name).values["targetPrim"].paths == ("/World/Franka",)


def test_commands_flow_from_subscriber_to_articulation():
    g = spec.joint_states_graph()
    into = {e.dst_attr: e.src for e in g.edges_into("articulation_controller")}
    for attr in ("jointNames", "positionCommand", "velocityCommand"):
        assert into[f"inputs:{attr}"] == "subscribe_joint_state"
    assert g.edges_into("publish_joint_state", "timeStamp")[0].src == "read_sim_time"


@pytest.fixture(scope="module")
def controllers():
    return {
        "generated": yaml.safe_load((CONFIG / "moveit_controllers.generated.yaml").read_text()),
        "fixed": yaml.safe_load((CONFIG / "moveit_controllers.yaml").read_text()),
    }


def test_generated_controllers_reproduce_the_bug(controllers):
    arm = controllers["generated"]["moveit_simple_controller_manager"]["panda_arm_controller"]
    assert "action_ns" not in arm and "default" not in arm  # "Plan & Execute -> failed" in tutorial #13


def test_fixed_controllers_have_action_ns_and_default(controllers):
    mgr = controllers["fixed"]["moveit_simple_controller_manager"]
    arm, gripper = mgr["panda_arm_controller"], mgr["panda_gripper_controller"]
    assert arm["type"] == "FollowJointTrajectory"
    assert arm["action_ns"] == "follow_joint_trajectory" and arm["default"] is True
    assert arm["joints"] == list(spec.ARM_JOINTS)
    assert gripper["type"] == "GripperCommand"  # tutorial #13: MoveIt controller type for the gripper


def test_ros2_controllers_gripper_type():
    cfg = yaml.safe_load((CONFIG / "ros2_controllers.yaml").read_text())
    cm = cfg["controller_manager"]["ros__parameters"]
    assert cm["panda_gripper_controller"]["type"] == "position_controllers/GripperActionController"
    assert cfg["panda_arm_controller"]["ros__parameters"]["joints"] == list(spec.ARM_JOINTS)


def _hardware(path: Path) -> tuple[str, dict[str, str], list[str]]:
    root = ET.parse(path).getroot()
    hw = root.find(".//ros2_control/hardware")
    params = {p.get("name"): p.text for p in hw.findall("param")}
    joints = [j.get("name") for j in root.findall(".//ros2_control/joint")]
    return hw.find("plugin").text, params, joints


def test_xacro_patch_switches_to_topic_based_hardware():
    plugin, params, joints = _hardware(CONFIG / "panda.ros2_control.generated.xacro")
    assert plugin == "mock_components/GenericSystem" and params == {}
    plugin, params, joints = _hardware(CONFIG / "panda.ros2_control.xacro")
    assert plugin == "topic_based_ros2_control/TopicBasedSystem"
    # Both sides must name the same topics (tutorial #13: "confirm that the topic names ... match").
    assert params["joint_commands_topic"] == "/" + spec.JOINT_COMMANDS_TOPIC
    assert params["joint_states_topic"] == "/" + spec.JOINT_STATES_TOPIC
    assert set(spec.ARM_JOINTS) <= set(joints)


def test_patch_files_are_current():
    assert make_patches.main(["--check"]) == 0


def test_xacro_patch_adds_four_lines():
    text = (HERE / "patches" / "panda.ros2_control.xacro.patch").read_text()
    added = [ln for ln in text.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    assert len(added) == 4  # tutorial #13: "we need to add four lines"


def test_ogspec_copy_matches_workflow_02():
    other = HERE.parent / "02-isaac-sim-ros2-nav2" / "ogspec.py"
    if not other.exists():
        pytest.skip("workflow 02 not present")
    assert other.read_text() == (HERE / "ogspec.py").read_text()


def test_scripts_import_on_cpu():
    import build_joint_states_graph
    import send_joint_command

    assert build_joint_states_graph.parse_args(["--play"]).play
    assert len(send_joint_command.POSES["ready"]) == len(spec.ARM_JOINTS)
