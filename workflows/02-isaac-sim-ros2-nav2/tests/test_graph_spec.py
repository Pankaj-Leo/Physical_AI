"""CPU checks of the SIL action graphs against the inspector screenshot and the tutorials."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import graph_spec as gs
import ogspec

CARTER = gs.PROFILES["nova_carter"]

# Transcribed by hand from assets/screens/nav2-omnigraph-inspector.jpg. Kept separate from graph_spec.py on purpose.
SCREENSHOT_NODES = {
    "on_playback_tick": "OnPlaybackTick",
    "ros2_context": "ROS2Context",
    "node_namespace": "ConstantString",
    "joint_wheel_left_name": "ConstantToken",
    "joint_wheel_right_name": "ConstantToken",
    "ros2_subscribe_twist": "ROS2SubscribeTwist",
    "joint_name_array": "ConstructArray",
    "break_3_vector": "BreakVector3",
    "break_3_vector_01": "BreakVector3",
    "differential_controller_01": "DifferentialController",
    "articulation_controller_01": "IsaacArticulationController",
}
SCREENSHOT_VALUES = {
    ("ros2_context", "domain_id"): 0,
    ("ros2_context", "useDomainIDEnvVar"): True,
    ("ros2_subscribe_twist", "topicName"): "cmd_vel",
    ("ros2_subscribe_twist", "queueSize"): 10,
    ("joint_name_array", "arraySize"): 2,
    ("joint_name_array", "arrayType"): "token[]",
    ("differential_controller_01", "wheelRadius"): 0.14,
    ("differential_controller_01", "wheelDistance"): 0.413,
    ("differential_controller_01", "maxLinearSpeed"): 1,
    ("differential_controller_01", "maxAngularSpeed"): 1.2,
    ("differential_controller_01", "maxAcceleration"): 2,
    ("differential_controller_01", "maxDeceleration"): 2,
    ("differential_controller_01", "maxAngularAcceleration"): 2.68,
    ("differential_controller_01", "maxWheelSpeed"): 0,
}


@pytest.mark.parametrize("profile", list(gs.PROFILES.values()), ids=list(gs.PROFILES))
def test_every_graph_is_internally_consistent(profile):
    for g in gs.all_graphs(profile):
        assert ogspec.validate(g) == [], g.path


def test_diff_drive_matches_screenshot_nodes_and_values():
    g = gs.differential_drive_graph(CARTER)
    assert g.path == "/World/Nova_Carter_ROS/differential_drive"  # graph path printed in the screenshot
    assert {n.name: n.type for n in g.nodes} == SCREENSHOT_NODES
    for (node, attr), expected in SCREENSHOT_VALUES.items():
        assert g.node(node).values[attr] == pytest.approx(expected), (node, attr)
    assert g.node("joint_wheel_left_name").values["value"].startswith("joint_wheel_")  # truncated in the shot
    assert g.node("joint_wheel_right_name").values["value"].startswith("joint_wheel_")


def test_diff_drive_dataflow_matches_screenshot_header():
    # Header: /cmd_vel (ROS2SubscribeTwist) -> BreakVector3 -> linear.x, angular.z -> DifferentialController ->
    # IsaacArticulationController -> wheel joints.
    g = gs.differential_drive_graph(CARTER)
    into = {(e.dst, e.dst_attr): (e.src, e.src_attr) for e in g.edges}
    lin = into["differential_controller_01", "inputs:linearVelocity"]
    ang = into["differential_controller_01", "inputs:angularVelocity"]
    assert lin[1] == "outputs:x" and ang[1] == "outputs:z"
    assert into[lin[0], "inputs:tuple"] == ("ros2_subscribe_twist", "outputs:linearVelocity")
    assert into[ang[0], "inputs:tuple"] == ("ros2_subscribe_twist", "outputs:angularVelocity")
    assert into["articulation_controller_01", "inputs:velocityCommand"] == (
        "differential_controller_01",
        "outputs:velocityCommand",
    )
    # dt reads 0.0167 in the screenshot, the same as on_playback_tick.deltaSeconds.
    assert into["differential_controller_01", "inputs:dt"] == ("on_playback_tick", "outputs:deltaSeconds")
    # Every node that does work is ticked.
    for n in ("ros2_subscribe_twist", "differential_controller_01", "articulation_controller_01"):
        assert into[n, "inputs:execIn"] == ("on_playback_tick", "outputs:tick")


def test_articulation_target_is_the_profile_root():
    for p in gs.PROFILES.values():
        ref = gs.differential_drive_graph(p).node("articulation_controller_01").values["targetPrim"]
        assert ref.paths == (p.articulation_root,)
    # Tutorial #6: TurtleBot3 works only with the root on the top-level robot prim.
    tb3 = gs.PROFILES["turtlebot3"]
    assert tb3.articulation_root == tb3.robot_root


def test_turtlebot3_values_from_tutorial_6():
    d = gs.differential_drive_graph(gs.PROFILES["turtlebot3"]).node("differential_controller_01").values
    assert d == pytest.approx({"wheelRadius": 0.033, "wheelDistance": 0.160, "maxLinearSpeed": 0.22})


def test_imu_uses_sim_time_not_sensor_time():
    g = gs.imu_graph(CARTER)
    assert not [e for e in g.edges if e.src_attr == "outputs:sensorTime"]
    assert g.edges_into("publish_imu", "timeStamp")[0].src == "sim_time"


def test_laser_scan_skips_vertical_fields_and_uses_sim_lidar():
    g = gs.lidar2d_graph(CARTER)
    wired = {e.src_attr for e in g.edges if e.src == "read_lidar"}
    assert "outputs:verticalFov" not in wired and "outputs:verticalResolution" not in wired
    pub = g.node("publish_scan").values
    assert pub == {"topicName": "scan", "frameId": "sim_lidar"}


def test_3d_lidar_is_a_render_product_and_shares_the_2d_frame():
    g = gs.lidar3d_graph(CARTER)
    types = [n.type for n in g.nodes]
    assert "IsaacRunOneSimulationFrame" in types and "IsaacCreateRenderProduct" in types
    helper = g.node("rtx_lidar_helper").values
    assert helper["topicName"] == "point_cloud" and helper["frameId"] == gs.LIDAR_FRAME
    assert helper["type"] == "point_cloud"


def test_published_topics_cover_what_nav2_needs():
    topics = gs.published_topics(CARTER)
    assert {"/clock", "/odom", "/tf", "/scan", "/point_cloud", "/rgb", "/camera_info", "/imu"} <= topics


@pytest.mark.parametrize("profile", list(gs.PROFILES.values()), ids=list(gs.PROFILES))
def test_tf_tree_is_one_connected_tree(profile):
    parents = gs.tf_edges(profile)  # raises if any frame has two parents
    parents["odom"] = "map"  # AMCL
    for frame in [gs.LIDAR_FRAME, gs.CAMERA_FRAME, gs.IMU_FRAME, profile.base_frame]:
        seen, f = set(), frame
        while f != "map":
            assert f not in seen, f"cycle at {f}"
            seen.add(f)
            assert f in parents, f"{frame} is not connected to map (stops at {f})"
            f = parents[f]


def test_no_world_to_odom_transform():
    # The tutorial #11 wizard adds world -> odom; with AMCL publishing map -> odom that would give odom two parents.
    assert "odom" not in gs.tf_edges(CARTER)


def test_controller_plan_creates_dynamic_array_input():
    plan = ogspec.controller_plan(gs.differential_drive_graph(CARTER))
    assert ("joint_name_array.inputs:input1", "token") in plan["CREATE_ATTRIBUTES"]
    assert ("articulation_controller_01.inputs:targetPrim", ogspec.PrimRef.to(CARTER.articulation_root)) in plan[
        "SET_PRIMS"
    ]
    assert len(plan["CONNECT"]) == len(gs.differential_drive_graph(CARTER).edges)


def test_validate_catches_bad_wiring():
    bad = ogspec.graph(
        "/bad",
        [ogspec.Node("t", "OnPlaybackTick"), ogspec.Node("c", "DifferentialController", {"dt": 0.1})],
        [("t.outputs:tick", "c.inputs:nope"), ("t.outputs:deltaSeconds", "c.inputs:dt"), ("t.outputs:x", "z.inputs:y")],
        evidence="test",
    )
    problems = ogspec.validate(bad)
    assert any("c.inputs:nope" in p for p in problems)
    assert any("both connected and set" in p for p in problems)
    assert any("undefined node" in p for p in problems)


def test_build_script_imports_on_cpu():
    import build_graphs

    args = build_graphs.parse_args(["--robot", "turtlebot3", "--headless"])
    assert args.robot == "turtlebot3" and args.mode == "build"
