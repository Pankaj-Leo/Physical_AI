"""CPU checks of the compose file, DDS profile, Nav2 params, map fixer, and Actor SDG config."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "nav2"))

import build_graphs  # noqa: E402
import fix_map_yaml  # noqa: E402
import graph_spec as gs  # noqa: E402
import verify_topics  # noqa: E402

CARTER = gs.PROFILES["nova_carter"]


def load(rel: str):
    return yaml.safe_load((HERE / rel).read_text())


@pytest.fixture(scope="module")
def compose():
    return load("docker-compose.yml")


@pytest.fixture(scope="module")
def nav2():
    return load("nav2/nav2_params.yaml")


def test_compose_services_share_domain_and_host_network(compose):
    sim, brain = compose["services"]["isaac-sim"], compose["services"]["nav2"]
    assert sim["environment"]["ROS_DOMAIN_ID"] == brain["environment"]["ROS_DOMAIN_ID"]
    assert sim["environment"]["RMW_IMPLEMENTATION"] == brain["environment"]["RMW_IMPLEMENTATION"]
    assert sim["network_mode"] == brain["network_mode"] == "host"
    assert sim["image"].endswith(":5.0.0")  # pinned
    assert "jazzy" in brain["image"]
    # The DDS profile only applies to the Isaac side (wizard step 3).
    assert sim["environment"]["FASTRTPS_DEFAULT_PROFILES_FILE"].endswith("fastdds_udp_only.xml")
    assert "FASTRTPS_DEFAULT_PROFILES_FILE" not in brain["environment"]


def test_compose_domain_default_matches_ros2_context(compose):
    # ROS2Context in every graph: domain_id 0 with useDomainIDEnvVar True -> the env var decides; default 0.
    assert compose["services"]["isaac-sim"]["environment"]["ROS_DOMAIN_ID"] == "${ROS_DOMAIN_ID:-0}"
    for g in gs.all_graphs(CARTER):
        for n in g.nodes:
            if n.type == "ROS2Context":
                assert n.values == {"domain_id": 0, "useDomainIDEnvVar": True}


def test_dds_profile_is_udp_only():
    ns = {"f": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}
    root = ET.parse(HERE / "dds" / "fastdds_udp_only.xml").getroot()
    assert [t.text for t in root.iterfind(".//f:transport_descriptor/f:type", ns)] == ["UDPv4"]
    assert root.find(".//f:useBuiltinTransports", ns).text == "false"


def test_nav2_params_all_nodes_use_sim_time(nav2):
    def params(section):
        inner = nav2[section]
        return inner["ros__parameters"] if "ros__parameters" in inner else next(iter(inner.values()))["ros__parameters"]

    for section in nav2:
        assert params(section)["use_sim_time"] is True, section


def test_nav2_bringup_order_matches_wizard(nav2):
    order = (
        nav2["lifecycle_manager_localization"]["ros__parameters"]["node_names"]
        + nav2["lifecycle_manager_navigation"]["ros__parameters"]["node_names"]
    )
    # Wizard step 6: Map Server -> AMCL -> Planner -> Controller -> Behavior Tree -> Safety Monitor.
    wizard = ["map_server", "amcl", "planner_server", "controller_server", "bt_navigator", "collision_monitor"]
    positions = [order.index(n) for n in wizard]
    assert positions == sorted(positions)
    assert order[:2] == ["map_server", "amcl"]  # localization phase first; navigation waits for map -> odom


def test_nav2_frames_and_topics_match_graphs(nav2):
    amcl = nav2["amcl"]["ros__parameters"]
    assert amcl["base_frame_id"] == CARTER.base_frame
    assert amcl["odom_frame_id"] == gs.ODOM_FRAME
    assert amcl["scan_topic"] == gs.SCAN_TOPIC
    assert nav2["bt_navigator"]["ros__parameters"]["robot_base_frame"] == CARTER.base_frame
    local = nav2["local_costmap"]["local_costmap"]["ros__parameters"]
    assert local["global_frame"] == gs.ODOM_FRAME
    obs = local["obstacle_layer"]
    assert obs["scan"]["topic"] == "/" + gs.SCAN_TOPIC
    assert obs["point_cloud"]["topic"] == "/" + gs.POINT_CLOUD_TOPIC
    cm = nav2["collision_monitor"]["ros__parameters"]
    assert cm["cmd_vel_out_topic"] == gs.CMD_VEL_TOPIC  # what ROS2 Subscribe Twist listens to


def test_dwb_limits_within_differential_controller(nav2):
    dwb = nav2["controller_server"]["ros__parameters"]["FollowPath"]
    d = CARTER.diff_values
    assert dwb["max_vel_x"] <= d["maxLinearSpeed"]
    assert dwb["max_vel_theta"] <= d["maxAngularSpeed"]
    assert dwb["acc_lim_x"] <= d["maxAcceleration"]
    assert -dwb["decel_lim_x"] <= d["maxDeceleration"]
    assert dwb["acc_lim_theta"] <= d["maxAngularAcceleration"]


def test_launch_file_remaps_through_safety_monitor():
    text = (HERE / "nav2" / "bringup.launch.py").read_text()
    assert '("cmd_vel", "cmd_vel_nav")' in text
    assert '"--parent", "map", "--child", "odom"' in text


def test_fix_map_yaml_rewrites_image_path(tmp_path):
    src_dir = tmp_path / "pkg" / "maps"
    src_dir.mkdir(parents=True)
    (src_dir / "warehouse.png").write_bytes(b"\x89PNG fake")
    src = src_dir / "warehouse.yaml"
    src.write_text(
        yaml.safe_dump(
            {
                "image": "/home/someone/old/warehouse.png",  # hard-coded path from another machine
                "resolution": 0.05,
                "origin": [-11.975, -17.975, 0.0],
                "negate": 0,
                "occupied_thresh": 0.65,
                "free_thresh": 0.196,
            }
        )
    )
    out = tmp_path / "out"
    fixed = fix_map_yaml.fix_map_yaml(src, out, "/ws/nav2/maps")
    assert fixed["image"] == "/ws/nav2/maps/warehouse.png"
    assert (out / "warehouse.png").exists()
    assert yaml.safe_load((out / "warehouse.yaml").read_text())["image"] == "/ws/nav2/maps/warehouse.png"


def test_check_map_flags_bad_metadata():
    problems = fix_map_yaml.check_map({"image": "m.png", "resolution": 5, "origin": [0, 0], "negate": 0})
    text = " ".join(problems)
    assert "occupied_thresh" in text and "resolution" in text and "origin" in text


def test_verify_topics_reports_missing():
    listed = "/clock\n/tf\n/odom\n/cmd_vel\n/rosout\n"
    missing = verify_topics.missing_topics(verify_topics.expected_topics(), listed)
    assert "/scan" in missing and "/clock" not in missing


def test_actor_sdg_config():
    cfg = load("actor_sdg/people.yaml")
    body = next(iter(cfg.values()))
    assert body["character"]["num"] == 7  # tutorial #15 raised it from 1 to 7
    assert body["scene"]["asset_path"].endswith("warehouse_with_forklifts.usd")
    assert body["scene"]["asset_path"] == build_graphs.WAREHOUSE  # humans walk in the scene the robot drives in
