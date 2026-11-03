"""Nav2 bring-up in the wizard's order: localization first, wait for map -> odom, then navigation.

Not executed: needs the ROS 2 Jazzy container and a running Isaac Sim (see ../docker-compose.yml).

Wizard step 6: "Start localization, wait for map->odom, then navigation - headless. All six Nav2 nodes should reach
'active'." Order: map server -> AMCL -> planner -> controller -> behavior tree -> safety monitor. The transcript gives
the reason: costmaps need the map frame; started too early they "sit there waiting for the map frame and then give up".

    ros2 launch /ws/nav2/bringup.launch.py map:=/ws/nav2/maps/carter_warehouse_navigation.yaml [rviz:=true]

The node order inside each phase lives in nav2_params.yaml (lifecycle manager ``node_names``), so tests can check it.
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo, RegisterEventHandler
    from launch.conditions import IfCondition
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node

    params = LaunchConfiguration("params")
    map_yaml = LaunchConfiguration("map")
    sim_time = {"use_sim_time": True}  # /clock comes from the ros_clock graph in Isaac Sim
    to_safety = [("cmd_vel", "cmd_vel_nav")]  # controller/behaviors -> collision_monitor -> cmd_vel -> Isaac

    def nav2(package: str, executable: str, name: str, remaps=()) -> Node:
        return Node(
            package=package,
            executable=executable,
            name=name,
            output="screen",
            parameters=[params, sim_time],
            remappings=list(remaps),
        )

    localization = [
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[params, sim_time, {"yaml_filename": map_yaml}],
        ),
        nav2("nav2_amcl", "amcl", "amcl"),
        nav2("nav2_lifecycle_manager", "lifecycle_manager", "lifecycle_manager_localization"),
    ]

    navigation = [
        nav2("nav2_planner", "planner_server", "planner_server"),
        nav2("nav2_controller", "controller_server", "controller_server", to_safety),
        nav2("nav2_behaviors", "behavior_server", "behavior_server", to_safety),
        nav2("nav2_bt_navigator", "bt_navigator", "bt_navigator"),
        nav2("nav2_collision_monitor", "collision_monitor", "collision_monitor"),
        nav2("nav2_lifecycle_manager", "lifecycle_manager", "lifecycle_manager_navigation"),
    ]

    wait_for_map_odom = ExecuteProcess(
        cmd=["python3", os.path.join(HERE, "wait_for_transform.py"), "--parent", "map", "--child", "odom"],
        output="screen",
    )

    def after_wait(event, _context):
        if event.returncode == 0:
            return [LogInfo(msg="map -> odom is up; starting navigation"), *navigation]
        return [
            LogInfo(msg="map -> odom never appeared (AMCL initial pose? /scan? /clock?)"),
            EmitEvent(event=Shutdown()),
        ]

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", "/opt/ros/jazzy/share/nav2_bringup/rviz/nav2_default_view.rviz"],
        parameters=[sim_time],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("params", default_value=os.path.join(HERE, "nav2_params.yaml")),
            DeclareLaunchArgument("map", default_value=os.path.join(HERE, "maps", "carter_warehouse_navigation.yaml")),
            DeclareLaunchArgument("rviz", default_value="false"),
            *localization,
            wait_for_map_odom,
            RegisterEventHandler(OnProcessExit(target_action=wait_for_map_odom, on_exit=after_wait)),
            rviz,
        ]
    )
