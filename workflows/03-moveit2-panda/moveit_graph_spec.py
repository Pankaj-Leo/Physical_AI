"""The Isaac Sim side of MoveIt 2 control for the Franka Panda (Isaac Sim tutorial #13), as pure data.

One action graph, ``ROS_JointStates`` (the tab name visible in ``moveit-isaac-sim-panda.jpg``):
- ROS2 Publish JointState on ``isaac_joint_states``: the arm's measured joint state, read by topic_based_ros2_control;
- ROS2 Subscribe JointState on ``isaac_joint_commands``: position + velocity targets from MoveIt's controllers,
  forwarded to an Articulation Controller whose target prim is the articulation root ``Franka``.

Tutorial #13 gets this graph two ways: the ``Robotics Examples > ROS2 > MoveIt > Franka MoveIt`` sample scene, or
``Tools > Robotics > ROS 2 OmniGraphs > Joint States`` with publisher and subscriber enabled. Both give the same graph.
Runs on CPU: data only. ``build_joint_states_graph.py`` applies it inside Isaac Sim.
"""

from __future__ import annotations

from ogspec import Graph, Node, PrimRef, graph

FRANKA_PRIM = "/World/Franka"  # tutorial #13: the target prim "is already set to Franka"
GRAPH_PATH = "/Graph/ROS_JointStates"  # verify: where the Joint States wizard puts the graph
JOINT_STATES_TOPIC = "isaac_joint_states"  # tutorial #13
JOINT_COMMANDS_TOPIC = "isaac_joint_commands"  # tutorial #13

ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))  # MoveIt group panda_arm (tutorial #13)
GRIPPER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")  # group panda_gripper (+ fixed panda_hand_joint)


def joint_states_graph(robot_prim: str = FRANKA_PRIM) -> Graph:
    return graph(
        GRAPH_PATH,
        [
            Node("on_playback_tick", "OnPlaybackTick"),
            Node("ros2_context", "ROS2Context", {"domain_id": 0, "useDomainIDEnvVar": True}),
            Node("read_sim_time", "IsaacReadSimulationTime"),
            Node(
                "publish_joint_state",
                "ROS2PublishJointState",
                {"topicName": JOINT_STATES_TOPIC, "targetPrim": PrimRef.to(robot_prim)},
            ),
            Node("subscribe_joint_state", "ROS2SubscribeJointState", {"topicName": JOINT_COMMANDS_TOPIC}),
            Node("articulation_controller", "IsaacArticulationController", {"targetPrim": PrimRef.to(robot_prim)}),
        ],
        [
            ("on_playback_tick.outputs:tick", "publish_joint_state.inputs:execIn"),
            ("on_playback_tick.outputs:tick", "subscribe_joint_state.inputs:execIn"),
            ("on_playback_tick.outputs:tick", "articulation_controller.inputs:execIn"),
            ("ros2_context.outputs:context", "publish_joint_state.inputs:context"),
            ("ros2_context.outputs:context", "subscribe_joint_state.inputs:context"),
            ("read_sim_time.outputs:simulationTime", "publish_joint_state.inputs:timeStamp"),
            ("subscribe_joint_state.outputs:jointNames", "articulation_controller.inputs:jointNames"),
            ("subscribe_joint_state.outputs:positionCommand", "articulation_controller.inputs:positionCommand"),
            ("subscribe_joint_state.outputs:velocityCommand", "articulation_controller.inputs:velocityCommand"),
            ("subscribe_joint_state.outputs:effortCommand", "articulation_controller.inputs:effortCommand"),
        ],
        evidence="tutorial #13 Part B/C; moveit-isaac-sim-panda.jpg (graph ROS_JointStates with On Playback Tick, "
        "ROS2 Context, Isaac Read Simulation Time, ROS2 Subscribe JointState)",
    )
