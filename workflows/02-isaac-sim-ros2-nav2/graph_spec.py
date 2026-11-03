"""The software-in-the-loop action graphs for a differential-drive robot, as pure data.

Isaac Sim is the body (physics + sensors); a ROS 2 container running Nav2 is the brain. These graphs are the nerves:
``/cmd_vel`` in, ``/clock``, ``/odom``, ``/tf``, ``/scan``, ``/point_cloud``, ``/rgb``, ``/camera_info``, ``/imu`` out.

- ``differential_drive_graph`` reproduces node for node the live OmniGraph inspector screenshot from the Nav2 wizard
  (``assets/screens/nav2-omnigraph-inspector.jpg``, graph ``/World/Nova_Carter_ROS/differential_drive``).
- The sensor, odometry and TF graphs follow Isaac Sim tutorials #10 (Parts 1-4) and #11.
- The clock graph is not in any tutorial; Nav2 needs it because every node runs with ``use_sim_time: true``.

Runs on CPU: this module only builds data. ``build_graphs.py`` applies it inside Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ogspec import Graph, Node, PrimRef, graph

# Topic and frame names. Topics follow the tutorials; frames are ours but must agree with nav2/nav2_params.yaml.
CMD_VEL_TOPIC = "cmd_vel"  # tutorial #6, screenshot
CLOCK_TOPIC = "clock"
ODOM_TOPIC = "odom"  # tutorial #11
TF_TOPIC = "tf"  # tutorial #11
SCAN_TOPIC = "scan"  # tutorial #10.3
POINT_CLOUD_TOPIC = "point_cloud"  # tutorial #10.4 renames it to avoid a clash with /scan
RGB_TOPIC = "rgb"  # tutorial #10.1
CAMERA_INFO_TOPIC = "camera_info"  # tutorial #10.1
IMU_TOPIC = "imu"  # tutorial #10.2

ODOM_FRAME = "odom"
LIDAR_FRAME = "sim_lidar"  # tutorial #10.3 frame_id; 2D and 3D LiDAR share it (tutorial #10.4)
CAMERA_FRAME = "sim_camera"
IMU_FRAME = "sim_imu"


@dataclass(frozen=True)
class RobotProfile:
    key: str
    robot_root: str  # USD path where the robot asset is referenced
    articulation_root: str  # the prim that carries PhysicsArticulationRootAPI; the controller's targetPrim
    chassis_prim: str  # parent of the sensor mounts; its prim name becomes a TF frame
    base_frame: str  # Nav2 robot_base_frame and odometry child frame
    left_joint: str
    right_joint: str
    diff_values: dict[str, Any]  # DifferentialController inputs
    asset: str  # path under the Isaac assets root (verify)
    lidar_mount_xyz: tuple[float, float, float]  # verify against your robot
    evidence: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def mount(self, name: str) -> str:
        return f"{self.chassis_prim}/{name}"


PROFILES: dict[str, RobotProfile] = {
    # Values from the live inspector screenshot (Nova Carter, NVIDIA's ROS 2 navigation sample).
    "nova_carter": RobotProfile(
        key="nova_carter",
        robot_root="/World/Nova_Carter_ROS",
        articulation_root="/World/Nova_Carter_ROS/chassis_link",  # B.T. 07 targets chassis_link
        chassis_prim="/World/Nova_Carter_ROS/chassis_link",
        base_frame="base_link",
        left_joint="joint_wheel_left",  # screenshot shows "joint_wheel_..." truncated; full names from B.T. 07
        right_joint="joint_wheel_right",
        diff_values={
            "wheelRadius": 0.14,
            "wheelDistance": 0.413,
            "maxLinearSpeed": 1.0,
            "maxAngularSpeed": 1.2,
            "maxAcceleration": 2.0,
            "maxDeceleration": 2.0,
            "maxAngularAcceleration": 2.68,
            "maxWheelSpeed": 0.0,
        },
        asset="/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",  # verify
        lidar_mount_xyz=(0.0, 0.0, 0.5),  # verify: our choice, not from a source
        evidence="nav2-omnigraph-inspector.jpg; Omniverse B.T. 07",
    ),
    # Values from tutorial #6 Part 1 (TurtleBot3 spec page, in meters).
    "turtlebot3": RobotProfile(
        key="turtlebot3",
        robot_root="/World/turtlebot3_burger",
        articulation_root="/World/turtlebot3_burger",  # tutorial #6: the top-level robot prim, not base_link
        chassis_prim="/World/turtlebot3_burger/base_link",
        base_frame="base_link",
        left_joint="wheel_left_joint",  # verify against your import
        right_joint="wheel_right_joint",
        diff_values={"wheelRadius": 0.033, "wheelDistance": 0.160, "maxLinearSpeed": 0.22},
        asset="/Isaac/Robots/Turtlebot/turtlebot3_burger.usd",  # verify
        lidar_mount_xyz=(-0.032, 0.0, 0.172),  # verify: TurtleBot3 base_scan offset from its URDF, not from a source
        evidence="tutorial #6 Part 1, tutorial #11",
    ),
}


def _leaf(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]


def differential_drive_graph(p: RobotProfile) -> Graph:
    """/cmd_vel -> wheel joint velocities, exactly the nodes in the inspector screenshot (names included)."""
    return graph(
        f"{p.robot_root}/differential_drive",
        [
            Node("on_playback_tick", "OnPlaybackTick"),
            Node("ros2_context", "ROS2Context", {"domain_id": 0, "useDomainIDEnvVar": True}),
            Node("node_namespace", "ConstantString", {"value": ""}),
            Node("joint_wheel_left_name", "ConstantToken", {"value": p.left_joint}),
            Node("joint_wheel_right_name", "ConstantToken", {"value": p.right_joint}),
            Node("ros2_subscribe_twist", "ROS2SubscribeTwist", {"topicName": CMD_VEL_TOPIC, "queueSize": 10}),
            Node("joint_name_array", "ConstructArray", {"arraySize": 2, "arrayType": "token[]"}),
            Node("break_3_vector", "BreakVector3"),
            Node("break_3_vector_01", "BreakVector3"),
            Node("differential_controller_01", "DifferentialController", dict(p.diff_values)),
            Node(
                "articulation_controller_01",
                "IsaacArticulationController",
                {"targetPrim": PrimRef.to(p.articulation_root)},
            ),
        ],
        [
            ("on_playback_tick.outputs:tick", "ros2_subscribe_twist.inputs:execIn"),
            ("on_playback_tick.outputs:tick", "differential_controller_01.inputs:execIn"),
            ("on_playback_tick.outputs:tick", "articulation_controller_01.inputs:execIn"),
            # dt shows 0.0167 = deltaSeconds in the screenshot (tutorial #6 says "time"; the screenshot wins).
            ("on_playback_tick.outputs:deltaSeconds", "differential_controller_01.inputs:dt"),
            ("ros2_context.outputs:context", "ros2_subscribe_twist.inputs:context"),
            ("node_namespace.inputs:value", "ros2_subscribe_twist.inputs:nodeNamespace"),
            # Wire routing read from the screenshot: angular -> break_3_vector, linear -> break_3_vector_01.
            ("ros2_subscribe_twist.outputs:angularVelocity", "break_3_vector.inputs:tuple"),
            ("ros2_subscribe_twist.outputs:linearVelocity", "break_3_vector_01.inputs:tuple"),
            ("break_3_vector.outputs:z", "differential_controller_01.inputs:angularVelocity"),
            ("break_3_vector_01.outputs:x", "differential_controller_01.inputs:linearVelocity"),
            ("joint_wheel_left_name.inputs:value", "joint_name_array.inputs:input0"),
            ("joint_wheel_right_name.inputs:value", "joint_name_array.inputs:input1"),
            ("joint_name_array.outputs:array", "articulation_controller_01.inputs:jointNames"),
            ("differential_controller_01.outputs:velocityCommand", "articulation_controller_01.inputs:velocityCommand"),
        ],
        evidence="nav2-omnigraph-inspector.jpg (all 11 visible nodes and values); tutorial #6 Part 1; B.T. 07",
    )


def _tick_ctx_time() -> list[Node]:
    return [
        Node("on_playback_tick", "OnPlaybackTick"),
        Node("ros2_context", "ROS2Context", {"domain_id": 0, "useDomainIDEnvVar": True}),
        Node("sim_time", "IsaacReadSimulationTime"),
    ]


def clock_graph(p: RobotProfile) -> Graph:
    """/clock from simulation time, so Nav2 can run with use_sim_time."""
    return graph(
        f"{p.robot_root}/ros_clock",
        [*_tick_ctx_time(), Node("publish_clock", "ROS2PublishClock", {"topicName": CLOCK_TOPIC})],
        [
            ("on_playback_tick.outputs:tick", "publish_clock.inputs:execIn"),
            ("ros2_context.outputs:context", "publish_clock.inputs:context"),
            ("sim_time.outputs:simulationTime", "publish_clock.inputs:timeStamp"),
        ],
        evidence="Nav2 wizard step 4 (clock is one of the topics that matter); not built in any tutorial",
    )


def odometry_graph(p: RobotProfile) -> Graph:
    """/odom and odom -> base_frame on /tf, as the tutorial #11 Odometry Publisher wizard builds it.

    The wizard also adds a world -> odom raw transform. We leave it out: with Nav2, AMCL publishes map -> odom, and a
    second parent for ``odom`` breaks the TF tree.
    """
    return graph(
        f"{p.robot_root}/ros_odometry",
        [
            *_tick_ctx_time(),
            Node("compute_odometry", "IsaacComputeOdometry", {"chassisPrim": PrimRef.to(p.articulation_root)}),
            Node(
                "publish_odometry",
                "ROS2PublishOdometry",
                {"topicName": ODOM_TOPIC, "odomFrameId": ODOM_FRAME, "chassisFrameId": p.base_frame},
            ),
            Node(
                "publish_odom_tf",
                "ROS2PublishRawTransformTree",
                {"topicName": TF_TOPIC, "parentFrameId": ODOM_FRAME, "childFrameId": p.base_frame},
            ),
        ],
        [
            ("on_playback_tick.outputs:tick", "compute_odometry.inputs:execIn"),
            ("compute_odometry.outputs:execOut", "publish_odometry.inputs:execIn"),
            ("compute_odometry.outputs:execOut", "publish_odom_tf.inputs:execIn"),
            ("ros2_context.outputs:context", "publish_odometry.inputs:context"),
            ("ros2_context.outputs:context", "publish_odom_tf.inputs:context"),
            ("sim_time.outputs:simulationTime", "publish_odometry.inputs:timeStamp"),
            ("sim_time.outputs:simulationTime", "publish_odom_tf.inputs:timeStamp"),
            ("compute_odometry.outputs:position", "publish_odometry.inputs:position"),
            ("compute_odometry.outputs:orientation", "publish_odometry.inputs:orientation"),
            ("compute_odometry.outputs:linearVelocity", "publish_odometry.inputs:linearVelocity"),
            ("compute_odometry.outputs:angularVelocity", "publish_odometry.inputs:angularVelocity"),
            ("compute_odometry.outputs:position", "publish_odom_tf.inputs:translation"),
            ("compute_odometry.outputs:orientation", "publish_odom_tf.inputs:rotation"),
        ],
        evidence="tutorial #11 (Odometry Publisher wizard: ROS2 Publish Odometry + Raw Transform Tree)",
    )


def tf_graph(p: RobotProfile) -> Graph:
    """Chassis -> sensor mounts from the prim hierarchy (tutorial #11 TF Publisher), plus base_frame -> chassis alias.

    ROS2 Publish Transform Tree names frames after prims (verify), so the chassis frame is the chassis prim's name.
    When that differs from ``base_frame`` (Nova Carter: ``chassis_link`` vs ``base_link``), an identity raw transform
    joins the two so the tree stays connected.
    """
    chassis = _leaf(p.chassis_prim)
    nodes = [
        *_tick_ctx_time(),
        Node(
            "publish_sensor_tf",
            "ROS2PublishTransformTree",
            {
                "topicName": TF_TOPIC,
                "parentPrim": PrimRef.to(p.chassis_prim),
                "targetPrims": PrimRef.to(p.mount(LIDAR_FRAME), p.mount(CAMERA_FRAME), p.mount(IMU_FRAME)),
            },
        ),
    ]
    edges = [
        ("on_playback_tick.outputs:tick", "publish_sensor_tf.inputs:execIn"),
        ("ros2_context.outputs:context", "publish_sensor_tf.inputs:context"),
        ("sim_time.outputs:simulationTime", "publish_sensor_tf.inputs:timeStamp"),
    ]
    if chassis != p.base_frame:
        nodes.append(
            Node(
                "publish_base_alias",
                "ROS2PublishRawTransformTree",
                {
                    "topicName": TF_TOPIC,
                    "parentFrameId": p.base_frame,
                    "childFrameId": chassis,
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [1.0, 0.0, 0.0, 0.0],  # verify: quaternion order of this input (IJKR vs WXYZ)
                },
            )
        )
        edges += [
            ("on_playback_tick.outputs:tick", "publish_base_alias.inputs:execIn"),
            ("ros2_context.outputs:context", "publish_base_alias.inputs:context"),
            ("sim_time.outputs:simulationTime", "publish_base_alias.inputs:timeStamp"),
        ]
    return graph(f"{p.robot_root}/ros_tf", nodes, edges, evidence="tutorial #11 (TF Publisher wizard)")


def lidar2d_graph(p: RobotProfile) -> Graph:
    """PhysX rotating LiDAR -> sensor_msgs/LaserScan on /scan (tutorial #10.3, no wizard exists)."""
    beams = [
        "azimuthRange",
        "depthRange",
        "horizontalFov",
        "horizontalResolution",
        "intensitiesData",
        "linearDepthData",
        "numCols",
        "numRows",
        "rotationRate",
    ]  # tutorial #10.3: wire everything except verticalFov / verticalResolution
    return graph(
        f"{p.robot_root}/ros_lidar_2d",
        [
            *_tick_ctx_time(),
            Node("read_lidar", "IsaacReadLidarBeams", {"lidarPrim": PrimRef.to(p.mount(LIDAR_FRAME) + "/physx_2d")}),
            Node("publish_scan", "ROS2PublishLaserScan", {"topicName": SCAN_TOPIC, "frameId": LIDAR_FRAME}),
        ],
        [
            ("on_playback_tick.outputs:tick", "read_lidar.inputs:execIn"),
            ("read_lidar.outputs:execOut", "publish_scan.inputs:execIn"),
            ("ros2_context.outputs:context", "publish_scan.inputs:context"),
            ("sim_time.outputs:simulationTime", "publish_scan.inputs:timeStamp"),
            *[(f"read_lidar.outputs:{b}", f"publish_scan.inputs:{b}") for b in beams],
        ],
        evidence="tutorial #10.3 (Isaac Read Lidar Beams -> ROS2 Publish Laser Scan, topic scan, frame sim_lidar)",
    )


def _render_product_graph(path: str, camera_prim: str, size: tuple[int, int], publishers: list[Node], ev: str) -> Graph:
    nodes = [
        Node("on_playback_tick", "OnPlaybackTick"),
        Node("ros2_context", "ROS2Context", {"domain_id": 0, "useDomainIDEnvVar": True}),
        Node("run_one_frame", "IsaacRunOneSimulationFrame"),
        Node(
            "render_product",
            "IsaacCreateRenderProduct",
            {"cameraPrim": PrimRef.to(camera_prim), "width": size[0], "height": size[1]},
        ),
        *publishers,
    ]
    edges = [
        ("on_playback_tick.outputs:tick", "run_one_frame.inputs:execIn"),
        ("run_one_frame.outputs:step", "render_product.inputs:execIn"),
    ]
    for pub in publishers:
        edges += [
            ("render_product.outputs:execOut", f"{pub.name}.inputs:execIn"),
            ("render_product.outputs:renderProductPath", f"{pub.name}.inputs:renderProductPath"),
            ("ros2_context.outputs:context", f"{pub.name}.inputs:context"),
        ]
    return graph(path, nodes, edges, evidence=ev)


def lidar3d_graph(p: RobotProfile) -> Graph:
    """RTX LiDAR -> PointCloud2 on /point_cloud. RTX LiDAR is a render product, so it mirrors the camera graph."""
    return _render_product_graph(
        f"{p.robot_root}/ros_lidar_3d",
        p.mount(LIDAR_FRAME) + "/rtx_3d",
        (1, 1),  # verify: width/height are ignored for RTX LiDAR render products
        [
            Node(
                "rtx_lidar_helper",
                "ROS2RtxLidarHelper",
                {"topicName": POINT_CLOUD_TOPIC, "frameId": LIDAR_FRAME, "type": "point_cloud"},
            )
        ],
        "tutorial #10.4 (Run One Simulation Frame -> Create Render Product -> ROS2 RTX Lidar Helper, PointCloud2)",
    )


def camera_graph(p: RobotProfile) -> Graph:
    """Camera -> /rgb and /camera_info, the graph the tutorial #10.1 Camera wizard generates."""
    return _render_product_graph(
        f"{p.robot_root}/ros_camera",
        p.mount(CAMERA_FRAME),
        (640, 480),  # our choice; tutorial #10.1 only says resolution is set on Create Render Product
        [
            Node("camera_rgb", "ROS2CameraHelper", {"topicName": RGB_TOPIC, "frameId": CAMERA_FRAME, "type": "rgb"}),
            Node("camera_info", "ROS2CameraInfoHelper", {"topicName": CAMERA_INFO_TOPIC, "frameId": CAMERA_FRAME}),
        ],
        "tutorial #10.1 (Camera wizard: rgb + camera_info)",
    )


def imu_graph(p: RobotProfile) -> Graph:
    """IMU -> sensor_msgs/Imu on /imu, stamped with simulation time (tutorial #10.2: do not wire sensorTime)."""
    return graph(
        f"{p.robot_root}/ros_imu",
        [
            *_tick_ctx_time(),
            Node("read_imu", "IsaacReadIMU", {"imuPrim": PrimRef.to(p.mount(IMU_FRAME) + "/imu_sensor")}),
            Node("qos", "ROS2QoSProfile", {"reliability": "reliable"}),  # optional in the tutorial
            Node("publish_imu", "ROS2PublishImu", {"topicName": IMU_TOPIC, "frameId": IMU_FRAME}),
        ],
        [
            ("on_playback_tick.outputs:tick", "read_imu.inputs:execIn"),
            ("read_imu.outputs:execOut", "publish_imu.inputs:execIn"),
            ("ros2_context.outputs:context", "publish_imu.inputs:context"),
            ("qos.outputs:qosProfile", "publish_imu.inputs:qosProfile"),
            ("sim_time.outputs:simulationTime", "publish_imu.inputs:timeStamp"),
            ("read_imu.outputs:angVel", "publish_imu.inputs:angularVelocity"),
            ("read_imu.outputs:linAcc", "publish_imu.inputs:linearAcceleration"),
            ("read_imu.outputs:orientation", "publish_imu.inputs:orientation"),
        ],
        evidence="tutorial #10.2 (Isaac Read IMU -> ROS2 Publish IMU, sim time, optional QoS)",
    )


def all_graphs(p: RobotProfile) -> list[Graph]:
    return [
        differential_drive_graph(p),
        clock_graph(p),
        odometry_graph(p),
        tf_graph(p),
        lidar2d_graph(p),
        lidar3d_graph(p),
        camera_graph(p),
        imu_graph(p),
    ]


def published_topics(p: RobotProfile) -> set[str]:
    """Every topic the graphs publish (what Nav2 can subscribe to)."""
    out: set[str] = set()
    for g in all_graphs(p):
        for n in g.nodes:
            if n.type.startswith("ROS2Publish") or n.type.endswith("Helper"):
                out.add("/" + n.values["topicName"])
    return out


def tf_edges(p: RobotProfile) -> dict[str, str]:
    """child frame -> parent frame, as published by the odometry and TF graphs (AMCL adds map -> odom)."""
    parents: dict[str, str] = {}

    def add(child: str, parent: str) -> None:
        if child in parents and parents[child] != parent:
            raise ValueError(f"frame {child} has two parents: {parents[child]} and {parent}")
        parents[child] = parent

    for g in (odometry_graph(p), tf_graph(p)):
        for n in g.nodes:
            if n.type == "ROS2PublishRawTransformTree":
                add(n.values["childFrameId"], n.values["parentFrameId"])
            elif n.type == "ROS2PublishTransformTree":
                parent = _leaf(n.values["parentPrim"].paths[0])
                for target in n.values["targetPrims"].paths:
                    add(_leaf(target), parent)
    return parents
