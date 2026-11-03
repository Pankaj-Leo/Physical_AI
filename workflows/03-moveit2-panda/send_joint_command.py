"""Drive the Isaac Sim Panda without MoveIt: publish one JointState on /isaac_joint_commands.

Tutorial #13: "You can also directly control it using a Python script by publishing joint command messages to the
topic that Isaac Sim is subscribing to." Not executed: needs ROS 2 Humble (rclpy) and Isaac Sim playing.

    python3 send_joint_command.py --pose ready
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import moveit_graph_spec as spec

# "ready" is the Panda ready pose used in MoveIt's panda_moveit_config (not from the tutorial, which made its own
# poses with sliders). Radians, panda_joint1..7.
POSES = {
    "home": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ready": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", choices=sorted(POSES), default="ready")
    args = ap.parse_args(argv)

    import rclpy
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("send_joint_command")
    pub = node.create_publisher(JointState, "/" + spec.JOINT_COMMANDS_TOPIC, 10)
    msg = JointState()
    msg.name = list(spec.ARM_JOINTS)
    msg.position = POSES[args.pose]
    for _ in range(20):  # publish for ~2 s so a late subscriber still gets it
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
