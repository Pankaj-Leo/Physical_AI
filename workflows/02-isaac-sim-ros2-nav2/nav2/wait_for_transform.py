"""Block until a TF transform exists (default map -> odom), then exit 0; exit 1 on timeout.

Used by bringup.launch.py between the localization and navigation phases (wizard step 6). Runs on simulation time,
so it also fails if Isaac Sim is not publishing /clock. Not executed here: needs ROS 2 Jazzy (rclpy, tf2_ros).
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default="map")
    ap.add_argument("--child", default="odom")
    ap.add_argument("--timeout", type=float, default=120.0, help="wall-clock seconds")
    args = ap.parse_args(argv)

    import time

    import rclpy
    from rclpy.parameter import Parameter
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = rclpy.create_node(
        "wait_for_transform", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)]
    )
    buffer = Buffer()
    TransformListener(buffer, node)
    deadline = time.monotonic() + args.timeout
    ok = False
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        if buffer.can_transform(args.parent, args.child, Time()):
            ok = True
            break
    node.get_logger().info(f"{args.parent} -> {args.child}: {'available' if ok else 'timed out'}")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
