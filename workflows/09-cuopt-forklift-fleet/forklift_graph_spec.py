"""Per-forklift Ackermann control graph (Forklift Fleet wizard; Isaac Sim tutorial #7), as pure data.

The dashboard's CONTROL GRAPH (``forklift-fleet-live.jpg``, ``forklift-fleet-complete.jpg``) shows, per forklift:
On Playback Tick -> Ackermann Controller (Dt 0.017 from Delta Seconds, Invert Steering true, Max Wheel Rotation 1.200)
-> two Articulation Controllers: wheel angles -> Position Command (steering, 2 values, e.g. [1.20, 0.87]) and wheel
rotation velocity -> Velocity Command (drive, 4 values). The fleet manager sets speed and steering angle each tick from
the cuOpt route; no ROS node appears in that graph.

``ros_input=True`` adds tutorial #7's ROS 2 path instead: ROS2 Subscribe AckermannDrive on ``ackermann_cmd``.

Runs on CPU (data and the Ackermann geometry). ``main`` applies the graph inside Isaac Sim (not executed).
"""

from __future__ import annotations

import math
import sys

from ogspec import Graph, Node, PrimRef, graph

# Joint names and geometry of NVIDIA's forklift B/C assets are not in the sources. verify all four, per asset.
STEER_JOINTS = ("left_rotator_joint", "right_rotator_joint")  # verify
DRIVE_JOINTS = ("left_front_wheel_joint", "right_front_wheel_joint", "left_back_wheel_joint", "right_back_wheel_joint")
# Placeholders, not from a source. Track is picked so track/wheelbase = 0.456, just above the ratio the screenshot's
# wheel-angle pair [1.20, 0.87] implies (0.455; a lower bound if the inner wheel is clipped at maxWheelRotation).
WHEEL_BASE_M = 1.6  # verify
TRACK_WIDTH_M = 0.73  # verify


def ackermann_graph(forklift_prim: str, *, ros_input: bool = False, wheel_radius_m: float = 0.25) -> Graph:
    nodes = [
        Node("on_playback_tick", "OnPlaybackTick"),
        Node(
            "ackermann_controller",
            "AckermannController",
            {
                "invertSteering": True,  # screenshot + talk: "how NVIDIA uses this controller for forklifts"
                "maxWheelRotation": 1.2,  # screenshot
                "wheelBase": WHEEL_BASE_M,
                "trackWidth": TRACK_WIDTH_M,
                "frontWheelRadius": wheel_radius_m,  # verify
                "backWheelRadius": wheel_radius_m,  # verify
            },
        ),
        Node("steer_names", "ConstructArray", {"arraySize": 2, "arrayType": "token[]"}),
        Node("drive_names", "ConstructArray", {"arraySize": 4, "arrayType": "token[]"}),
        *[Node(f"steer_{i}", "ConstantToken", {"value": j}) for i, j in enumerate(STEER_JOINTS)],
        *[Node(f"drive_{i}", "ConstantToken", {"value": j}) for i, j in enumerate(DRIVE_JOINTS)],
        Node("steering_controller", "IsaacArticulationController", {"targetPrim": PrimRef.to(forklift_prim)}),
        Node("drive_controller", "IsaacArticulationController", {"targetPrim": PrimRef.to(forklift_prim)}),
    ]
    edges = [
        ("on_playback_tick.outputs:tick", "ackermann_controller.inputs:execIn"),
        ("on_playback_tick.outputs:deltaSeconds", "ackermann_controller.inputs:dt"),
        ("ackermann_controller.outputs:execOut", "steering_controller.inputs:execIn"),
        ("ackermann_controller.outputs:execOut", "drive_controller.inputs:execIn"),
        ("ackermann_controller.outputs:wheelAngles", "steering_controller.inputs:positionCommand"),
        ("ackermann_controller.outputs:wheelRotationVelocity", "drive_controller.inputs:velocityCommand"),
        ("steer_names.outputs:array", "steering_controller.inputs:jointNames"),
        ("drive_names.outputs:array", "drive_controller.inputs:jointNames"),
        *[(f"steer_{i}.inputs:value", f"steer_names.inputs:input{i}") for i in range(len(STEER_JOINTS))],
        *[(f"drive_{i}.inputs:value", f"drive_names.inputs:input{i}") for i in range(len(DRIVE_JOINTS))],
    ]
    if ros_input:
        nodes += [
            Node("ros2_context", "ROS2Context", {"domain_id": 0, "useDomainIDEnvVar": True}),
            Node("subscribe_ackermann", "ROS2SubscribeAckermannDrive", {"topicName": "ackermann_cmd"}),
        ]
        edges += [
            ("on_playback_tick.outputs:tick", "subscribe_ackermann.inputs:execIn"),
            ("ros2_context.outputs:context", "subscribe_ackermann.inputs:context"),
            ("subscribe_ackermann.outputs:speed", "ackermann_controller.inputs:speed"),
            ("subscribe_ackermann.outputs:steeringAngle", "ackermann_controller.inputs:steeringAngle"),
            ("subscribe_ackermann.outputs:acceleration", "ackermann_controller.inputs:acceleration"),
        ]
    return graph(
        f"{forklift_prim}/ackermann_drive",
        nodes,
        edges,
        evidence="forklift-fleet-live.jpg, forklift-fleet-complete.jpg; Forklift Fleet talk; tutorial #7",
    )


def ackermann_wheel_angles(steer: float, wheel_base: float, track: float) -> tuple[float, float]:
    """(inner, outer) wheel angles in rad for a bicycle-model steering angle ``steer`` (rad, > 0)."""
    radius = wheel_base / math.tan(steer)
    return math.atan(wheel_base / (radius - track / 2)), math.atan(wheel_base / (radius + track / 2))


def implied_track_ratio(inner: float, outer: float) -> float:
    """track / wheelbase implied by an Ackermann pair: cot(outer) - cot(inner)."""
    return 1 / math.tan(outer) - 1 / math.tan(inner)


def main(argv: list[str] | None = None) -> int:
    """Apply the graph to each forklift prim on the open stage. Not executed: needs GPU (Isaac Sim 5.x)."""
    import argparse

    from ogspec import apply

    ap = argparse.ArgumentParser()
    ap.add_argument("forklifts", nargs="+", help="USD paths of the forklift articulation roots")
    ap.add_argument("--ros", action="store_true", help="tutorial #7 variant: drive from /ackermann_cmd")
    args = ap.parse_args(argv)
    for prim in args.forklifts:
        apply(ackermann_graph(prim, ros_input=args.ros))
    return 0


if __name__ == "__main__":
    sys.exit(main())
