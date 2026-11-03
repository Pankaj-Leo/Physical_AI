"""OmniGraph action graphs as plain data, plus a thin applier for Isaac Sim.

The data half (``NodeType``, ``Node``, ``Edge``, ``Graph``, ``validate``, ``controller_plan``) is pure Python and runs
on CPU, so tests can check every graph before anyone opens Isaac Sim. The applier (``apply``) imports
``omni.graph.core`` inside the function and is only usable inside a running Isaac Sim 5.x process.

The same file is copied into workflows 02, 03 and 09 so each folder stands alone.

Evidence rules for ``CATALOG``:
- ``short`` is the node name as Isaac Sim shows it. Where a screenshot shows the node, the port names below are copied
  from that screenshot; otherwise they come from the tutorial transcripts (spoken names, normalized to camelCase).
- ``type_id`` is the extension-qualified id ``og.Controller`` needs. **No source in this repo shows a full type id.**
  They follow Isaac Sim 5.x naming (``isaacsim.ros2.bridge.*``, ``isaacsim.core.nodes.*``, ...). Verify each one in the
  Isaac Sim graph editor (hover the node) before relying on it. ``type_id_in_sources`` is therefore False everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PrimRef:
    """A relationship input (``targetPrim``, ``parentPrim``, ...). Set through usdrt paths, not as a plain value."""

    paths: tuple[str, ...]

    @classmethod
    def to(cls, *paths: str) -> PrimRef:
        return cls(tuple(paths))


@dataclass(frozen=True)
class NodeType:
    short: str
    type_id: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    evidence: str
    # Constant nodes expose their value input as a connectable source ("inputs:value" -> other node).
    value_sources: tuple[str, ...] = ()
    type_id_in_sources: bool = False


def _t(short: str, type_id: str, inputs: str, outputs: str, evidence: str, *, value_sources: str = "") -> NodeType:
    return NodeType(
        short=short,
        type_id=type_id,
        inputs=tuple(inputs.split()),
        outputs=tuple(outputs.split()),
        evidence=evidence,
        value_sources=tuple(value_sources.split()),
    )


SHOT_NAV2 = "screenshot nav2-omnigraph-inspector.jpg"
SHOT_FORK = "screenshots forklift-fleet-live.jpg, forklift-fleet-complete.jpg"

# verify: every type_id below. Short names and ports carry the evidence named in the last column.
CATALOG: dict[str, NodeType] = {
    t.short: t
    for t in [
        # --- seen in the live OmniGraph inspector (Nav2 wizard), ports copied from the screenshot ---
        _t("OnPlaybackTick", "omni.graph.action.OnPlaybackTick", "", "tick deltaSeconds frame time", SHOT_NAV2),
        _t("ROS2Context", "isaacsim.ros2.bridge.ROS2Context", "domain_id useDomainIDEnvVar", "context", SHOT_NAV2),
        _t(
            "ROS2SubscribeTwist",
            "isaacsim.ros2.bridge.ROS2SubscribeTwist",
            "context execIn nodeNamespace qosProfile queueSize topicName",
            "angularVelocity execOut linearVelocity",
            SHOT_NAV2,
        ),
        _t("BreakVector3", "omni.graph.nodes.BreakVector3", "tuple", "x y z", SHOT_NAV2),
        _t(
            "ConstructArray",
            "omni.graph.nodes.ConstructArray",
            "arraySize arrayType input0 input1 input2 input3 input4 input5",
            "array",
            SHOT_NAV2 + " (input2-5 are dynamic inputs added for 4- and 6-joint arrays, tutorial #7)",
        ),
        _t("ConstantToken", "omni.graph.nodes.ConstantToken", "value", "", SHOT_NAV2, value_sources="value"),
        _t("ConstantString", "omni.graph.nodes.ConstantString", "value", "", SHOT_NAV2, value_sources="value"),
        _t(
            "DifferentialController",
            "isaacsim.robot.wheeled_robots.DifferentialController",
            "angularVelocity dt execIn linearVelocity maxAcceleration maxAngularAcceleration maxAngularSpeed "
            "maxDeceleration maxLinearSpeed maxWheelSpeed wheelDistance wheelRadius",
            "velocityCommand",
            SHOT_NAV2,
        ),
        _t(
            "IsaacArticulationController",
            "isaacsim.core.nodes.IsaacArticulationController",
            "effortCommand execIn jointIndices jointNames positionCommand robotPath targetPrim velocityCommand",
            "",
            SHOT_NAV2 + "; targetPrim from tutorial #6 and the forklift screenshot",
        ),
        # --- seen in the Forklift Fleet dashboard (labels converted to camelCase; verify spelling) ---
        _t(
            "AckermannController",
            "isaacsim.robot.wheeled_robots.AckermannController",
            "acceleration backWheelRadius dt execIn frontWheelRadius invertSteering maxAcceleration "
            "maxSteeringAngleVelocity maxWheelRotation maxWheelVelocity speed steeringAngle steeringAngleVelocity "
            "trackWidth wheelBase",
            "execOut wheelAngles wheelRotationVelocity",
            SHOT_FORK + "; tutorial #7",
        ),
        # --- named in the tutorial transcripts, ports from the transcripts (verify names) ---
        _t(
            "ROS2SubscribeAckermannDrive",
            "isaacsim.ros2.bridge.ROS2SubscribeAckermannDrive",
            "context execIn nodeNamespace qosProfile queueSize topicName",
            "acceleration execOut speed steeringAngle",
            "tutorial #7",
        ),
        _t(
            "IsaacReadSimulationTime",
            "isaacsim.core.nodes.IsaacReadSimulationTime",
            "resetOnStop",
            "simulationTime",
            "tutorials #10.2, #10.3",
        ),
        _t(
            "ROS2PublishClock",
            "isaacsim.ros2.bridge.ROS2PublishClock",
            "context execIn qosProfile timeStamp topicName",
            "",
            "Nav2 wizard step 4 lists a clock topic; no tutorial builds this node",
        ),
        _t(
            "IsaacComputeOdometry",
            "isaacsim.core.nodes.IsaacComputeOdometry",
            "chassisPrim execIn",
            "angularVelocity execOut linearVelocity orientation position",
            "tutorial #11 (created by the Odometry Publisher wizard)",
        ),
        _t(
            "ROS2PublishOdometry",
            "isaacsim.ros2.bridge.ROS2PublishOdometry",
            "angularVelocity chassisFrameId context execIn linearVelocity odomFrameId orientation position "
            "qosProfile timeStamp topicName",
            "",
            "tutorial #11",
        ),
        _t(
            "ROS2PublishRawTransformTree",
            "isaacsim.ros2.bridge.ROS2PublishRawTransformTree",
            "childFrameId context execIn parentFrameId rotation timeStamp topicName translation",
            "",
            "tutorial #11",
        ),
        _t(
            "ROS2PublishTransformTree",
            "isaacsim.ros2.bridge.ROS2PublishTransformTree",
            "context execIn parentPrim targetPrims timeStamp topicName",
            "",
            "tutorial #11",
        ),
        _t(
            "IsaacReadLidarBeams",
            "isaacsim.sensors.physx.IsaacReadLidarBeams",
            "execIn lidarPrim",
            "azimuthRange depthRange execOut horizontalFov horizontalResolution intensitiesData linearDepthData "
            "numCols numRows rotationRate verticalFov verticalResolution",
            "tutorial #10.3",
        ),
        _t(
            "ROS2PublishLaserScan",
            "isaacsim.ros2.bridge.ROS2PublishLaserScan",
            "azimuthRange context depthRange execIn frameId horizontalFov horizontalResolution intensitiesData "
            "linearDepthData numCols numRows rotationRate timeStamp topicName",
            "",
            "tutorial #10.3",
        ),
        _t(
            "IsaacRunOneSimulationFrame",
            "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame",
            "execIn",
            "step",
            "tutorials #10.1, #10.4",
        ),
        _t(
            "IsaacCreateRenderProduct",
            "isaacsim.core.nodes.IsaacCreateRenderProduct",
            "cameraPrim enabled execIn height width",
            "execOut renderProductPath",
            "tutorials #10.1, #10.4",
        ),
        _t(
            "ROS2RtxLidarHelper",
            "isaacsim.ros2.bridge.ROS2RtxLidarHelper",
            "context execIn frameId renderProductPath topicName type",
            "",
            "tutorial #10.4",
        ),
        _t(
            "ROS2CameraHelper",
            "isaacsim.ros2.bridge.ROS2CameraHelper",
            "context execIn frameId renderProductPath topicName type",
            "",
            "tutorial #10.1",
        ),
        _t(
            "ROS2CameraInfoHelper",
            "isaacsim.ros2.bridge.ROS2CameraInfoHelper",
            "context execIn frameId renderProductPath topicName",
            "",
            "tutorial #10.1",
        ),
        _t(
            "IsaacReadIMU",
            "isaacsim.sensors.physics.IsaacReadIMU",
            "execIn imuPrim readGravity",
            "angVel execOut linAcc orientation sensorTime",
            "tutorial #10.2",
        ),
        _t(
            "ROS2PublishImu",
            "isaacsim.ros2.bridge.ROS2PublishImu",
            "angularVelocity context execIn frameId linearAcceleration orientation qosProfile timeStamp topicName",
            "",
            "tutorial #10.2",
        ),
        _t(
            "ROS2QoSProfile",
            "isaacsim.ros2.bridge.ROS2QoSProfile",
            "createProfile deadline depth durability history leaseDuration lifespan liveliness reliability",
            "qosProfile",
            "tutorial #10.2",
        ),
        _t(
            "ROS2PublishJointState",
            "isaacsim.ros2.bridge.ROS2PublishJointState",
            "context execIn nodeNamespace qosProfile targetPrim timeStamp topicName",
            "",
            "tutorial #13",
        ),
        _t(
            "ROS2SubscribeJointState",
            "isaacsim.ros2.bridge.ROS2SubscribeJointState",
            "context execIn nodeNamespace qosProfile queueSize topicName",
            "effortCommand execOut jointNames positionCommand timeStamp velocityCommand",
            "tutorial #13",
        ),
    ]
}


@dataclass(frozen=True)
class Node:
    name: str
    type: str  # key into CATALOG
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    src: str
    src_attr: str  # "outputs:tick" or, for constant nodes, "inputs:value"
    dst: str
    dst_attr: str  # "inputs:execIn"

    @classmethod
    def parse(cls, src: str, dst: str) -> Edge:
        """``Edge.parse("tick.outputs:tick", "ctl.inputs:execIn")``."""
        s_node, s_attr = src.split(".", 1)
        d_node, d_attr = dst.split(".", 1)
        return cls(s_node, s_attr, d_node, d_attr)


@dataclass(frozen=True)
class Graph:
    path: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    evidence: str

    def node(self, name: str) -> Node:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def edges_into(self, name: str, attr: str | None = None) -> list[Edge]:
        return [e for e in self.edges if e.dst == name and (attr is None or e.dst_attr == f"inputs:{attr}")]


def graph(path: str, nodes: list[Node], edges: list[tuple[str, str]], evidence: str) -> Graph:
    return Graph(path, tuple(nodes), tuple(Edge.parse(s, d) for s, d in edges), evidence)


def validate(g: Graph) -> list[str]:
    """Return a list of problems; empty means the graph is internally consistent."""
    problems: list[str] = []
    names = [n.name for n in g.nodes]
    if len(names) != len(set(names)):
        problems.append(f"{g.path}: duplicate node names")
    by_name = {n.name: n for n in g.nodes}
    for n in g.nodes:
        t = CATALOG.get(n.type)
        if t is None:
            problems.append(f"{g.path}/{n.name}: unknown node type {n.type}")
            continue
        problems += [f"{g.path}/{n.name}: unknown input {k}" for k in n.values if k not in t.inputs]
    seen_inputs: set[tuple[str, str]] = set()
    for e in g.edges:
        src, dst = by_name.get(e.src), by_name.get(e.dst)
        if src is None or dst is None:
            problems.append(f"{g.path}: edge {e} references an undefined node")
            continue
        st, dt = CATALOG[src.type], CATALOG[dst.type]
        kind, _, attr = e.src_attr.partition(":")
        ok_src = (kind == "outputs" and attr in st.outputs) or (kind == "inputs" and attr in st.value_sources)
        if not ok_src:
            problems.append(f"{g.path}: {e.src}.{e.src_attr} is not an output of {src.type}")
        kind, _, attr = e.dst_attr.partition(":")
        if kind != "inputs" or attr not in dt.inputs:
            problems.append(f"{g.path}: {e.dst}.{e.dst_attr} is not an input of {dst.type}")
        key = (e.dst, e.dst_attr)
        if key in seen_inputs:
            problems.append(f"{g.path}: {e.dst}.{e.dst_attr} has two incoming connections")
        seen_inputs.add(key)
        if attr in dst.values:
            problems.append(f"{g.path}: {e.dst}.{e.dst_attr} is both connected and set to a value")
    return problems


def controller_plan(g: Graph) -> dict[str, list]:
    """Plain-data version of the ``og.Controller.edit`` call (CPU-testable; ``apply`` maps it to og keys)."""
    plan: dict[str, list] = {
        "CREATE_NODES": [],
        "CREATE_ATTRIBUTES": [],
        "SET_VALUES": [],
        "SET_PRIMS": [],
        "CONNECT": [],
    }
    for n in g.nodes:
        plan["CREATE_NODES"].append((n.name, CATALOG[n.type].type_id))
        for k, v in n.values.items():
            target = "SET_PRIMS" if isinstance(v, PrimRef) else "SET_VALUES"
            plan[target].append((f"{n.name}.inputs:{k}", v))
        if n.type == "ConstructArray":
            # verify: ConstructArray ships with input0 only; extra inputs are dynamic attributes of the element type.
            element = str(n.values.get("arrayType", "token[]")).removesuffix("[]")
            for i in range(1, int(n.values.get("arraySize", 1))):
                plan["CREATE_ATTRIBUTES"].append((f"{n.name}.inputs:input{i}", element))
    plan["CONNECT"] = [(f"{e.src}.{e.src_attr}", f"{e.dst}.{e.dst_attr}") for e in g.edges]
    return plan


def apply(g: Graph) -> None:
    """Create ``g`` on the open stage. Not executed: needs GPU (Isaac Sim 5.x process)."""
    import omni.graph.core as og
    import usdrt.Sdf

    plan = controller_plan(g)
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": g.path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: plan["CREATE_NODES"],
            keys.CREATE_ATTRIBUTES: plan["CREATE_ATTRIBUTES"],
            keys.SET_VALUES: plan["SET_VALUES"]
            # verify: relationship inputs take a list of usdrt.Sdf.Path in Isaac Sim 4.5+.
            + [(attr, [usdrt.Sdf.Path(p) for p in ref.paths]) for attr, ref in plan["SET_PRIMS"]],
            keys.CONNECT: plan["CONNECT"],
        },
    )
