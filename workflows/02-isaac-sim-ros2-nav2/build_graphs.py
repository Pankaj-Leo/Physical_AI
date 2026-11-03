"""Isaac Sim 5.0.0 standalone script: load a robot in a warehouse and build the SIL graphs from ``graph_spec.py``.

Not executed: needs GPU (Isaac Sim 5.0.0 on Linux with an RTX GPU). Static checks (compile, ruff) pass on CPU.

Two modes:
- ``--mode build`` (default): open a warehouse, reference the robot, fix the articulation root, add sensor prims,
  then create every graph in ``graph_spec.all_graphs``. This is the "from zero" scene the wizard author plans.
- ``--mode sample``: open NVIDIA's Nova Carter ROS 2 navigation sample, which ships its own graphs (what the Nav2
  wizard loads in step 1). Nothing is built; use it to compare against the inspector screenshot.

Run inside the Isaac Sim container (see docker-compose.yml) or a workstation install:
    ./python.sh build_graphs.py --robot nova_carter --headless --play
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import graph_spec as gs
from ogspec import apply

# verify: asset paths under the Isaac assets root for Isaac Sim 5.0.0.
# Default scene = tutorial #15's warehouse, which ships a NavMesh, so actor_sdg/people.yaml can add walking humans.
# Its occupancy map is not NVIDIA's carter map: generate one with Tools > Robotics > Occupancy Map (tutorial #5).
WAREHOUSE = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
SAMPLE_SCENE = "/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot", choices=sorted(gs.PROFILES), default="nova_carter")
    ap.add_argument("--mode", choices=["build", "sample"], default="build")
    ap.add_argument("--env", default=WAREHOUSE, help="environment USD under the assets root (build mode)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--play", action="store_true", help="press Play and keep stepping until the app closes")
    ap.add_argument("--save", type=Path, help="save the stage as .usd after building")
    return ap.parse_args(argv)


def ensure_articulation_root(stage, profile: gs.RobotProfile) -> None:
    """Tutorial #6 fix: exactly one articulation root, on the prim the controller targets.

    "No robot prim found" means the root is missing or sits on a prim that does not contain all joints. The URDF
    importer put it on ``base_footprint`` for TurtleBot3; the fix was the top-level robot prim.
    """
    from pxr import Usd, UsdPhysics

    target = stage.GetPrimAtPath(profile.articulation_root)
    if not target.IsValid():
        raise RuntimeError(f"articulation root prim not found: {profile.articulation_root}")
    for prim in Usd.PrimRange(stage.GetPrimAtPath(profile.robot_root)):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and prim.GetPath() != target.GetPath():
            print(f"[build_graphs] removing stray articulation root on {prim.GetPath()}")
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if not target.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(target)


def check_wheel_drives(stage, profile: gs.RobotProfile) -> None:
    """Wheels need velocity drives: stiffness 0, damping > 0 (tutorials #5, #7). Warn, do not change."""
    from pxr import Usd, UsdPhysics

    wanted = {profile.left_joint, profile.right_joint}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(profile.robot_root)):
        if prim.GetName() not in wanted:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        stiffness = drive.GetStiffnessAttr().Get() if drive else None
        damping = drive.GetDampingAttr().Get() if drive else None
        if not drive or stiffness not in (0, 0.0) or not damping:
            print(f"[build_graphs] WARNING {prim.GetPath()}: not a velocity drive (k={stiffness}, d={damping})")


def create_sensors(stage, profile: gs.RobotProfile) -> None:
    """Sensor prims under the chassis, each at the origin of its mount (tutorial #10.4: zero translate AND rotate)."""
    import omni.kit.commands
    from pxr import Gf, UsdGeom

    lidar_mount = UsdGeom.Xform.Define(stage, profile.mount(gs.LIDAR_FRAME))
    lidar_mount.AddTranslateOp().Set(Gf.Vec3d(*profile.lidar_mount_xyz))
    UsdGeom.Xform.Define(stage, profile.mount(gs.IMU_FRAME))
    UsdGeom.Camera.Define(stage, profile.mount(gs.CAMERA_FRAME))

    # verify: command names and arguments for Isaac Sim 5.0.0.
    # 2D: PhysX rotating LiDAR (tutorial #10.3). It is deprecated in newer releases; check yours still ships it.
    omni.kit.commands.execute(
        "RangeSensorCreateLidar",
        path="/physx_2d",
        parent=profile.mount(gs.LIDAR_FRAME),
        min_range=0.4,
        max_range=100.0,
        draw_points=False,
        draw_lines=False,
        horizontal_fov=360.0,
        vertical_fov=30.0,
        horizontal_resolution=0.4,
        vertical_resolution=4.0,
        rotation_rate=20.0,
        high_lod=False,  # one row: a 2D scan
        yaw_offset=0.0,
        enable_semantics=True,  # tutorial #15: lets returns off walking humans carry labels
    )
    # 3D: RTX LiDAR "Example Rotary" (tutorial #10.4 raised its base rate from 10 to 60 for a steadier cloud).
    omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path="/rtx_3d",
        parent=profile.mount(gs.LIDAR_FRAME),
        config="Example_Rotary",
        translation=(0.0, 0.0, 0.0),
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
    )
    omni.kit.commands.execute(
        "IsaacSensorCreateImuSensor",
        path="/imu_sensor",
        parent=profile.mount(gs.IMU_FRAME),
        translation=Gf.Vec3d(0.0, 0.0, 0.0),
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = gs.PROFILES[args.robot]

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.ros2.bridge")  # tutorial #1: the bridge must be on before ROS2 nodes exist
        app.update()

        import omni.timeline
        import omni.usd
        from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading, open_stage
        from isaacsim.storage.native import get_assets_root_path  # verify: module path in 5.0.0

        root = get_assets_root_path()
        if root is None:
            raise RuntimeError("Isaac assets root not found; set it in the Nucleus / local assets settings")

        if args.mode == "sample":
            open_stage(root + SAMPLE_SCENE)
        else:
            open_stage(root + args.env)
            add_reference_to_stage(root + profile.asset, profile.robot_root)
        while is_stage_loading():
            app.update()

        stage = omni.usd.get_context().get_stage()
        if args.mode == "build":
            ensure_articulation_root(stage, profile)
            check_wheel_drives(stage, profile)
            create_sensors(stage, profile)
            for g in gs.all_graphs(profile):
                print(f"[build_graphs] creating {g.path} ({len(g.nodes)} nodes, {len(g.edges)} edges)")
                apply(g)
        if args.save:
            omni.usd.get_context().save_as_stage(str(args.save))

        if args.play:
            omni.timeline.get_timeline_interface().play()
            while app.is_running():
                app.update()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
