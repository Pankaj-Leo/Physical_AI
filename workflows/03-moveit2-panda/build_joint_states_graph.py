"""Isaac Sim 5.0.0 standalone script: Franka in the Simple Room with the ROS_JointStates graph (tutorial #13 Part C).

Not executed: needs GPU (Isaac Sim 5.0.0 on Linux with an RTX GPU). Static checks (compile, ruff) pass on CPU.

    ./python.sh build_joint_states_graph.py --play
Then, in a ROS 2 Humble shell: ros2 launch panda_moveit_config demo.launch.py (see README).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import moveit_graph_spec as spec
from ogspec import apply

# verify: asset paths under the Isaac assets root for Isaac Sim 5.0.0.
SIMPLE_ROOM = "/Isaac/Environments/Simple_Room/simple_room.usd"
FRANKA = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--save", type=Path)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.ros2.bridge")
        app.update()

        import omni.timeline
        import omni.usd
        from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading, open_stage
        from isaacsim.storage.native import get_assets_root_path  # verify: module path in 5.0.0

        root = get_assets_root_path()
        open_stage(root + SIMPLE_ROOM)
        # Tutorial #13 Part C: the URDF importer reports Panda "already in Isaac assets"; use the asset instead.
        add_reference_to_stage(root + FRANKA, spec.FRANKA_PRIM)
        while is_stage_loading():
            app.update()

        apply(spec.joint_states_graph())
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
