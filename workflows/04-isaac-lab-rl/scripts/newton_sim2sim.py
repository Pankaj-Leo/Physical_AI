"""Sim-to-sim: run an ANYmal C velocity policy exported from Isaac Lab (PhysX) in Newton (MuJoCo solver), on CPU.

Reuses Newton's ``example_robot_anymal_c_walk`` scene unchanged (URDF, PD gains 150/5, 50 Hz policy, 4 substeps,
``SolverMuJoCo``) and swaps in the ONNX file you pass. With no ``--policy`` it runs the PhysX-trained policy that ships
with the Newton asset, which is the baseline every exported policy should be compared to.

    python scripts/newton_sim2sim.py --frames 150                                  # shipped policy (executed here)
    python scripts/newton_sim2sim.py --policy artifacts/anymal_c_flat_policy.onnx  # your export (needs a GPU-trained
                                                                                   # policy, so not executed here)

Reports forward distance along the commanded heading, lateral drift, minimum base height and a fall flag; the same
metrics as ``ssw.terrain`` in ``physical-ai-project/project``. Newton 1.7.0.dev0 from the repo's ``newton-main``
snapshot, Warp 1.17, warp_nn; Python 3.11. The robot spawns yawed +90 deg, so body +x (forward) is world +y.
"""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

import numpy as np


def run(policy_path: str | None, frames: int, command: tuple[float, float, float], device: str) -> dict:
    import warp as wp

    wp.set_device(device)
    import newton.viewer
    from newton.examples.robot import example_robot_anymal_c_walk as ex_mod
    from newton.examples.robot.onnx_policy_utils import validate_policy_io_shapes
    from warp_nn.runtime import OnnxRuntime

    ex = ex_mod.Example(newton.viewer.ViewerNull(num_frames=frames), SimpleNamespace(use_mujoco_contacts=False))
    if policy_path is not None:
        ex.policy = OnnxRuntime(policy_path, device=ex.device)
        ex._policy_input_name = ex.policy.input_names[0]
        ex._policy_output_name = ex.policy.output_names[0]
        validate_policy_io_shapes(
            policy_path, ex._policy_input_name, ex._policy_output_name, obs_width=48, action_width=12, context="sim2sim"
        )
    ex._command = wp.vec3(*command)
    ex.capture()  # re-record the step graph so it uses this policy and command

    q_start = ex.state_0.joint_q.numpy()[:3].copy()
    min_z = float(q_start[2])
    for _ in range(frames):
        ex.step()
        min_z = min(min_z, float(ex.state_0.joint_q.numpy()[2]))
    q_end = ex.state_0.joint_q.numpy()[:3]
    sim_time = frames * ex.frame_dt
    forward = float(q_end[1] - q_start[1])  # body +x == world +y after the +90 deg spawn yaw
    lateral = float(-(q_end[0] - q_start[0]))
    return {
        "policy": policy_path or "shipped anymal_walking_policy_physx.onnx",
        "frames": frames,
        "sim_time_s": round(sim_time, 3),
        "command": list(command),
        "forward_distance_m": round(forward, 4),
        "lateral_drift_m": round(lateral, 4),
        "mean_forward_speed_mps": round(forward / sim_time, 4),
        "min_base_z_m": round(min_z, 4),
        "fell": bool(min_z < 0.3),
        "device": device,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default=None, help="ONNX exported by Isaac Lab play.py; default: the shipped policy")
    ap.add_argument("--frames", type=int, default=150, help="frames at 50 fps")
    ap.add_argument("--command", type=float, nargs=3, default=[1.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the metrics as JSON here")
    args = ap.parse_args(argv)
    metrics = run(args.policy, args.frames, tuple(args.command), args.device)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0 if np.isfinite(metrics["forward_distance_m"]) else 1


if __name__ == "__main__":
    sys.exit(main())
