"""Render K parallel worlds with ``SensorTiledCamera``: color, depth and semantic clips plus ground truth.

Each world holds the same quadruped (``quadruped.urdf``, PD-held standing pose, no locomotion policy) dropped
with a different initial roll, so some worlds land upright and some fall over. One tiled-camera call renders
every world per frame. Output per world (the inputs workflow 07, Cosmos Transfer and Reasoner, consumes):

    clips/world_000/{color,depth,depth_mm,seg}/frame_0000.png ... frame_0120.png
    clips/world_000_{color,depth,seg}.mp4      121 frames at 30 fps, H.264 (if ffmpeg is on PATH)
    clips/world_000.json                       ground truth sidecar (camera, per-frame base pose, fall flag)
    clips/clips.jsonl                          manifest in the format ``python -m ssw.judge ask`` reads

``depth`` is planar (forward) depth normalized to 8 bit for video; ``depth_mm`` keeps it lossless as 16-bit
millimetres. ``seg`` colors pixels by semantic class (ground, base, hip, thigh, shank).

Usage::

    python render_clips.py --worlds 16 --width 1280 --height 720 --out clips                  # GPU target (not yet run)
    python render_clips.py --device cpu --worlds 2 --width 128 --height 72 --out out/clips   # CPU smoke test (executed)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

try:
    from ssw.terrain import tilt_deg
except ImportError:  # pragma: no cover

    def tilt_deg(q_xyzw: np.ndarray) -> float:
        x, y, _z, _w = (float(v) for v in q_xyzw)
        return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"wf06_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# Semantic classes: id -> (name, RGB). Background (no hit) is 0.
CLASSES = {
    0: ("background", (0, 0, 0)),
    1: ("ground", (90, 90, 90)),
    2: ("base", (40, 110, 220)),
    3: ("hip", (240, 140, 20)),
    4: ("thigh", (40, 180, 70)),
    5: ("shank", (230, 210, 40)),
}
FALL_HEIGHT_M = 0.25  # same thresholds as ssw.terrain.TerrainConfig
FALL_TILT_DEG = 60.0


@dataclass
class ClipConfig:
    worlds: int = 2
    width: int = 128
    height: int = 72
    frames: int = 121
    fps: int = 30
    substeps: int = 32  # even, so a captured graph needs no state copy; dt = 1/(30*32) s ~ 1.04 ms
    fov_deg: float = 60.0
    max_roll_deg: float = 50.0  # below the 60 deg fall threshold, so no world starts "fallen"
    camera_eye: tuple[float, float, float] = (1.5, -1.5, 0.9)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.3)
    depth_far_m: float = 8.0
    device: str = "auto"  # resolved in __post_init__: cuda:0 if present, else cpu
    graph: bool | None = None
    write_png: bool = True
    write_mp4: bool = True
    rolls_deg: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.device = _sibling("scale_benchmark").pick_device(self.device)

    def roll_for(self, world: int) -> float:
        if self.rolls_deg:
            return self.rolls_deg[world]
        return 0.0 if self.worlds == 1 else self.max_roll_deg * world / (self.worlds - 1)


def gpu_preset() -> ClipConfig:
    """16 worlds at 1280x720, the resolution Cosmos Transfer's 720p mode expects.

    Written for a CUDA GPU; not yet run on one.
    """
    return ClipConfig(worlds=16, width=1280, height=720, device="cuda:0")


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera-to-world transform looking from eye to target (camera -Z forward, +Y up), as in newton's asv bench."""
    import warp as wp

    eye_v = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - eye_v
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    rot = np.stack([right, cam_up, -fwd], axis=1)
    return wp.transformf(wp.vec3f(*eye), wp.quat_from_matrix(wp.mat33f(rot.flatten())))


def shape_classes(model) -> np.ndarray:
    """Semantic class per shape index, from the owning body's label."""
    body = model.shape_body.numpy()
    labels = [str(lb).rsplit("/", 1)[-1] for lb in model.body_label]
    out = np.zeros(model.shape_count, dtype=np.uint8)
    for s, b in enumerate(body):
        if b < 0:
            out[s] = 1
            continue
        name = labels[b]
        out[s] = 2 if name == "base" else 3 if name.endswith("HAA") else 4 if name.endswith("THIGH") else 5
    return out


def write_mp4(frames_dir: Path, out: Path, fps: int) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out),
    ]  # fmt: skip
    return subprocess.run(cmd, check=False).returncode == 0


def render(cfg: ClipConfig, out_dir: Path) -> dict:
    """Simulate and render all worlds; write PNG/MP4/JSON. Returns a timing and ground-truth summary."""
    import newton
    import warp as wp
    from newton.sensors import SensorTiledCamera
    from PIL import Image

    wp.config.log_level = wp.LOG_WARNING
    bench = _sibling("scale_benchmark")
    bcfg = bench.BenchConfig(solver="xpbd", fps=cfg.fps, substeps=cfg.substeps, device=cfg.device, graph=cfg.graph)
    model = bench.build_model(cfg.worlds, bcfg)

    # Per-world initial roll about the body x axis (quaternions are XYZW in Newton/Warp).
    q = model.joint_q.numpy()
    q_per_world = len(q) // cfg.worlds
    for w in range(cfg.worlds):
        r = math.radians(cfg.roll_for(w))
        q[w * q_per_world + 3 : w * q_per_world + 7] = [math.sin(r / 2), 0.0, 0.0, math.cos(r / 2)]
    model.joint_q.assign(q)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    sim = bench.ParallelSim(model, bcfg)

    sensor = SensorTiledCamera(model=model)
    sensor.default_render_config.enable_shadows = True
    sensor.default_render_config.enable_textures = True
    sensor.utils.create_default_light(enable_shadows=True)
    rays = sensor.utils.compute_camera_rays_pinhole(cfg.width, cfg.height, camera_fovs=math.radians(cfg.fov_deg))
    color = sensor.utils.create_color_image_output(cfg.width, cfg.height)
    depth = sensor.utils.create_depth_image_output(cfg.width, cfg.height)  # same dtype for forward depth
    shape_idx = sensor.utils.create_shape_index_image_output(cfg.width, cfg.height)
    cam_tf = look_at(cfg.camera_eye, cfg.camera_target)
    cams = wp.array([[cam_tf] * cfg.worlds], dtype=wp.transformf, device=cfg.device)
    classes = shape_classes(model)
    palette = np.array([CLASSES[i][1] for i in sorted(CLASSES)], dtype=np.uint8)
    # Textured ground gives the color and depth videos visible motion cues (as in the upstream example).
    sensor.utils.assign_checkerboard_material(shape_indices=np.flatnonzero(classes == 1).astype(np.uint32))

    bodies_per_world = model.body_count // cfg.worlds
    poses = np.zeros((cfg.worlds, cfg.frames, 7), dtype=np.float64)
    sim_ms, render_ms = [], []
    dirs = {}
    for w in range(cfg.worlds):
        for kind in ("color", "depth", "depth_mm", "seg"):
            d = out_dir / f"world_{w:03d}" / kind
            if cfg.write_png:
                d.mkdir(parents=True, exist_ok=True)
            dirs[(w, kind)] = d

    if sim.graph_wanted:
        sim.capture()  # warm-up frame + capture; the clip starts after it (t = 2 frames)
    for f in range(cfg.frames):
        if f > 0:
            tic = time.perf_counter()
            sim.step()
            wp.synchronize_device(sim.device)
            sim_ms.append((time.perf_counter() - tic) * 1e3)
        tic = time.perf_counter()
        model.bvh_refit_shapes(sim.state_0)
        sensor.update(
            sim.state_0, cams, rays, color_image=color, forward_depth_image=depth, shape_index_image=shape_idx
        )
        rgba = sensor.utils.to_rgba_from_color(color).numpy()  # (worlds, H, W, 4)
        wp.synchronize_device(sim.device)
        render_ms.append((time.perf_counter() - tic) * 1e3)

        dep = depth.numpy()[:, 0]
        sid = shape_idx.numpy()[:, 0]
        hit = (sid < model.shape_count) & np.isfinite(dep) & (dep > 0)
        cls = np.where(hit, classes[np.minimum(sid, model.shape_count - 1)], 0)
        poses[:, f] = sim.state_0.body_q.numpy()[::bodies_per_world][: cfg.worlds]
        if cfg.write_png:
            dep_vis = np.where(hit, np.clip(1.0 - dep / cfg.depth_far_m, 0.0, 1.0) * 255.0, 0).astype(np.uint8)
            dep_mm = np.where(hit, np.clip(dep * 1000.0, 0, 65535), 0).astype(np.uint16)
            for w in range(cfg.worlds):
                name = f"frame_{f:04d}.png"
                Image.fromarray(rgba[w, ..., :3]).save(dirs[(w, "color")] / name)
                Image.fromarray(dep_vis[w]).save(dirs[(w, "depth")] / name)
                Image.fromarray(dep_mm[w]).save(dirs[(w, "depth_mm")] / name)
                Image.fromarray(palette[cls[w]]).save(dirs[(w, "seg")] / name)

    manifest, worlds_gt = [], []
    for w in range(cfg.worlds):
        z = poses[w, :, 2]
        tilts = np.array([tilt_deg(qq) for qq in poses[w, :, 3:7]])
        fall_mask = (z < FALL_HEIGHT_M) | (tilts > FALL_TILT_DEG)
        fell = bool(fall_mask.any())
        videos = {}
        if cfg.write_png and cfg.write_mp4:
            for kind in ("color", "depth", "seg"):
                mp4 = out_dir / f"world_{w:03d}_{kind}.mp4"
                if write_mp4(dirs[(w, kind)], mp4, cfg.fps):
                    videos[kind] = mp4.name
        gt = {
            "world": w,
            "initial_roll_deg": round(cfg.roll_for(w), 2),
            "fps": cfg.fps,
            "frames": cfg.frames,
            "resolution": [cfg.width, cfg.height],
            "camera": {
                "eye": list(cfg.camera_eye),
                "target": list(cfg.camera_target),
                "fov_deg_vertical": cfg.fov_deg,
                "convention": "camera -Z forward, +Y up; world Z up",
            },
            "videos": videos,
            "fell": fell,
            "fall_frame": int(np.argmax(fall_mask)) if fell else None,
            "fall_rule": f"base z < {FALL_HEIGHT_M} m or tilt > {FALL_TILT_DEG} deg",
            "forward_distance_m": round(float(poses[w, -1, 0] - poses[w, 0, 0]), 4),
            "max_tilt_deg": round(float(tilts.max()), 2),
            "min_base_z_m": round(float(z.min()), 4),
            "semantic_classes": {str(k): {"name": v[0], "rgb": list(v[1])} for k, v in CLASSES.items()},
            "base_pose_xyz_qxyzw": np.round(poses[w], 5).tolist(),
        }
        (out_dir / f"world_{w:03d}.json").write_text(json.dumps(gt) + "\n") if cfg.write_png else None
        worlds_gt.append({k: v for k, v in gt.items() if k != "base_pose_xyz_qxyzw"})
        manifest.append(
            {
                "clip": str((out_dir / videos.get("color", f"world_{w:03d}/color")).resolve()),
                "depth": str((out_dir / videos.get("depth", f"world_{w:03d}/depth")).resolve()),
                "seg": str((out_dir / videos.get("seg", f"world_{w:03d}/seg")).resolve()),
                "material": "rigid",
                "seed": w,
                "fell": fell,
                "forward_distance_m": gt["forward_distance_m"],
            }
        )
    if cfg.write_png:
        (out_dir / "clips.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))

    summary = {
        "config": asdict(cfg),
        "device": str(sim.device),
        "graph": sim.graph is not None,
        "shape_count": model.shape_count,
        "first_render_ms_incl_compile": round(render_ms[0], 1),
        "median_render_ms_per_frame_all_worlds": round(float(np.median(render_ms[1:])), 2),
        "median_sim_ms_per_frame_all_worlds": round(float(np.median(sim_ms)), 2) if sim_ms else None,
        "pixels_per_frame": cfg.worlds * cfg.width * cfg.height,
        "worlds": worlds_gt,
    }
    if cfg.write_png:
        (out_dir / "render_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def contact_sheet(out_dir: Path, cfg: ClipConfig, frames: tuple[int, ...] = (0, 30, 60, 120)) -> Path:
    """One small PNG: rows = worlds, columns = (color, depth, seg) at a few frames."""
    from PIL import Image

    cols = [(k, f) for f in frames if f < cfg.frames for k in ("color", "depth", "seg")]
    sheet = Image.new("RGB", (cfg.width * len(cols), cfg.height * cfg.worlds), (255, 255, 255))
    for w in range(cfg.worlds):
        for c, (kind, f) in enumerate(cols):
            img = Image.open(out_dir / f"world_{w:03d}" / kind / f"frame_{f:04d}.png").convert("RGB")
            sheet.paste(img, (c * cfg.width, w * cfg.height))
    path = out_dir / "contact_sheet.png"
    sheet.save(path, optimize=True)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help="auto (cuda:0 if present, else cpu), cpu, cuda:N")
    ap.add_argument("--worlds", type=int, default=2)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=72)
    ap.add_argument("--frames", type=int, default=121)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "clips")
    args = ap.parse_args(argv)
    cfg = ClipConfig(
        worlds=args.worlds,
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
        device=args.device,
        write_mp4=not args.no_mp4,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    summary = render(cfg, args.out)
    contact_sheet(args.out, cfg)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
