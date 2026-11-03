#!/usr/bin/env python3
"""Turn a multi-camera drive capture into a Gaussian-splatting training layout, with sync, pose, and coverage checks.

This is the generic preparation any Gaussian reconstruction needs (posed, time-aligned, masked images). It is NOT the
NuRec input format: NuRec ingests NCore (the Mcity team converted ROS bags to it with NVIDIA's example conversion
scripts; consult the NuRec docs). The outputs here are the two open layouts that gsplat and nerfstudio read:

* ``transforms.json``  - nerfstudio style: per-frame camera-to-world (OpenGL axes), per-frame OPENCV intrinsics,
                          optional ``mask_path`` (an ego mask hides the hood, as in the Mcity pipeline).
* ``sparse/0/*.txt``   - COLMAP text model: ``cameras.txt`` (OPENCV), ``images.txt`` (world-to-camera, QW QX QY QZ),
                          ``points3D.txt`` (from ``points.npy`` if given, else empty).

Input layout (this repo's convention; adapt the loader to your recorder)::

    capture/
      frames/<camera>/<timestamp_ns>.png|jpg
      poses.csv          timestamp_ns,x,y,z,qx,qy,qz,qw   world-from-rig, Z-up, quaternion XYZW
      intrinsics.json    {"<camera>": {"width", "height", "fx", "fy", "cx", "cy", "dist": [k1, k2, p1, p2],
                                       "T_rig_cam": 4x4 rig-from-camera, OpenCV camera axes (x right, y down, z fwd)}}
      masks/<camera>.png optional static ego mask (white = keep)
      points.npy         optional (N, 3) or (N, 6) xyz[rgb] points (e.g. LiDAR) for Gaussian initialization

Checks (all CPU): cross-camera time skew per trigger group (Mcity used PTP sync and camera trigger mode), dropped
frames, frames outside the pose time range, pose jumps (speed, acceleration, yaw rate), quaternion norms, and
trajectory coverage (path length, frames per metre, heading span, lateral spread of camera centres).

Usage::

    python prepare_capture.py --capture capture/ --out prepared/        # writes layout + capture_report.json
    python prepare_capture.py --synthetic demo_capture/                 # make a small synthetic capture first
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMAGE_EXT = (".png", ".jpg", ".jpeg")
GL_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # flip camera y and z: OpenCV -> OpenGL/nerfstudio camera axes


# ---------------------------------------------------------------------------
# Rotations (quaternions XYZW, as in the rest of this repo)
# ---------------------------------------------------------------------------


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion XYZW with w >= 0."""
    m = np.asarray(R, float)
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:  # take the short way (q and -q are the same rotation)
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def yaw_of(q: Sequence[float]) -> float:
    R = quat_to_matrix(q)
    return math.atan2(R[1, 0], R[0, 0])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class Capture:
    root: Path
    frames: dict[str, list[tuple[int, Path]]]  # camera -> sorted (timestamp_ns, path)
    pose_t: np.ndarray  # (N,) int64 ns
    pose_p: np.ndarray  # (N, 3)
    pose_q: np.ndarray  # (N, 4) XYZW
    intrinsics: dict[str, dict]
    masks: dict[str, Path] = field(default_factory=dict)
    points: np.ndarray | None = None


def load_capture(root: Path) -> Capture:
    root = Path(root)
    frames: dict[str, list[tuple[int, Path]]] = {}
    for cam_dir in sorted(p for p in (root / "frames").iterdir() if p.is_dir()):
        items = []
        for f in cam_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXT:
                try:
                    items.append((int(f.stem), f))
                except ValueError:
                    continue
        frames[cam_dir.name] = sorted(items)
    with (root / "poses.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    pose_t = np.array([int(r["timestamp_ns"]) for r in rows], dtype=np.int64)
    pose_p = np.array([[float(r[k]) for k in "xyz"] for r in rows])
    pose_q = np.array([[float(r[k]) for k in ("qx", "qy", "qz", "qw")] for r in rows])
    intr = json.loads((root / "intrinsics.json").read_text())
    masks = {p.stem: p for p in (root / "masks").glob("*.png")} if (root / "masks").is_dir() else {}
    pts = np.load(root / "points.npy") if (root / "points.npy").is_file() else None
    return Capture(root, frames, pose_t, pose_p, pose_q, intr, masks, pts)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@dataclass
class Limits:
    max_skew_ms: float = 1.0  # cross-camera skew inside one trigger group
    max_speed: float = 70.0  # m/s
    max_accel: float = 15.0  # m/s^2
    max_yaw_rate: float = 2.0  # rad/s


def check_sync(cap: Capture, max_skew_ms: float) -> dict:
    """Group frames by the reference camera's timestamps; report per-camera offsets and per-group skew."""
    cams = list(cap.frames)
    ref = cams[0]
    ref_t = np.array([t for t, _ in cap.frames[ref]], dtype=np.int64)
    period = float(np.median(np.diff(ref_t))) if len(ref_t) > 1 else float("inf")
    offsets: dict[str, list[float]] = {c: [] for c in cams[1:]}
    unmatched = {c: 0 for c in cams[1:]}
    skews = []
    for t in ref_t:
        group = [0.0]
        for c in cams[1:]:
            ts = np.array([x for x, _ in cap.frames[c]], dtype=np.int64)
            if ts.size == 0:
                unmatched[c] += 1
                continue
            j = int(np.argmin(np.abs(ts - t)))
            off = float(ts[j] - t)
            if abs(off) > period / 2:
                unmatched[c] += 1
                continue
            offsets[c].append(off / 1e6)
            group.append(off / 1e6)
        skews.append(max(group) - min(group))
    skews = np.array(skews)
    return {
        "reference_camera": ref,
        "frame_period_ms": period / 1e6,
        "groups": len(ref_t),
        "max_group_skew_ms": float(skews.max()) if skews.size else 0.0,
        "groups_over_limit": int((skews > max_skew_ms).sum()),
        "limit_ms": max_skew_ms,
        "per_camera_offset_ms": {
            c: {"median": float(np.median(v)), "max_abs": float(np.max(np.abs(v)))} if v else None
            for c, v in offsets.items()
        },
        "unmatched_frames": unmatched,
    }


def check_frame_intervals(cap: Capture) -> dict:
    out = {}
    for c, items in cap.frames.items():
        t = np.array([x for x, _ in items], dtype=np.int64)
        if t.size < 2:
            out[c] = {"frames": int(t.size)}
            continue
        dt = np.diff(t)
        med = float(np.median(dt))
        gaps = dt[dt > 1.5 * med]
        out[c] = {
            "frames": int(t.size),
            "median_dt_ms": med / 1e6,
            "fps": 1e9 / med if med else float("inf"),
            "dropped_frames_est": int(np.sum(np.round(gaps / med) - 1)),
            "duplicate_timestamps": int(np.sum(dt == 0)),
        }
    return out


def check_poses(cap: Capture, lim: Limits) -> dict:
    t = cap.pose_t.astype(float) / 1e9
    dt = np.diff(t)
    issues = []
    if np.any(dt <= 0):
        issues.append(f"non-increasing pose timestamps at {np.flatnonzero(dt <= 0).tolist()[:10]}")
    dt = np.where(dt > 0, dt, np.nan)
    vel = np.diff(cap.pose_p, axis=0) / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)
    acc = np.linalg.norm(np.diff(vel, axis=0), axis=1) / dt[1:]
    yaw = np.unwrap([yaw_of(q) for q in cap.pose_q])
    yaw_rate = np.abs(np.diff(yaw)) / dt
    qn = np.linalg.norm(cap.pose_q, axis=1)
    jumps = {
        "speed": np.flatnonzero(speed > lim.max_speed).tolist(),
        "accel": np.flatnonzero(acc > lim.max_accel).tolist(),
        "yaw_rate": np.flatnonzero(yaw_rate > lim.max_yaw_rate).tolist(),
    }
    return {
        "samples": len(t),
        "rate_hz": float(1.0 / np.nanmedian(dt)) if len(t) > 1 else 0.0,
        "t_start_s": float(t[0]),
        "t_end_s": float(t[-1]),
        "max_speed_mps": float(np.nanmax(speed)) if speed.size else 0.0,
        "max_accel_mps2": float(np.nanmax(acc)) if acc.size else 0.0,
        "max_yaw_rate_rps": float(np.nanmax(yaw_rate)) if yaw_rate.size else 0.0,
        "quat_norm_max_error": float(np.max(np.abs(qn - 1.0))),
        "jump_indices": {k: v[:20] for k, v in jumps.items()},
        "jumps": sum(len(v) for v in jumps.values()),
        "issues": issues,
    }


def interpolate_pose(cap: Capture, t_ns: int) -> np.ndarray | None:
    """World-from-rig 4x4 at ``t_ns`` (linear position, slerp rotation); None outside the pose range."""
    t = cap.pose_t
    if t_ns < t[0] or t_ns > t[-1]:
        return None
    j = int(np.searchsorted(t, t_ns, side="right"))
    j = min(max(j, 1), len(t) - 1)
    a = (t_ns - t[j - 1]) / max(t[j] - t[j - 1], 1)
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(slerp(cap.pose_q[j - 1], cap.pose_q[j], a))
    T[:3, 3] = (1 - a) * cap.pose_p[j - 1] + a * cap.pose_p[j]
    return T


def camera_poses(cap: Capture) -> tuple[list[dict], dict[str, int]]:
    """Camera-to-world (OpenCV axes) for every frame with a pose; count frames outside the pose range."""
    out, outside = [], dict.fromkeys(cap.frames, 0)
    for cam, items in cap.frames.items():
        T_rc = np.asarray(cap.intrinsics[cam]["T_rig_cam"], float)
        for ts, path in items:
            T_wr = interpolate_pose(cap, ts)
            if T_wr is None:
                outside[cam] += 1
                continue
            out.append({"camera": cam, "timestamp_ns": ts, "path": path, "c2w": T_wr @ T_rc})
    return out, outside


def coverage(cap: Capture, frames: list[dict]) -> dict:
    if not frames:
        return {}
    t0 = min(f["timestamp_ns"] for f in frames)
    t1 = max(f["timestamp_ns"] for f in frames)
    sel = (cap.pose_t >= t0) & (cap.pose_t <= t1)
    p = cap.pose_p[sel]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    length = float(seg.sum())
    yaw = np.unwrap([yaw_of(q) for q in cap.pose_q[sel]])
    centres = np.array([f["c2w"][:3, 3] for f in frames])
    # lateral spread of camera centres relative to the rig path (how far off the driven line any view was taken)
    lat = []
    for f in frames:
        T = interpolate_pose(cap, f["timestamp_ns"])
        left = T[:3, :3] @ np.array([0.0, 1.0, 0.0])
        lat.append(float(np.dot(f["c2w"][:3, 3] - T[:3, 3], left)))
    per_cam_frames = {c: sum(1 for f in frames if f["camera"] == c) for c in cap.frames}
    return {
        "duration_s": (t1 - t0) / 1e9,
        "path_length_m": length,
        "mean_speed_mps": length / max((t1 - t0) / 1e9, 1e-9),
        "frames_used": len(frames),
        "frames_per_metre": len(frames) / length if length > 0 else float("inf"),
        "metres_between_frames_per_camera": {
            c: length / max(n - 1, 1) if n else None for c, n in per_cam_frames.items()
        },
        "heading_span_deg": float(np.degrees(yaw.max() - yaw.min())) if yaw.size else 0.0,
        "bbox_m": {"min": centres.min(axis=0).round(3).tolist(), "max": centres.max(axis=0).round(3).tolist()},
        "camera_lateral_spread_m": float(max(lat) - min(lat)),
        "note": "Every view lies within camera_lateral_spread_m of the driven line; novel views farther out are "
        "extrapolation (see novel_view_envelope.md).",
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.relpath(src.resolve(), dst.parent.resolve()), dst)


def write_layout(cap: Capture, frames: list[dict], out: Path, mode: str = "symlink") -> dict:
    """Write images/, masks/, transforms.json, and sparse/0/*.txt under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    cams = list(cap.frames)
    cam_id = {c: i + 1 for i, c in enumerate(cams)}
    for c, m in cap.masks.items():
        _place(m, out / "masks" / f"{c}.png", "copy")
    tf_frames = []
    images_txt = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME", "# (second line per image: 2D points, empty)"]
    for i, f in enumerate(sorted(frames, key=lambda x: (x["timestamp_ns"], x["camera"]))):
        name = f"{f['camera']}/{f['timestamp_ns']}{f['path'].suffix.lower()}"
        _place(f["path"], out / "images" / name, mode)
        k = cap.intrinsics[f["camera"]]
        dist = list(k.get("dist", [0, 0, 0, 0])) + [0.0] * 4
        entry = {
            "file_path": f"images/{name}",
            "transform_matrix": (f["c2w"] @ GL_FROM_CV).round(9).tolist(),
            "fl_x": k["fx"],
            "fl_y": k["fy"],
            "cx": k["cx"],
            "cy": k["cy"],
            "w": k["width"],
            "h": k["height"],
            "k1": dist[0],
            "k2": dist[1],
            "p1": dist[2],
            "p2": dist[3],
            "camera": f["camera"],
            "timestamp_ns": f["timestamp_ns"],
        }
        if f["camera"] in cap.masks:
            entry["mask_path"] = f"masks/{f['camera']}.png"
        tf_frames.append(entry)
        w2c = np.linalg.inv(f["c2w"])
        qx, qy, qz, qw = matrix_to_quat(w2c[:3, :3])
        tx, ty, tz = w2c[:3, 3]
        images_txt.append(
            f"{i + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.6f} {ty:.6f} {tz:.6f} {cam_id[f['camera']]} {name}"
        )
        images_txt.append("")
    (out / "transforms.json").write_text(
        json.dumps(
            {"camera_model": "OPENCV", "orientation_note": "c2w, OpenGL camera axes", "frames": tf_frames}, indent=1
        )
        + "\n"
    )
    sparse = out / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    cam_lines = ["# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[] (OPENCV: fx fy cx cy k1 k2 p1 p2)"]
    for c in cams:
        k = cap.intrinsics[c]
        d = list(k.get("dist", [0, 0, 0, 0])) + [0.0] * 4
        cam_lines.append(
            f"{cam_id[c]} OPENCV {k['width']} {k['height']} "
            f"{k['fx']} {k['fy']} {k['cx']} {k['cy']} {d[0]} {d[1]} {d[2]} {d[3]}"
        )
    (sparse / "cameras.txt").write_text("\n".join(cam_lines) + "\n")
    (sparse / "images.txt").write_text("\n".join(images_txt) + "\n")
    pts_lines = ["# POINT3D_ID X Y Z R G B ERROR TRACK[]"]
    if cap.points is not None:
        P = np.asarray(cap.points, float)
        rgb = P[:, 3:6].astype(int) if P.shape[1] >= 6 else np.full((len(P), 3), 128)
        for i, (p, c) in enumerate(zip(P[:, :3], rgb, strict=True)):
            pts_lines.append(f"{i + 1} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]} 0")
    (sparse / "points3D.txt").write_text("\n".join(pts_lines) + "\n")
    return {"images": len(tf_frames), "cameras": len(cams), "points3D": 0 if cap.points is None else len(cap.points)}


def prepare(capture: Path, out: Path, lim: Limits | None = None, mode: str = "symlink") -> dict:
    lim = lim or Limits()
    cap = load_capture(capture)
    frames, outside = camera_poses(cap)
    report = {
        "capture": str(capture),
        "format_note": "generic 3DGS layout (nerfstudio transforms.json + COLMAP text); not the NuRec/NCore format",
        "sync": check_sync(cap, lim.max_skew_ms),
        "frame_intervals": check_frame_intervals(cap),
        "poses": check_poses(cap, lim),
        "frames_outside_pose_range": outside,
        "coverage": coverage(cap, frames),
    }
    errors, warnings = [], []
    if report["sync"]["groups_over_limit"]:
        errors.append(
            f"{report['sync']['groups_over_limit']} trigger groups exceed {lim.max_skew_ms} ms skew: check PTP sync "
            "and camera trigger mode"
        )
    if report["poses"]["jumps"]:
        errors.append(f"{report['poses']['jumps']} pose jumps (speed/accel/yaw-rate limits): fix the pose source")
    errors += report["poses"]["issues"]
    if any(outside.values()):
        warnings.append(f"frames outside the pose time range were dropped: {outside}")
    drops = {c: v.get("dropped_frames_est", 0) for c, v in report["frame_intervals"].items()}
    if any(drops.values()):
        warnings.append(f"estimated dropped frames per camera: {drops}")
    if cap.points is None:
        warnings.append("no points.npy: Gaussians need an initialization (LiDAR points, triangulation, or random)")
    report["errors"], report["warnings"], report["ok"] = errors, warnings, not errors
    report["written"] = write_layout(cap, frames, out, mode)
    (out / "capture_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


# ---------------------------------------------------------------------------
# Synthetic capture (for tests and demos)
# ---------------------------------------------------------------------------

_CAMS = {  # rig-from-camera rotation columns: camera x, y, z axes in the rig frame (rig: x fwd, y left, z up)
    "front": ((0, -1, 0), (0, 0, -1), (1, 0, 0), (1.0, 0.0, 1.6)),
    "left": ((1, 0, 0), (0, 0, -1), (0, 1, 0), (0.0, 0.8, 1.6)),
    "right": ((-1, 0, 0), (0, 0, -1), (0, -1, 0), (0.0, -0.8, 1.6)),
}


def make_synthetic_capture(
    root: Path,
    *,
    duration_s: float = 4.0,
    fps: float = 10.0,
    pose_hz: float = 100.0,
    speed: float = 8.0,
    skew_ms: dict[str, float] | None = None,
    drop: dict[str, Sequence[int]] | None = None,
    pose_jump_at: int | None = None,
    with_mask: bool = True,
    with_points: bool = True,
) -> Path:
    """A three-camera S-curve drive with tiny PNG frames. Faults can be injected for the checks."""
    from PIL import Image

    root = Path(root)
    skew_ms, drop = skew_ms or {}, drop or {}
    t0 = 1_700_000_000_000_000_000

    def rig(t: float) -> tuple[np.ndarray, float]:
        x, y = speed * t, 2.0 * math.sin(0.5 * t)
        yaw = math.atan2(1.0 * math.cos(0.5 * t), speed)
        return np.array([x, y, 0.0]), yaw

    n_pose = int(duration_s * pose_hz) + 1
    root.mkdir(parents=True, exist_ok=True)
    with (root / "poses.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for i in range(n_pose):
            t = i / pose_hz
            p, yaw = rig(t)
            if pose_jump_at is not None and i >= pose_jump_at:
                p = p + np.array([0.0, 5.0, 0.0])  # a GNSS jump
            w.writerow(
                [t0 + round(t * 1e9), *p.round(6), 0.0, 0.0, round(math.sin(yaw / 2), 9), round(math.cos(yaw / 2), 9)]
            )
    intr = {}
    for cam, (cx_, cy_, cz_, trans) in _CAMS.items():
        T = np.eye(4)
        T[:3, :3] = np.array([cx_, cy_, cz_], float).T
        T[:3, 3] = trans
        intr[cam] = {"width": 32, "height": 24, "fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0,
                     "dist": [0.0, 0.0, 0.0, 0.0], "T_rig_cam": T.tolist()}  # fmt: skip
        d = root / "frames" / cam
        d.mkdir(parents=True, exist_ok=True)
        for k in range(int(duration_s * fps)):
            if k in set(drop.get(cam, ())):
                continue
            ts = t0 + round((k / fps + 0.05) * 1e9 + skew_ms.get(cam, 0.0) * 1e6)
            img = np.zeros((24, 32, 3), np.uint8)
            img[..., 0] = (k * 7) % 255
            img[..., 1] = np.linspace(0, 255, 32, dtype=np.uint8)[None, :]
            Image.fromarray(img).save(d / f"{ts}.png")
    (root / "intrinsics.json").write_text(json.dumps(intr, indent=1) + "\n")
    if with_mask:
        (root / "masks").mkdir(exist_ok=True)
        m = np.full((24, 32), 255, np.uint8)
        m[20:, :] = 0  # the hood
        Image.fromarray(m).save(root / "masks" / "front.png")
    if with_points:
        r = np.random.default_rng(0)
        np.save(
            root / "points.npy", np.column_stack([r.uniform(0, 40, 200), r.uniform(-10, 10, 200), r.uniform(0, 5, 200)])
        )
    return root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def summarize(rep: dict) -> str:
    s, c, p = rep["sync"], rep.get("coverage", {}), rep["poses"]
    lines = [
        f"ok: {rep['ok']}",
        f"sync: max group skew {s['max_group_skew_ms']:.3f} ms "
        f"({s['groups_over_limit']} groups over {s['limit_ms']} ms)",
        f"poses: {p['samples']} samples at {p['rate_hz']:.1f} Hz, {p['jumps']} jumps, "
        f"max speed {p['max_speed_mps']:.1f} m/s",
    ]
    if c:
        lines.append(
            f"coverage: {c['path_length_m']:.1f} m in {c['duration_s']:.1f} s, {c['frames_used']} frames, "
            f"{c['frames_per_metre']:.2f} frames/m, heading span {c['heading_span_deg']:.1f} deg, "
            f"camera lateral spread {c['camera_lateral_spread_m']:.2f} m"
        )
    lines += [f"ERROR: {e}" for e in rep["errors"]] + [f"warning: {w}" for w in rep["warnings"]]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--synthetic", type=Path, help="write a synthetic capture to this folder and exit")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    ap.add_argument("--max-skew-ms", type=float, default=1.0)
    args = ap.parse_args(argv)
    if args.synthetic:
        make_synthetic_capture(args.synthetic)
        print(f"[prepare_capture] synthetic capture in {args.synthetic}")
        return 0
    if not (args.capture and args.out):
        ap.error("--capture and --out are required")
    rep = prepare(args.capture, args.out, Limits(max_skew_ms=args.max_skew_ms), "copy" if args.copy else "symlink")
    print(summarize(rep))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
