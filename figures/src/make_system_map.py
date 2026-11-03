#!/usr/bin/env python3
# ruff: noqa: RUF001, PLR0917  (multiplication signs and en dashes are display text; node() is a data table)
"""Generate the Physical AI system map: an interactive HTML page with one inline SVG diagram.

Rows are stack layers (data model → platform → physics → robotics frameworks → world models → agents).
Columns are where the work runs (real world, the train computer, the simulate computer, the deploy computer).
Edges are labeled data flows, routed through the gutters between rows and columns, colored by flow.
Clicking a component shows what it does, its connections, the NVIDIA slide it comes from (vetted copies in
assets/screens/, never the confidential Newton deck), and the atlas folder that rebuilds it.

Icons: Lucide (ISC license), inlined as <symbol>s. Font: IBM Plex Sans Condensed (OFL), inlined as woff2.

Usage:
  python physical-ai-project/figures/src/make_system_map.py \
      --lucide <dir with lucide-static/icons> --fonts <dir with fontsource woff2 files>
Writes physical-ai-project/guide/physical-ai-system-map.html
"""

from __future__ import annotations

import argparse
import base64
import html
import itertools
import json
import re
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
OUT = PROJ / "guide" / "physical-ai-system-map.html"

XO = 46  # left strip for layer labels
NODE_H = 58

# ----------------------------------------------------------------------------------------------- geometry
COLS = {
    "real": 110,
    "train": 320,
    "c1": 545,
    "c2": 755,
    "c3": 965,
    "c4": 1175,
    "d1": 1405,
    "d2": 1595,
}
WIDTH = {"real": 160, "train": 172}
DEFAULT_W = 170
ROWS = {"R1": 190, "R2": 325, "R3": 460, "R4": 590, "R5": 715}
LAYERS = [
    ("R0", 52, 100, "Agents & skills"),
    ("R1", 112, 268, "World models & reconstruction"),
    ("R2", 268, 400, "Robotics frameworks"),
    ("R3", 400, 528, "Physics & optimization"),
    ("R4", 528, 654, "Platform & sensors"),
    ("R5", 654, 766, "Data model"),
    ("HW", 766, 820, "Hardware"),
]
SHORT = {
    "R0": "Agents",
    "R1": "World models",
    "R2": "Frameworks",
    "R3": "Physics",
    "R4": "Platform",
    "R5": "Data model",
    "HW": "Hardware",
}
COLUMNS = [
    ("real", 22, 202, "Real world", "sources"),
    ("train", 222, 420, "Train", "DGX · GB200"),
    ("sim", 440, 1282, "Simulate", "OVX · RTX PRO 6000 · DGX Spark"),
    ("deploy", 1302, 1690, "Deploy", "Jetson Thor · DRIVE AGX · edge servers"),
]
VB_W, VB_H = 1690 + XO + 14, 880

# ----------------------------------------------------------------------------------------------- content
# id: (col, row, title, subtitle, icon, column-kind, layer text, description, evidence image, atlas folder)
N = {}


def node(nid, col, row, title, sub, icon, desc, evidence=None, atlas=None, evidence_note=None):
    kind = "real" if col == "real" else "train" if col == "train" else "deploy" if col.startswith("d") else "sim"
    N[nid] = {
        "id": nid,
        "col": col,
        "row": row,
        "title": title,
        "sub": sub,
        "icon": icon,
        "kind": kind,
        "desc": desc,
        "evidence": evidence,
        "atlas": atlas,
        "evidence_note": evidence_note,
    }


# Real world
node(
    "web",
    "real",
    "R1",
    "Web & human video",
    "no actions, no forces",
    "video",
    "Internet and egocentric human video. Endless, but seen through human eyes and without contact forces. Used to "
    "pre-train world models; Cosmos 3 turns egocentric human video into robot data through its action vocabulary.",
)
node(
    "teleop",
    "real",
    "R2",
    "Teleoperation",
    "1 operator-hour ≤ 1 h data",
    "hand",
    "A person drives the robot (for example with a headset) to record demonstrations. Highest fidelity, slowest to "
    "scale: at most one hour of data per operator-hour, often minutes of useful data after resets (as reported).",
)
node(
    "fleet",
    "real",
    "R3",
    "Fleet & robot logs",
    "cameras · LiDAR · poses",
    "radar",
    "Recorded drives and robot runs: synchronized cameras, LiDAR, GPS/IMU and poses. The input to neural "
    "reconstruction, and where field failures come back from.",
    "mcity-open-twin.jpg",
    "08-nurec-av",
)
node(
    "cad",
    "real",
    "R5",
    "CAD · URDF · MJCF",
    "robots and facilities",
    "box",
    "Robot descriptions and facility CAD as they exist in engineering tools. Imported into OpenUSD and restructured "
    "into SimReady assets.",
    None,
    "01-openusd-simready",
)

# Train
node(
    "cosmostrain",
    "train",
    "R1",
    "Cosmos training",
    "pre/post-training · 8× H100+",
    "brain-circuit",
    "Pre-training and post-training of Cosmos world models with the Cosmos Framework. Every published fine-tuning "
    "recipe assumes 8× H100 or more. Produces the Generator, the Reasoner and the distilled Edge model.",
    "cosmos-dreams-recipe.jpg",
    "07-cosmos-world-models",
)
node(
    "vla",
    "train",
    "R2",
    "Policy & VLA training",
    "GR00T N1 · PPO · imitation",
    "bot",
    "Robot foundation models (GR00T N1: a VLM 'System 2' plus a 120 Hz diffusion 'System 1') and task policies "
    "trained on a mix of real demos, simulated trajectories and generated video.",
    None,
    "04-isaac-lab-rl",
)
node(
    "pnemo",
    "train",
    "R3",
    "PhysicsNeMo",
    "AI-physics surrogates",
    "thermometer",
    "Open-source framework for physics-ML: trains surrogate models on trusted high-fidelity simulation (CFD, "
    "thermal) so engineering twins answer in milliseconds.",
    "physicsnemo.jpg",
    "11-industrial-twins",
)
node(
    "tao",
    "train",
    "R4",
    "Perception training",
    "TAO · PyTorch detectors",
    "scan-eye",
    "Fine-tunes detectors and segmenters on synthetic labeled frames, then exports TensorRT engines for the edge. "
    "In an NVIDIA tutorial, a DetectNet_v2 detector reached above 81% mAP on Replicator data (as reported).",
    None,
    "05-replicator-sdg",
)
node(
    "curator",
    "train",
    "R5",
    "NeMo Curator",
    "curate & search video",
    "database",
    "Curates and searches large video collections to find gaps and corner cases before training.",
)

# Simulate: R1 world models & reconstruction
node(
    "nurec",
    "c1",
    "R1",
    "NuRec",
    "3D Gaussian reconstruction",
    "scan-line",
    "Turns sensor logs into 3D Gaussian-splat scenes you can re-drive, rendering cameras and LiDAR from one "
    "representation. Exports USDZ to Isaac Sim, CARLA or AlpaSim. Mcity: about $2,310 and 2 days versus about "
    "$150k and 6 months by hand.",
    "nurec-reference-pipeline.jpg",
    "08-nurec-av",
)
node(
    "gen",
    "c2",
    "R1",
    "Cosmos 3 Generator",
    "transfer · predict · policy",
    "sparkles",
    "Cosmos 3 (May 2026) generation tower: video, audio and action. Transfer turns depth, segmentation or edge "
    "renders into photoreal video with geometry and labels preserved. Forward dynamics, inverse dynamics and policy "
    "modes live in the same checkpoint.",
    "physical-ai-flywheel.jpg",
    "07-cosmos-world-models",
)
node(
    "reasoner",
    "c3",
    "R1",
    "Cosmos 3 Reasoner",
    "caption · ground · judge",
    "eye",
    "Cosmos 3 reasoning tower: video and text in, text out. Captions with timestamps, 2D grounding in 0–1000 "
    "coordinates, plans, and physical-plausibility verdicts. Used to filter generated data and judge rollouts.",
    "physical-ai-flywheel.jpg",
    "07-cosmos-world-models",
)
node(
    "dreams",
    "c4",
    "R1",
    "Cosmos-Dreams",
    "closed-loop AV world model",
    "orbit",
    "Real-time, action-conditioned driving world model. The policy emits 8 frames of actions; the model renders "
    "the next 8 frames. Reported 68 fps on one GB300 (as reported). Ranks AV policies the same way as NuRec.",
    "cosmos-dreams-loop.jpg",
    "07-cosmos-world-models",
)

# Simulate: R2 robotics frameworks
node(
    "isaacsim",
    "c1",
    "R2",
    "Isaac Sim",
    "robot sim · ROS 2 bridge",
    "monitor-cog",
    "Robot simulator built on Omniverse Kit: URDF/MJCF import, physical sensors, OmniGraph Action Graphs and a ROS 2 "
    "bridge. Software in the loop: the robot's real software runs against this virtual body.",
    "nav2-omnigraph-inspector.jpg",
    "02-isaac-sim-ros2-nav2",
)
node(
    "isaaclab",
    "c2",
    "R2",
    "Isaac Lab",
    "RL / IL · 1k–16k envs",
    "boxes",
    "GPU-parallel reinforcement- and imitation-learning framework on Isaac Sim, with PhysX or Newton underneath. "
    "Trains policies across thousands of environments, then exports them to ONNX.",
    "isaac-lab-rl-loop.jpg",
    "04-isaac-lab-rl",
)
node(
    "mimic",
    "c3",
    "R2",
    "GR00T Mimic & Dreams",
    "multiply demonstrations",
    "wand-sparkles",
    "Multiplies a few demonstrations: DexMimicGen turned 50 demos into 3,000 (as reported). GR00T Dreams uses a "
    "post-trained Cosmos to imagine new tasks and recovers actions with inverse dynamics.",
)
node(
    "arena",
    "c4",
    "R2",
    "Isaac Lab Arena",
    "parallel policy evaluation",
    "gauge",
    "Large-scale, parallel policy evaluation in simulated sites. A partner reported 100,000 parallel trials on its "
    "Arena-based benchmark (as reported).",
)

# Simulate: R3 physics & optimization
node(
    "newton",
    "c1",
    "R3",
    "Newton",
    "GPU physics for robots",
    "atom",
    "Open-source GPU physics engine (NVIDIA, Google DeepMind, Disney Research; Linux Foundation). Model / State / "
    "Control / Contacts / Solver, with MuJoCo, XPBD, VBD, MPM, Featherstone and Kamino solvers. Reported 2–3× steps/s "
    "versus Isaac Lab 2 on PhysX (as reported).",
    None,
    "06-newton-gpu-scale",
    "Newton's slide deck is marked confidential and is not shown. Source: the public SIGGRAPH 2026 talk and the "
    "Newton code.",
)
node(
    "physx",
    "c2",
    "R3",
    "PhysX",
    "industrial digital twins",
    "cog",
    "Omniverse's established physics engine for industrial twins; still an Isaac Lab backend. Newton's FAQ: "
    "'Will Newton replace PhysX? No.'",
)
node(
    "warp",
    "c3",
    "R3",
    "Warp · MuJoCo Warp",
    "GPU kernels in Python",
    "cpu",
    "NVIDIA Warp compiles Python kernels for GPU (and CPU). Newton is built on it; MuJoCo Warp is its main "
    "rigid-body backend. Also the base for custom solvers in engineering twins.",
    None,
    "11-industrial-twins",
)
node(
    "cuopt",
    "c4",
    "R3",
    "cuOpt",
    "GPU fleet routing",
    "route",
    "GPU-accelerated vehicle routing. In the forklift-fleet demo it assigned jobs and routes over a waypoint graph "
    "while Isaac Sim drove the forklifts.",
    "forklift-fleet-live.jpg",
    "09-cuopt-forklift-fleet",
)

# Simulate: R4 platform & sensors
node(
    "kit",
    "c1",
    "R4",
    "Omniverse Kit",
    "libraries: OV Stage · ovphysx",
    "layers",
    "Omniverse's app framework and its 2026 libraries for agents and apps: OV Stage, OVRTX, ovphysx, ovstorage, "
    "OVUI. Isaac Sim is a Kit app.",
    "omniverse-libraries-stack.jpg",
    "10-omniverse-kit-extension",
)
node(
    "rtx",
    "c2",
    "R4",
    "Sensor RTX · OVRTX",
    "camera · LiDAR · radar",
    "camera",
    "Physically based sensor simulation from the same scene: cameras with lens models, LiDAR and radar, plus depth "
    "and segmentation ground truth.",
    "av-data-pipeline.jpg",
    "06-newton-gpu-scale",
)
node(
    "replicator",
    "c3",
    "R4",
    "Replicator",
    "randomize → labels",
    "dices",
    "Domain randomization and synthetic data: randomize poses, materials, lighting and distractors, then write "
    "KITTI or COCO labels.",
    None,
    "05-replicator-sdg",
)
node(
    "stream",
    "c4",
    "R4",
    "Kit App Streaming",
    "OV Stream to browser",
    "cloud",
    "Streams a Kit app to a browser over WebRTC; the delivery path for real-time digital twins.",
    "omniverse-libraries-launch.jpg",
    "10-omniverse-kit-extension",
)

# Simulate: R5 data model
node(
    "openusd",
    "c1",
    "R5",
    "OpenUSD",
    "layers · LIVERPS",
    "file-stack",
    "The scene description every tool reads and writes. Layers hold opinions; the stage composes them by LIVERPS "
    "strength order.",
    "usd-content-journey.jpg",
    "01-openusd-simready",
)
node(
    "simready",
    "c2",
    "R5",
    "SimReady assets",
    "joints · colliders · mass",
    "package",
    "Content specification for assets a robot can use: base → sim → optimized layers, with physics, sensors, ROS "
    "and control as payloads behind one asset.usd.",
    "simready-robot-asset-structure.jpg",
    "01-openusd-simready",
)
node(
    "nucleus",
    "c3",
    "R5",
    "Nucleus · ovstorage",
    "shared assets, live sync",
    "hard-drive",
    "Shared storage and live collaboration for USD content.",
    "omniverse-overview.jpg",
)
node(
    "schemas",
    "c4",
    "R5",
    "AOUSD schemas",
    "UsdPhysics · deformables",
    "puzzle",
    "Standard schemas through the Alliance for OpenUSD: UsdPhysics today, deformable bodies proposed (Newton was the "
    "first implementation).",
    "simready-playbook.png",
)

# Deploy
node(
    "edgepolicy",
    "d1",
    "R1",
    "Cosmos3-Edge",
    "4B policy on Jetson Thor",
    "microchip",
    "The 4B Cosmos 3 model targets real-time robot policies on Jetson Orin and Thor; shown driving an arm on a Thor "
    "at about 130 W (as reported).",
    None,
    "07-cosmos-world-models",
)
node(
    "metropolis",
    "d2",
    "R1",
    "Metropolis · VSS",
    "vision agents for facilities",
    "cctv",
    "Outside-in view of warehouses and cities: multi-camera tracking, specialist agents and an LLM orchestrator, "
    "scored on answers and tool-call trajectories.",
    None,
    "07-cosmos-world-models",
)
node(
    "ros2",
    "d1",
    "R2",
    "ROS 2 stack",
    "Nav2 · MoveIt 2 · Isaac ROS",
    "network",
    "The robot's software: navigation (Nav2), arm planning (MoveIt 2), GPU-accelerated Isaac ROS packages. Tested "
    "against Isaac Sim over DDS before it ships.",
    "nav2-wizard-step6.jpg",
    "02-isaac-sim-ros2-nav2",
)
node(
    "avstack",
    "d2",
    "R2",
    "AV stack + Halos",
    "DRIVE AGX",
    "car",
    "End-to-end driving stack on DRIVE AGX with the Halos safety architecture: an independent classical stack, an "
    "arbitrator, redundant computers and a minimum-risk maneuver.",
    "av-simulation-evolution.jpg",
    "08-nurec-av",
)
node(
    "twin",
    "d2",
    "R3",
    "Real-time twin",
    "Kit CAE · AI surrogates",
    "factory",
    "Interactive engineering twin: AI surrogates answer design questions in real time inside an Omniverse app.",
    "industrial-engineering-twins.jpg",
    "11-industrial-twins",
)
node(
    "trt",
    "d1",
    "R4",
    "TensorRT · ONNX",
    "distilled · quantized",
    "zap",
    "The deploy format: policies and perception models exported to ONNX and compiled with TensorRT (FP8 / FP4) for "
    "Jetson Thor. Teachers are distilled into small students first.",
    None,
    "04-isaac-lab-rl",
)

AGENTS = {
    "id": "agents",
    "title": "Agents & skills",
    "sub": "Claude Code / Codex · NVIDIA skills · NuRec agent skills · content & physics agents",
    "icon": "terminal",
    "desc": "Every 2026 product in this stack ships skills for coding agents: Omniverse skills, NuRec's five agent "
    "skills, Cosmos Framework skill folders. Plan, change, test, verify.",
    "evidence": "omniverse-skills.jpg",
    "atlas": None,
    "evidence_note": None,
    "kind": "agents",
}

FLOWS = {
    "robot": ("Robot learning", "--f-robot"),
    "data": ("Perception & synthetic data", "--f-data"),
    "av": ("Autonomous vehicles", "--f-av"),
    "ops": ("Industrial & operations", "--f-ops"),
    "loop": ("Feedback loop", "--f-loop"),
}


def box(nid):
    if nid == "agents":
        return {"x0": 440, "x1": 1690, "y0": 58, "y1": 98, "cx": 1065, "cy": 78}
    n = N[nid]
    cx, cy = COLS[n["col"]], ROWS[n["row"]]
    w = WIDTH.get(n["col"], DEFAULT_W)
    return {
        "x0": cx - w / 2,
        "x1": cx + w / 2,
        "y0": cy - NODE_H / 2,
        "y1": cy + NODE_H / 2,
        "cx": cx,
        "cy": cy,
    }


def top(n, dx=0.0):
    b = box(n)
    return (b["cx"] + dx, b["y0"])


def bot(n, dx=0.0):
    b = box(n)
    return (b["cx"] + dx, b["y1"])


def lef(n, dy=0.0):
    b = box(n)
    return (b["x0"], b["cy"] + dy)


def rig(n, dy=0.0):
    b = box(n)
    return (b["x1"], b["cy"] + dy)


# (id, from, to, label, flow, points, label_segment_index, label_t)
E = []


def edge(src, dst, label, flow, pts, seg=None, t=0.5, two_way=False):
    E.append(
        {
            "src": src,
            "dst": dst,
            "label": label,
            "flow": flow,
            "pts": pts,
            "seg": seg,
            "t": t,
            "two_way": two_way,
        }
    )


def path(a, *wps, b):
    """Orthogonal route from point a through waypoints ('x', v) / ('y', v) to point b."""
    pts = [a]
    cur = a
    for axis, v in wps:
        cur = (v, cur[1]) if axis == "x" else (cur[0], v)
        pts.append(cur)
    # finish: align on the axis that keeps the last approach perpendicular to b's side
    return pts, b


def route(a, wps, b, end_side):
    pts, b = path(a, *wps, b=b)
    cur = pts[-1]
    if end_side in ("top", "bottom"):
        if cur[0] != b[0]:
            pts.append((b[0], cur[1]))
    elif cur[1] != b[1]:
        pts.append((cur[0], b[1]))
    pts.append(b)
    return pts


# One line per flow: (source, target, label, flow, route, label segment, label position along it).
# Routes run through the gutters between rows and columns; y/x values below are gutter lanes.
# --- data in
edge(
    "web",
    "cosmostrain",
    "pre-training video",
    "av",
    route(top("web", 20), [("y", 142)], top("cosmostrain", -40), "top"),
    seg=1,
)
edge("teleop", "vla", "teleop demos", "robot", route(top("teleop", 30), [("y", 280)], top("vla", -50), "top"), seg=1)
edge(
    "fleet",
    "nurec",
    "sensor logs",
    "av",
    route(rig("fleet", -10), [("x", 212), ("y", 244)], bot("nurec", -30), "bottom"),
    seg=2,
    t=0.45,
)
edge(
    "cad",
    "openusd",
    "URDF · MJCF · CAD import",
    "robot",
    route(bot("cad"), [("y", 756)], bot("openusd", -30), "bottom"),
    seg=1,
    t=0.62,
)
# --- train → simulate / deploy
edge(
    "cosmostrain",
    "gen",
    "post-trained weights",
    "data",
    route(top("cosmostrain", 0), [("y", 130)], top("gen", -40), "top"),
    seg=1,
    t=0.78,
)
edge(
    "cosmostrain",
    "edgepolicy",
    "distilled 4B Edge model",
    "robot",
    route(top("cosmostrain", 40), [("y", 114)], top("edgepolicy", 0), "top"),
    seg=1,
    t=0.6,
)
edge(
    "pnemo",
    "twin",
    "trained surrogate",
    "ops",
    route(bot("pnemo", 0), [("y", 534)], bot("twin", -30), "bottom"),
    seg=1,
    t=0.62,
)
edge(
    "tao",
    "trt",
    "detector → TensorRT engine",
    "data",
    route(bot("tao", 30), [("y", 650)], bot("trt", 20), "bottom"),
    seg=1,
    t=0.82,
)
# --- data model & platform
edge(
    "openusd",
    "simready",
    "SimReady layering",
    "robot",
    route(bot("openusd", 30), [("y", 760)], bot("simready", -30), "bottom"),
    seg=1,
)
edge(
    "simready",
    "kit",
    "composed stage",
    "robot",
    route(top("simready", -40), [("y", 640)], bot("kit", 30), "bottom"),
    seg=1,
)
edge(
    "kit",
    "isaacsim",
    "Isaac Sim is a Kit app",
    "robot",
    route(rig("kit", -10), [("x", 650)], rig("isaacsim", 10), "right"),
    seg=1,
    t=0.13,
)
edge(
    "rtx",
    "gen",
    "depth · seg · edge renders",
    "data",
    route(rig("rtx", -10), [("x", 860), ("y", 250)], bot("gen", 0), "bottom"),
    seg=1,
    t=0.11,
)
edge(
    "replicator",
    "tao",
    "labeled frames (KITTI / COCO)",
    "data",
    route(bot("replicator", 0), [("y", 628)], bot("tao", -20), "bottom"),
    seg=1,
    t=0.28,
)
edge(
    "stream",
    "twin",
    "stream to browser",
    "ops",
    route(bot("stream", 30), [("y", 640)], bot("twin", 30), "bottom"),
    seg=1,
    t=0.62,
)
# --- physics
edge(
    "newton",
    "isaaclab",
    "Newton · 16k worlds",
    "robot",
    route(top("newton", 30), [("y", 386)], bot("isaaclab", -30), "bottom"),
    seg=1,
    t=0.3,
)
edge(
    "physx",
    "isaacsim",
    "PhysX",
    "robot",
    route(top("physx", -30), [("y", 366)], bot("isaacsim", 30), "bottom"),
    seg=1,
    t=0.3,
)
edge(
    "warp",
    "newton",
    "Warp kernels",
    "robot",
    route(bot("warp", -20), [("y", 512)], bot("newton", 0), "bottom"),
    seg=1,
    t=0.35,
)
edge("cuopt", "twin", "routes & tasks", "ops", route(rig("cuopt", 0), [], lef("twin", 0), "left"), seg=0, t=0.22)
# --- world models
edge("nurec", "isaacsim", "USDZ scene", "av", route(bot("nurec", 30), [], top("isaacsim", 30), "top"), seg=0, t=0.5)
edge(
    "reasoner",
    "gen",
    "grades & filters",
    "data",
    route(top("reasoner", -40), [("y", 146)], top("gen", 40), "top"),
    seg=1,
    t=0.5,
)
edge(
    "gen",
    "vla",
    "photoreal synthetic video",
    "data",
    route(bot("gen", -40), [("y", 258)], top("vla", 10), "top"),
    seg=1,
    t=0.45,
)
edge(
    "gen",
    "mimic",
    "dreamed videos",
    "robot",
    route(bot("gen", 40), [("y", 264)], top("mimic", -30), "top"),
    seg=1,
    t=0.55,
)
edge(
    "mimic",
    "vla",
    "synthetic trajectories",
    "robot",
    route(bot("mimic", 0), [("y", 408)], bot("vla", 0), "bottom"),
    seg=1,
    t=0.12,
)
edge(
    "reasoner",
    "arena",
    "judges rollouts",
    "robot",
    route(bot("reasoner", 30), [("y", 252)], top("arena", -30), "top"),
    seg=1,
    t=0.55,
)
edge(
    "reasoner",
    "metropolis",
    "VLM reasoning over video",
    "ops",
    route(top("reasoner", 30), [("y", 122)], top("metropolis", 0), "top"),
    seg=1,
    t=0.72,
)
edge(
    "dreams",
    "avstack",
    "8 actions ↔ 8 frames",
    "av",
    route(rig("dreams", 0), [("x", 1292), ("y", 236)], top("avstack", 0), "top"),
    seg=2,
    t=0.5,
    two_way=True,
)
# --- frameworks → deploy
edge(
    "isaaclab",
    "trt",
    "ONNX policy",
    "robot",
    route(top("isaaclab", 30), [("y", 274), ("x", 1286)], lef("trt", 0), "left"),
    seg=1,
    t=0.86,
)
edge(
    "isaacsim",
    "ros2",
    "DDS · /cmd_vel /scan /tf",
    "robot",
    route(top("isaacsim", -30), [("y", 284), ("x", 1300)], lef("ros2", 0), "left"),
    seg=1,
    t=0.42,
    two_way=True,
)
edge("trt", "ros2", "runs on Jetson Thor", "robot", route(top("trt", 0), [], bot("ros2", 0), "bottom"), seg=0, t=0.3)
# --- agents
edge(
    "agents",
    "nurec",
    "agent skills",
    "loop",
    route((box("nurec")["cx"] - 40, 98), [], top("nurec", -40), "top"),
    seg=0,
    t=0.42,
)
# --- feedback loop around the whole system
edge(
    "avstack",
    "fleet",
    "field logs & failures feed back",
    "loop",
    route(rig("avstack", 0), [("x", 1698), ("y", 848), ("x", 12)], lef("fleet", 0), "left"),
    seg=2,
    t=0.5,
)


# ----------------------------------------------------------------------------------------------- rendering
def rounded_path(pts, r=8.0):
    if len(pts) < 2:
        return ""
    d = [f"M{pts[0][0] + XO:.1f},{pts[0][1]:.1f}"]
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        l1 = max(abs(x1 - x0), abs(y1 - y0))
        l2 = max(abs(x2 - x1), abs(y2 - y1))
        rr = min(r, l1 / 2, l2 / 2)

        def towards(a, b, dist):
            ax, ay = a
            bx, by = b
            length = max(abs(bx - ax), abs(by - ay)) or 1
            return (bx + (ax - bx) * dist / length, by + (ay - by) * dist / length)

        p1 = towards((x0, y0), (x1, y1), rr)
        p2 = towards((x2, y2), (x1, y1), rr)
        d.append(f"L{p1[0] + XO:.1f},{p1[1]:.1f}")
        d.append(f"Q{x1 + XO:.1f},{y1:.1f} {p2[0] + XO:.1f},{p2[1]:.1f}")
    d.append(f"L{pts[-1][0] + XO:.1f},{pts[-1][1]:.1f}")
    return " ".join(d)


def label_pos(e):
    pts = e["pts"]
    segs = list(itertools.pairwise(pts))
    i = e["seg"] if e["seg"] is not None else max(range(len(segs)), key=lambda k: abs(segs[k][1][0] - segs[k][0][0]))
    (x0, y0), (x1, y1) = segs[i]
    t = e["t"]
    return x0 + (x1 - x0) * t + XO, y0 + (y1 - y0) * t


def load_icon(lucide_dir: Path, name: str) -> str:
    svg = (lucide_dir / f"{name}.svg").read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", svg, re.DOTALL).group(1)
    inner = re.sub(r"\s+", " ", inner).strip()
    return f'<symbol id="i-{name}" viewBox="0 0 24 24">{inner}</symbol>'


def esc(s):
    return html.escape(s, quote=True)


def text_w(s, size):  # rough width of IBM Plex Sans Condensed
    return len(s) * size * 0.5 + 10


def render_svg(lucide_dir: Path) -> str:
    icons = sorted({n["icon"] for n in N.values()} | {AGENTS["icon"]})
    out = [
        f'<svg id="map" viewBox="0 0 {VB_W} {VB_H}" role="img" aria-label="Physical AI system map: components '
        f'placed by stack layer and by the computer they run on, with labeled data flows between them.">',
        "<defs>",
        *[load_icon(lucide_dir, i) for i in icons],
    ]
    for key in FLOWS:
        out.append(
            f'<marker id="ah-{key}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ah ah-{key}"/></marker>'
        )
    out.append("</defs>")
    # column bands
    for cid, x0, x1, title, sub in COLUMNS:
        out.append(
            f'<rect class="colband col-{cid}" x="{x0 + XO}" y="8" width="{x1 - x0}" height="{820 - 8}" rx="10"/>'
        )
        out.append(f'<text class="coltitle col-{cid}-t" x="{x0 + XO + 12}" y="30">{esc(title.upper())}</text>')
        out.append(f'<text class="colsub" x="{x0 + XO + 12 + len(title) * 14 * 0.72 + 10}" y="30">{esc(sub)}</text>')
    # layer separators + labels
    for _lid, y0, y1, _label in LAYERS:
        out.append(f'<line class="layerline" x1="{XO - 34}" x2="{VB_W - 10}" y1="{y0}" y2="{y0}"/>')
        cy = (y0 + y1) / 2
        out.append(
            f'<text class="layerlabel" transform="translate({XO - 16},{cy}) rotate(-90)" text-anchor="middle">'
            f"{esc(SHORT[_lid].upper())}</text>"
        )
    # hardware row
    hw = [
        ("real", "Robots · cars · facilities"),
        ("train", "DGX · GB200"),
        ("sim", "OVX · RTX PRO 6000 · DGX Spark"),
        ("deploy", "Jetson Thor · DRIVE AGX"),
    ]
    for cid, label in hw:
        _, x0, x1, *_ = next(c for c in COLUMNS if c[0] == cid)
        out.append(f'<rect class="hw" x="{x0 + XO + 8}" y="776" width="{x1 - x0 - 16}" height="36" rx="6"/>')
        out.append(f'<use href="#i-server" class="hwicon" x="{x0 + XO + 18}" y="785" width="18" height="18"/>')
        out.append(f'<text class="hwtext" x="{x0 + XO + 44}" y="799">{esc(label)}</text>')
    # edges (under nodes)
    out.append('<g id="edges">')
    for k, e in enumerate(E):
        d = rounded_path(e["pts"])
        start = f' marker-start="url(#ah-{e["flow"]})"' if e["two_way"] else ""
        out.append(
            f'<path class="edge f-{e["flow"]}" data-e="{k}" data-src="{e["src"]}" data-dst="{e["dst"]}" '
            f'd="{d}" marker-end="url(#ah-{e["flow"]})"{start}/>'
        )
    out.append("</g>")
    # nodes
    out.append('<g id="nodes">')
    a = box("agents")
    out.append(
        f'<g class="node k-agents" data-n="agents" tabindex="0" role="button" aria-label="Agents and skills">'
        f'<rect x="{a["x0"] + XO}" y="{a["y0"]}" width="{a["x1"] - a["x0"]}" height="{a["y1"] - a["y0"]}" rx="8"/>'
        f'<use href="#i-{AGENTS["icon"]}" class="nicon" x="{a["x0"] + XO + 12}" y="{a["y0"] + 10}" width="20" '
        f'height="20"/><text class="ntitle" x="{a["x0"] + XO + 42}" y="{a["y0"] + 25}">{esc(AGENTS["title"])}</text>'
        f'<text class="nsub" x="{a["x0"] + XO + 42 + text_w(AGENTS["title"], 13) + 6}" y="{a["y0"] + 25}">'
        f"{esc(AGENTS['sub'])}</text></g>"
    )
    for nid, n in N.items():
        b = box(nid)
        x0 = b["x0"] + XO
        out.append(
            f'<g class="node k-{n["kind"]}" data-n="{nid}" tabindex="0" role="button" aria-label="{esc(n["title"])}">'
            f'<rect x="{x0}" y="{b["y0"]}" width="{b["x1"] - b["x0"]}" height="{NODE_H}" rx="8"/>'
            f'<use href="#i-{n["icon"]}" class="nicon" x="{x0 + 10}" y="{b["cy"] - 10}" width="20" height="20"/>'
            f'<text class="ntitle" x="{x0 + 38}" y="{b["cy"] - 3}">{esc(n["title"])}</text>'
            f'<text class="nsub" x="{x0 + 38}" y="{b["cy"] + 14}">{esc(n["sub"])}</text>'
            + (
                '<circle class="evdot" cx="{:.1f}" cy="{:.1f}" r="3.2"/>'.format(b["x1"] + XO - 9, b["y0"] + 9)
                if n["evidence"]
                else ""
            )
            + "</g>"
        )
    out.append("</g>")
    # edge labels (above everything)
    out.append('<g id="labels">')
    for k, e in enumerate(E):
        if not e["label"]:
            continue
        x, y = label_pos(e)
        w = text_w(e["label"], 11)
        out.append(
            f'<g class="elabel f-{e["flow"]}" data-e="{k}"><rect x="{x - w / 2:.1f}" y="{y - 9:.1f}" width="{w:.1f}" '
            f'height="18" rx="9"/><text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle">{esc(e["label"])}</text></g>'
        )
    out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


def font_face(fonts_dir: Path) -> str:
    css = []
    for weight in (400, 600, 700):
        f = fonts_dir / f"ibm-plex-sans-condensed-latin-{weight}-normal.woff2"
        b64 = base64.b64encode(f.read_bytes()).decode()
        css.append(
            f"@font-face{{font-family:'Plex Condensed';font-style:normal;font-weight:{weight};font-display:swap;"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}"
        )
    return "\n".join(css)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lucide", type=Path, required=True)
    ap.add_argument("--fonts", type=Path, required=True)
    args = ap.parse_args()

    svg = render_svg(args.lucide)
    data = {
        "nodes": {
            **{
                k: {
                    kk: v[kk]
                    for kk in (
                        "title",
                        "sub",
                        "desc",
                        "evidence",
                        "atlas",
                        "evidence_note",
                        "kind",
                        "row",
                        "col",
                    )
                }
                for k, v in N.items()
            },
            "agents": {
                **{
                    kk: AGENTS[kk]
                    for kk in (
                        "title",
                        "sub",
                        "desc",
                        "evidence",
                        "atlas",
                        "evidence_note",
                        "kind",
                    )
                },
                "row": "R0",
                "col": "all",
            },
        },
        "edges": [{"src": e["src"], "dst": e["dst"], "label": e["label"], "flow": e["flow"]} for e in E],
        "flows": {k: v[0] for k, v in FLOWS.items()},
        "layers": {lid: label for lid, *_r, label in LAYERS},
    }
    template = (Path(__file__).resolve().parent / "system_map_template.html").read_text()
    page = (
        template.replace("/*FONT_FACE*/", font_face(args.fonts))
        .replace("<!--SVG-->", svg)
        .replace("/*DATA*/", json.dumps(data, ensure_ascii=False))
    )
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT.relative_to(PROJ)} ({OUT.stat().st_size // 1024} KB, {len(N) + 1} components, {len(E)} flows)")


if __name__ == "__main__":
    main()
