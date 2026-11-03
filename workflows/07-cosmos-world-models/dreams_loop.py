#!/usr/bin/env python3
"""Closed-loop generative simulation, the Cosmos-Dreams pattern, as interfaces plus a CPU toy that runs end to end.

CONCEPTUAL REPRODUCTION, NOT COSMOS-DREAMS. No neural network is involved. The "world model" here is a rasterizer of
the abstract state; it stands in for Cosmos-Dreams so the loop, the chunking, the history cache, and the evaluation
can be exercised on a laptop.

The loop on the SIGGRAPH 2026 slide "Simulation Loop with Cosmos-Dreams" (assets/screens/cosmos-dreams-loop.jpg) and
in the talk transcript (24:22-25:59):

    Policy (Alpamayo) or user --Action (next 8 frames)--> Simulation runtime (AlpaSim)
        --Abstract state update (HD map + agent boxes, incl. next actions)--> World model (Cosmos-Dreams)
        --Synthesized camera frames (8 RGB frames = 2 latent frames)--> back to the policy

The world model's inputs, per the talk: a text prompt, the first RGB frame, the history in a KV cache, and the
abstract next state. Cosmos-Dreams is built on Cosmos 2.5 (a Cosmos 3 version was "in progress" at the talk).

Toy stand-ins (all CPU, NumPy):

* ``ToyRuntime``        - kinematic bicycle ego, a two-lane sinusoidal road (the "HD map"), a lead car that brakes, an
                          oncoming car. Holds the true state; scores progress, collisions, lane departures.
* ``RasterWorldModel``  - renders each abstract state to a 2-channel bird's-eye frame (lane lines, agent boxes). Keeps a
                          fixed-size ring buffer of past frames (the analogue of the "streaming static-shape KV cache"
                          on the FlashDreams slide). Optional degradations model a weaker generator: pixel noise, box
                          jitter, agents that vanish for a chunk, lane lines that drift, and ghosting from history.
* ``FramePolicy``       - reads only the rendered frame (plus its own speed), fits the lane, finds the lead car, and
                          emits the next 8 actions (pure pursuit + IDM), open loop within the chunk.

The validation the talk reports (37:50-38:14): the stack ranking of policies stayed the same in NuRec and in
Cosmos-Dreams. ``rank`` runs five policies through the exact renderer and two degraded ones and reports Kendall tau
between the policy rankings.

Usage::

    python dreams_loop.py rank --seeds 10 --out results/           # ranking study, writes JSON/CSV + preview PNG
    python dreams_loop.py export-wsm --out /tmp/wsm --frames 101   # WSM-like control frames (PNG) for specs/av
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np

FRAMES_PER_CHUNK = 8  # "the next eight frames" (talk 24:43, 27:07)
DT = 0.1  # s per frame; toy choice (the Cosmos 3 cookbook's AV action domain runs at 10 fps)
WHEELBASE = 2.7
CAR_L, CAR_W = 4.5, 1.9
LANE_W = 3.5
LINE_OFFSETS = (-LANE_W, 0.0, LANE_W)  # right road edge, center line, left road edge (y is left-positive)
EGO_LANE = -LANE_W / 2  # drive in the right lane

# Bird's-eye raster: rows go forward (up), columns go right.
BEV_H, BEV_W, BEV_RES = 64, 48, 0.5  # 32 m forward x 24 m wide
EGO_ROW, EGO_COL = 56, 24


# ---------------------------------------------------------------------------
# Abstract state: HD map + agent boxes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoadMap:
    amplitude: float = 2.0
    wavelength: float = 120.0

    def center_y(self, x):
        return self.amplitude * np.sin(2 * np.pi * np.asarray(x) / self.wavelength)

    def heading(self, x):
        return np.arctan(
            self.amplitude * 2 * np.pi / self.wavelength * np.cos(2 * np.pi * np.asarray(x) / self.wavelength)
        )


@dataclass(frozen=True)
class Box:
    """An agent box on the ground plane: center, yaw, size."""

    id: str
    x: float
    y: float
    yaw: float
    length: float = CAR_L
    width: float = CAR_W


@dataclass
class EgoState:
    x: float
    y: float
    yaw: float
    v: float


@dataclass(frozen=True)
class AbstractState:
    t: float
    ego: EgoState
    agents: tuple[Box, ...]
    road: RoadMap


def _iround(x: float) -> int:
    return math.floor(float(x) + 0.5)


def obb_corners(b: Box) -> np.ndarray:
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    hl, hw = b.length / 2, b.width / 2
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return local @ np.array([[c, s], [-s, c]]) + np.array([b.x, b.y])


def obb_overlap(a: Box, b: Box) -> bool:
    """Separating-axis test for two oriented rectangles."""
    ca, cb = obb_corners(a), obb_corners(b)
    for yaw in (a.yaw, b.yaw):
        for axis in (np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])):
            pa, pb = ca @ axis, cb @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


# ---------------------------------------------------------------------------
# Interfaces (what a real Policy / AlpaSim / Cosmos-Dreams binding would implement)
# ---------------------------------------------------------------------------


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, frame: np.ndarray, ego_speed: float) -> np.ndarray:
        """Return the next ``FRAMES_PER_CHUNK`` actions, shape (8, 2): [acceleration m/s^2, steering rad]."""
        ...


class SimulationRuntime(Protocol):
    def reset(self) -> AbstractState: ...

    def step(self, actions: np.ndarray) -> list[AbstractState]:
        """Apply one action chunk; return one abstract state per frame (empty if the episode ended)."""
        ...


class WorldModel(Protocol):
    def reset(self, first_frame: np.ndarray, prompt: str) -> None: ...

    def generate(self, states: Sequence[AbstractState]) -> np.ndarray:
        """Render one frame per abstract state, conditioned on the history cache. Shape (n, C, H, W)."""
        ...


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    seed: int
    road: RoadMap
    lead_gap: float
    lead_speed: float
    brake_at: float
    brake_decel: float
    brake_dur: float
    oncoming_x: float
    oncoming_speed: float

    @classmethod
    def sample(cls, seed: int) -> Scenario:
        r = np.random.default_rng(seed)
        return cls(
            seed=seed,
            road=RoadMap(amplitude=float(r.uniform(1.0, 4.0)), wavelength=float(r.uniform(80.0, 160.0))),
            lead_gap=float(r.uniform(20.0, 40.0)),
            lead_speed=float(r.uniform(9.0, 13.0)),
            brake_at=float(r.uniform(3.0, 8.0)),
            brake_decel=float(r.uniform(3.0, 6.0)),
            brake_dur=float(r.uniform(1.0, 2.5)),
            oncoming_x=float(r.uniform(60.0, 120.0)),
            oncoming_speed=float(r.uniform(8.0, 12.0)),
        )


@dataclass
class EpisodeMetrics:
    progress_m: float = 0.0
    collided: bool = False
    departure_s: float = 0.0
    min_gap_m: float = float("inf")
    steps: int = 0

    @property
    def score(self) -> float:
        """Driving score: metres of progress, -100 for a collision, -10 per second outside the lane."""
        return self.progress_m - 100.0 * self.collided - 10.0 * self.departure_s


class ToyRuntime:
    """AlpaSim stand-in: owns the true state, integrates the ego, moves the agents, scores the episode."""

    def __init__(self, scenario: Scenario, ego_speed: float = 10.0):
        self.sc = scenario
        self.v0 = ego_speed
        self.reset()

    def reset(self) -> AbstractState:
        road = self.sc.road
        self.t = 0.0
        self.ego = EgoState(0.0, float(road.center_y(0.0)) + EGO_LANE, float(road.heading(0.0)), self.v0)
        self.lead_x, self.lead_v = self.sc.lead_gap, self.sc.lead_speed
        self.onc_x = self.sc.oncoming_x
        self.metrics = EpisodeMetrics()
        self.done = False
        return self.state()

    @property
    def ego_speed(self) -> float:
        return self.ego.v

    def _agents(self) -> tuple[Box, ...]:
        road = self.sc.road
        lead = Box("lead", self.lead_x, float(road.center_y(self.lead_x)) + EGO_LANE, float(road.heading(self.lead_x)))
        onc_yaw = float(road.heading(self.onc_x)) + math.pi
        onc = Box("oncoming", self.onc_x, float(road.center_y(self.onc_x)) - EGO_LANE, onc_yaw)
        return (lead, onc)

    def state(self) -> AbstractState:
        e = self.ego
        return AbstractState(round(self.t, 6), EgoState(e.x, e.y, e.yaw, e.v), self._agents(), self.sc.road)

    def step(self, actions: np.ndarray) -> list[AbstractState]:
        if self.done:
            return []
        out = []
        for a_raw, steer_raw in np.asarray(actions, dtype=float).reshape(-1, 2):
            a, steer = float(np.clip(a_raw, -8.0, 3.0)), float(np.clip(steer_raw, -0.5, 0.5))
            e = self.ego
            e.x += e.v * math.cos(e.yaw) * DT
            e.y += e.v * math.sin(e.yaw) * DT
            e.yaw += e.v / WHEELBASE * math.tan(steer) * DT
            e.v = max(0.0, e.v + a * DT)
            # agents
            braking = self.sc.brake_at <= self.t < self.sc.brake_at + self.sc.brake_dur
            if braking:
                self.lead_v = max(0.0, self.lead_v - self.sc.brake_decel * DT)
            self.lead_x += self.lead_v * DT
            self.onc_x -= self.sc.oncoming_speed * DT
            self.t += DT
            # scoring on the true state
            m = self.metrics
            m.steps += 1
            m.progress_m = e.x
            lat = e.y - float(self.sc.road.center_y(e.x)) - EGO_LANE
            if abs(lat) > (LANE_W - CAR_W) / 2:
                m.departure_s += DT
            ego_box = Box("ego", e.x, e.y, e.yaw)
            agents = self._agents()
            m.min_gap_m = min(m.min_gap_m, self.lead_x - e.x - CAR_L)
            if any(obb_overlap(ego_box, b) for b in agents):
                m.collided = True
                self.done = True
            out.append(self.state())
            if self.done:
                break
        return out


# ---------------------------------------------------------------------------
# World model (renderer) with a static-shape history cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Degradation:
    name: str = "exact"
    noise_std: float = 0.0  # pixel noise
    box_jitter_m: float = 0.0  # per-frame box position error
    drop_prob: float = 0.0  # per chunk and agent: the agent is not rendered (object permanence failure)
    lane_drift_m: float = 0.0  # stationary std of a slowly drifting lateral bias of the rendered lane lines
    ghost: float = 0.0  # blend weight of the previous frame (over-reliance on history)


EXACT = Degradation()
MILD = Degradation("mild", noise_std=0.10, box_jitter_m=0.2)
MODERATE = Degradation("moderate", noise_std=0.15, box_jitter_m=0.4, drop_prob=0.1, lane_drift_m=0.3, ghost=0.25)
SEVERE = Degradation("severe", noise_std=0.25, box_jitter_m=0.8, drop_prob=0.25, lane_drift_m=0.8, ghost=0.5)
RENDERERS = {d.name: d for d in (EXACT, MILD, MODERATE, SEVERE)}


def _pixel_world(ego: EgoState) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.mgrid[0:BEV_H, 0:BEV_W]
    fwd = (EGO_ROW - rows) * BEV_RES
    left = (EGO_COL - cols) * BEV_RES
    c, s = math.cos(ego.yaw), math.sin(ego.yaw)
    return ego.x + fwd * c - left * s, ego.y + fwd * s + left * c


def rasterize(state: AbstractState, lane_bias: float = 0.0, boxes: Sequence[Box] | None = None) -> np.ndarray:
    """Exact bird's-eye render of one abstract state: channel 0 lane lines, channel 1 agent boxes."""
    X, Y = _pixel_world(state.ego)
    frame = np.zeros((2, BEV_H, BEV_W), dtype=np.float32)
    d = Y - state.road.center_y(X) - lane_bias
    for off in LINE_OFFSETS:
        frame[0][np.abs(d - off) < 0.3] = 1.0
    for b in state.agents if boxes is None else boxes:
        dx, dy = X - b.x, Y - b.y
        u = dx * math.cos(b.yaw) + dy * math.sin(b.yaw)
        v = -dx * math.sin(b.yaw) + dy * math.cos(b.yaw)
        frame[1][(np.abs(u) < b.length / 2) & (np.abs(v) < b.width / 2)] = 1.0
    return frame


class RasterWorldModel:
    """Cosmos-Dreams stand-in. ``cache`` is a fixed-shape ring buffer of the last ``history`` frames."""

    def __init__(self, degradation: Degradation = EXACT, seed: int = 0, history: int = 2 * FRAMES_PER_CHUNK):
        self.deg = degradation
        self.seed = seed
        self.history = history
        self.cache = np.zeros((history, 2, BEV_H, BEV_W), dtype=np.float32)
        self.ptr = 0
        self.prompt = ""

    def reset(self, first_frame: np.ndarray, prompt: str) -> None:
        self.rng = np.random.default_rng(self.seed)  # same noise stream for every policy (common random numbers)
        self.cache[:] = first_frame  # the first frame seeds the whole history
        self.ptr = 0
        self.prompt = prompt
        self.bias = 0.0

    def _push(self, frame: np.ndarray) -> None:
        self.cache[self.ptr % self.history] = frame
        self.ptr += 1

    def last(self) -> np.ndarray:
        return self.cache[(self.ptr - 1) % self.history]

    def generate(self, states: Sequence[AbstractState]) -> np.ndarray:
        g, r = self.deg, self.rng
        dropped = {b.id for b in (states[0].agents if states else ()) if r.random() < g.drop_prob}
        frames = []
        tau = 3.0  # s, correlation time of the lane drift (Ornstein-Uhlenbeck)
        for st in states:
            if g.lane_drift_m:
                self.bias += -self.bias * DT / tau + g.lane_drift_m * math.sqrt(2 * DT / tau) * r.normal()
            boxes = [
                replace(b, x=b.x + g.box_jitter_m * r.normal(), y=b.y + g.box_jitter_m * r.normal())
                for b in st.agents
                if b.id not in dropped
            ]
            f = rasterize(st, self.bias, boxes)
            if g.ghost:
                f = (1 - g.ghost) * f + g.ghost * self.last()
            if g.noise_std:
                f = f + r.normal(0.0, g.noise_std, f.shape).astype(np.float32)
            f = np.clip(f, 0.0, 1.0).astype(np.float32)
            self._push(f)
            frames.append(f)
        return np.stack(frames) if frames else np.zeros((0, 2, BEV_H, BEV_W), np.float32)


# ---------------------------------------------------------------------------
# Policy that drives from pixels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyParams:
    name: str
    v0: float  # desired speed
    T: float  # time headway
    s0: float  # minimum gap
    a_max: float = 2.0
    b: float = 2.5
    k_steer: float = 1.0
    lookahead_s: float = 0.8


POLICIES = (
    PolicyParams("cautious", v0=10.0, T=2.0, s0=4.0, a_max=1.5, b=2.0),
    PolicyParams("nominal", v0=13.0, T=1.5, s0=3.0),
    PolicyParams("eager", v0=16.0, T=1.0, s0=2.5, a_max=2.5, b=3.0),
    PolicyParams("understeer", v0=13.0, T=1.5, s0=3.0, k_steer=0.35),
    PolicyParams("tailgater", v0=17.0, T=0.5, s0=1.0, a_max=3.0, b=4.0),
)


def _clusters(cols: np.ndarray) -> list[float]:
    if cols.size == 0:
        return []
    groups, cur = [], [cols[0]]
    for c in cols[1:]:
        if c - cur[-1] <= 1:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]


def perceive_lane(frame: np.ndarray) -> np.ndarray | None:
    """Fit the ego lane center (lateral offset vs forward distance, ego frame) from the lane-line channel."""
    lanes = frame[0]
    ds, cs = [], []
    for d in (3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0):
        row = _iround(EGO_ROW - d / BEV_RES)
        band = lanes[max(row - 1, 0) : row + 2].mean(axis=0)
        lat = sorted((EGO_COL - c) * BEV_RES for c in _clusters(np.flatnonzero(band > 0.5)))
        if len(lat) >= 2 and 2.5 <= lat[1] - lat[0] <= 4.5:
            center = 0.5 * (lat[0] + lat[1])
        elif lat:
            center = lat[0] + LANE_W / 2
        else:
            continue
        ds.append(d)
        cs.append(center)
    if len(ds) < 2:
        return None
    return np.polyfit(ds, cs, 2 if len(ds) >= 4 else 1)


def perceive_gap(frame: np.ndarray, lane: np.ndarray) -> float:
    """Bumper gap to the first agent pixel inside the ego lane corridor; inf if none.

    A 3x3 mean filter before thresholding suppresses isolated noise pixels (phantom obstacles), as any real
    detector would; agents that the renderer drops or smears still fool it.
    """
    padded = np.pad(frame[1], 1, mode="edge")
    agents = sum(padded[i : i + BEV_H, j : j + BEV_W] for i in range(3) for j in range(3)) / 9.0
    for row in range(EGO_ROW - int(CAR_L / 2 / BEV_RES) - 1, -1, -1):
        d = (EGO_ROW - row) * BEV_RES
        c = float(np.polyval(lane, d))
        lo = _iround(EGO_COL - (c + 1.0) / BEV_RES)
        hi = _iround(EGO_COL - (c - 1.0) / BEV_RES)
        lo, hi = max(lo, 0), min(hi, BEV_W - 1)
        if lo <= hi and (agents[row, lo : hi + 1] > 0.5).any():
            return d - CAR_L / 2
    return float("inf")


class FramePolicy:
    """Pure pursuit on the perceived lane + IDM on the perceived gap; plans 8 actions open loop."""

    def __init__(self, p: PolicyParams):
        self.p = p
        self.name = p.name
        self.reset()

    def reset(self) -> None:
        self.lane = np.array([0.0, 0.0, 0.0])
        self.prev_gap = float("inf")

    def act(self, frame: np.ndarray, ego_speed: float) -> np.ndarray:
        p = self.p
        lane = perceive_lane(frame)
        if lane is not None:
            self.lane = lane
        gap = perceive_gap(frame, self.lane)
        chunk_t = FRAMES_PER_CHUNK * DT
        v_lead = ego_speed
        if math.isfinite(gap) and math.isfinite(self.prev_gap):
            v_lead = float(np.clip(ego_speed + (gap - self.prev_gap) / chunk_t, 0.0, 40.0))
        self.prev_gap = gap
        x = y = yaw = 0.0
        v, s = ego_speed, gap
        acts = np.zeros((FRAMES_PER_CHUNK, 2))
        for k in range(FRAMES_PER_CHUNK):
            # IDM
            free = 1.0 - (v / p.v0) ** 4
            if math.isfinite(s):
                s_star = p.s0 + v * p.T + v * (v - v_lead) / (2 * math.sqrt(p.a_max * p.b))
                acc = p.a_max * (free - (max(s_star, 0.0) / max(s, 0.1)) ** 2)
            else:
                acc = p.a_max * free
            acc = float(np.clip(acc, -8.0, p.a_max))
            # pure pursuit toward the lane center at the lookahead distance
            ld = max(6.0, p.lookahead_s * v)
            tx = x + ld
            ty = float(np.polyval(self.lane, tx))
            lx = (tx - x) * math.cos(yaw) + (ty - y) * math.sin(yaw)
            ly = -(tx - x) * math.sin(yaw) + (ty - y) * math.cos(yaw)
            kappa = 2.0 * ly / max(lx * lx + ly * ly, 1e-6)
            steer = float(np.clip(p.k_steer * math.atan(WHEELBASE * kappa), -0.5, 0.5))
            acts[k] = (acc, steer)
            # predict own motion (same bicycle model as the runtime)
            x += v * math.cos(yaw) * DT
            y += v * math.sin(yaw) * DT
            yaw += v / WHEELBASE * math.tan(steer) * DT
            v = max(0.0, v + acc * DT)
            if math.isfinite(s):
                s -= (v - v_lead) * DT
        return acts


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

PROMPT = "A two-lane suburban road on an overcast afternoon, dashcam view."


def run_episode(policy: Policy, runtime: ToyRuntime, world: WorldModel, n_chunks: int = 15) -> EpisodeMetrics:
    """Policy -> runtime -> world model -> policy, one 8-frame chunk at a time."""
    policy.reset()
    state = runtime.reset()
    first = rasterize(state)  # in Cosmos-Dreams this is a real first RGB frame
    world.reset(first, PROMPT)
    frame = first
    for _ in range(n_chunks):
        actions = policy.act(frame, runtime.ego_speed)
        states = runtime.step(actions)
        if not states:
            break
        frame = world.generate(states)[-1]
        if runtime.done:
            break
    return runtime.metrics


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall tau-b between two score vectors (ties handled)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    conc = disc = ta = tb = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif da == db:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + ta) * (conc + disc + tb))
    return (conc - disc) / denom if denom else float("nan")


@dataclass
class RankingStudy:
    seeds: list[int]
    policies: list[str]
    renderers: list[str]
    rows: list[dict] = field(default_factory=list)

    def mean_scores(self, renderer: str) -> list[float]:
        return [
            float(np.mean([r["score"] for r in self.rows if r["renderer"] == renderer and r["policy"] == p]))
            for p in self.policies
        ]

    def summary(self) -> dict:
        ref = self.mean_scores("exact")
        out = {
            "what": "Toy ranking-preservation study (CPU). Conceptual reproduction of the NuRec vs Cosmos-Dreams "
            "policy-ranking check; no Cosmos model involved.",
            "seeds": self.seeds,
            "chunk_frames": FRAMES_PER_CHUNK,
            "dt_s": DT,
            "episode_chunks": 15,
            "score": "progress_m - 100*collided - 10*lane_departure_s",
            "renderers": {k: asdict(RENDERERS[k]) for k in self.renderers},
            "policies": {p.name: asdict(p) for p in POLICIES if p.name in self.policies},
            "mean_score": {
                r: dict(zip(self.policies, map(_r2, self.mean_scores(r)), strict=True)) for r in self.renderers
            },
            "ranking": {
                r: [p for _, p in sorted(zip(self.mean_scores(r), self.policies, strict=True), reverse=True)]
                for r in self.renderers
            },
            "kendall_tau_vs_exact": {},
            "per_seed_tau_vs_exact_mean": {},
            "collisions": {},
            "lane_departure_s_mean": {},
        }
        for r in self.renderers:
            out["kendall_tau_vs_exact"][r] = _r2(kendall_tau(ref, self.mean_scores(r)))
            taus = []
            for s in self.seeds:
                a = [self._row(r_="exact", p=p, s=s)["score"] for p in self.policies]
                b = [self._row(r_=r, p=p, s=s)["score"] for p in self.policies]
                taus.append(kendall_tau(a, b))
            out["per_seed_tau_vs_exact_mean"][r] = _r2(float(np.nanmean(taus)))
            out["collisions"][r] = {
                p: sum(x["collided"] for x in self.rows if x["renderer"] == r and x["policy"] == p)
                for p in self.policies
            }
            out["lane_departure_s_mean"][r] = {
                p: _r2(np.mean([x["departure_s"] for x in self.rows if x["renderer"] == r and x["policy"] == p]))
                for p in self.policies
            }
        return out

    def _row(self, r_: str, p: str, s: int) -> dict:
        return next(x for x in self.rows if x["renderer"] == r_ and x["policy"] == p and x["seed"] == s)


def _r2(x: float) -> float:
    return round(float(x), 3)


def ranking_study(
    seeds: Sequence[int], renderers: Sequence[str] = tuple(RENDERERS), n_chunks: int = 15
) -> RankingStudy:
    study = RankingStudy(list(seeds), [p.name for p in POLICIES], list(renderers))
    for ri, rname in enumerate(renderers):
        for seed in seeds:
            sc = Scenario.sample(seed)
            for pp in POLICIES:
                world = RasterWorldModel(RENDERERS[rname], seed=1000 * ri + seed)
                m = run_episode(FramePolicy(pp), ToyRuntime(sc), world, n_chunks)
                study.rows.append(
                    {
                        "renderer": rname,
                        "policy": pp.name,
                        "seed": seed,
                        "progress_m": _r2(m.progress_m),
                        "collided": bool(m.collided),
                        "departure_s": _r2(m.departure_s),
                        "min_gap_m": _r2(m.min_gap_m),
                        "score": _r2(m.score),
                    }
                )
    return study


# ---------------------------------------------------------------------------
# Pictures: BEV strips and a WSM-like ego-view render of the abstract state
# ---------------------------------------------------------------------------


def bev_to_rgb(frame: np.ndarray, scale: int = 4) -> np.ndarray:
    img = np.zeros((BEV_H, BEV_W, 3), np.float32)
    img += frame[0][..., None] * np.array([1.0, 1.0, 1.0])
    img += frame[1][..., None] * np.array([0.0, 0.8, 1.0])
    img[EGO_ROW - 4 : EGO_ROW + 5, EGO_COL - 2 : EGO_COL + 2] = (0.2, 1.0, 0.3)  # ego marker
    img = np.clip(img, 0, 1)
    return np.kron((img * 255).astype(np.uint8), np.ones((scale, scale, 1), np.uint8))


def _draw_polyline(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], r: int = 1) -> None:
    h, w = img.shape[:2]
    for p, q in itertools.pairwise(pts):
        n = int(max(abs(q[0] - p[0]), abs(q[1] - p[1]))) + 1
        for t in np.linspace(0.0, 1.0, min(n, 4000)):
            u, v = _iround(p[0] + t * (q[0] - p[0])), _iround(p[1] + t * (q[1] - p[1]))
            if 0 <= u < w and 0 <= v < h:
                img[max(v - r, 0) : v + r + 1, max(u - r, 0) : u + r + 1] = color


def render_wsm_like(state: AbstractState, width: int = 1280, height: int = 720, fov_deg: float = 90.0) -> np.ndarray:
    """Project the abstract state into a forward camera: lane lines and 3D agent cuboids on black.

    Approximates the look of the Cosmos 3 cookbook's ``control_wsm.mp4`` sample (lane lines, road edges, vehicle
    cuboids on black). The Cosmos 2.5 paper describes the world scenario map as HD-map lanes, poles, lights, and
    colour/heading-coded cuboids; the exact rendering spec is not published in the cookbook, so this is not a
    drop-in WSM renderer.
    """
    img = np.zeros((height, width, 3), np.uint8)
    e, cam_h = state.ego, 1.5
    f = (width / 2) / math.tan(math.radians(fov_deg) / 2)
    c, s = math.cos(e.yaw), math.sin(e.yaw)

    def project(xw, yw, zw):
        dx, dy = np.asarray(xw) - e.x, np.asarray(yw) - e.y
        fwd, left = dx * c + dy * s, -dx * s + dy * c
        ok = fwd > 0.5
        u = width / 2 - f * left / np.where(ok, fwd, 1.0)
        v = height / 2 + f * (cam_h - np.asarray(zw)) / np.where(ok, fwd, 1.0)
        return np.stack([u, v], -1), ok

    xs = e.x + np.linspace(1.0, 120.0, 400)
    for off, color in zip(LINE_OFFSETS, ((255, 0, 200), (255, 210, 0), (255, 0, 200)), strict=True):
        pts, ok = project(xs, state.road.center_y(xs) + off, np.zeros_like(xs))
        _draw_polyline(img, pts[ok], color, r=2)
    for b in state.agents:
        base = obb_corners(b)
        bottom, okb = project(base[:, 0], base[:, 1], np.zeros(4))
        top, okt = project(base[:, 0], base[:, 1], np.full(4, 1.6))
        if not (okb.all() and okt.all()):
            continue
        for i in range(4):
            j = (i + 1) % 4
            front = i == 0  # corners 0-1 are the front edge: heading cue
            col = (255, 60, 60) if front else (60, 170, 255)
            _draw_polyline(img, np.array([bottom[i], bottom[j]]), col, r=1)
            _draw_polyline(img, np.array([top[i], top[j]]), col, r=1)
            _draw_polyline(img, np.array([bottom[i], top[i]]), (60, 170, 255), r=1)
    return img


def save_preview(path: Path, seed: int = 0, policy: str = "nominal") -> Path:
    """One image: frames 0, 2, 4, 6 of chunk 6 from each renderer (rows), and a WSM-like ego view below."""
    from PIL import Image, ImageDraw

    sc = Scenario.sample(seed)
    pp = next(p for p in POLICIES if p.name == policy)
    strips = []
    wsm_state = None
    for ri, rname in enumerate(RENDERERS):
        rt, world, pol = ToyRuntime(sc), RasterWorldModel(RENDERERS[rname], seed=1000 * ri + seed), FramePolicy(pp)
        state = rt.reset()
        world.reset(rasterize(state), PROMPT)
        frame, frames = rasterize(state), None
        for _ in range(6):
            states = rt.step(pol.act(frame, rt.ego_speed))
            frames = world.generate(states)
            frame = frames[-1]
        strips.append(np.concatenate([bev_to_rgb(fr) for fr in frames[::2]], axis=1))
        if rname == "exact":
            wsm_state = states[-1]
    label_w, pad = 110, 4
    grid = np.concatenate([np.pad(s, ((pad, pad), (pad, pad), (0, 0)), constant_values=40) for s in strips], axis=0)
    width = label_w + grid.shape[1]
    wsm = render_wsm_like(wsm_state, width - 2 * pad, (width - 2 * pad) * 9 // 16)
    canvas = np.full((grid.shape[0] + wsm.shape[0] + 3 * pad + 20, width, 3), 40, np.uint8)
    canvas[: grid.shape[0], label_w:] = grid
    canvas[grid.shape[0] + 20 + pad : grid.shape[0] + 20 + pad + wsm.shape[0], pad : pad + wsm.shape[1]] = wsm
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    row_h = grid.shape[0] // len(strips)
    for i, rname in enumerate(RENDERERS):
        draw.text((8, i * row_h + row_h // 2 - 6), rname, fill=(230, 230, 230))
    draw.text(
        (8, grid.shape[0] + 4),
        "Same abstract state (exact loop) projected to a forward camera: WSM-like control render (toy)",
        fill=(230, 230, 230),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def export_wsm(out_dir: Path, seed: int = 0, policy: str = "nominal", frames: int = 101, encode: bool = False) -> Path:
    """Drive the exact loop and write ``frames`` WSM-like PNGs (10 fps, matching the cookbook wsm spec)."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    sc = Scenario.sample(seed)
    pp = next(p for p in POLICIES if p.name == policy)
    rt, world, pol = ToyRuntime(sc), RasterWorldModel(EXACT, seed=seed), FramePolicy(pp)
    state = rt.reset()
    world.reset(rasterize(state), PROMPT)
    seq, frame = [state], rasterize(state)
    while len(seq) < frames and not rt.done:
        states = rt.step(pol.act(frame, rt.ego_speed))
        if not states:
            break
        frame = world.generate(states)[-1]
        seq.extend(states)
    for i, st in enumerate(seq[:frames]):
        Image.fromarray(render_wsm_like(st)).save(out_dir / f"frame_{i:04d}.png")
    cmd = ["ffmpeg", "-y", "-framerate", "10", "-i", str(out_dir / "frame_%04d.png"), "-c:v", "libx264"]
    cmd += ["-pix_fmt", "yuv420p", str(out_dir / "control_wsm.mp4")]
    print("[dreams_loop] encode with:", " ".join(cmd))
    if encode and shutil.which("ffmpeg"):
        subprocess.run(cmd, check=True, capture_output=True)
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank", help="ranking-preservation study")
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--out", type=Path, default=Path("results"))
    w = sub.add_parser("export-wsm", help="WSM-like control frames from the toy loop")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--policy", default="nominal", choices=[p.name for p in POLICIES])
    w.add_argument("--frames", type=int, default=101)
    w.add_argument("--encode", action="store_true", help="run ffmpeg to write control_wsm.mp4")
    args = ap.parse_args(argv)

    if args.cmd == "export-wsm":
        export_wsm(args.out, args.seed, args.policy, args.frames, args.encode)
        return 0
    study = ranking_study(range(args.seeds))
    summ = study.summary()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dreams_ranking.json").write_text(json.dumps(summ, indent=2) + "\n")
    with (args.out / "dreams_ranking_episodes.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(study.rows[0]))
        wr.writeheader()
        wr.writerows(study.rows)
    save_preview(args.out / "dreams_preview.png")
    print(json.dumps({k: summ[k] for k in ("ranking", "kendall_tau_vs_exact", "per_seed_tau_vs_exact_mean")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
