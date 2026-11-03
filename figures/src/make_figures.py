"""Regenerate every blog figure.

    python -m venv .venv && .venv/bin/pip install matplotlib numpy
    .venv/bin/python physical-ai-project/figures/src/make_figures.py          # all
    .venv/bin/python physical-ai-project/figures/src/make_figures.py stack    # one

Outputs land in physical-ai-project/figures/ (PNG at 2x, SVG for diagrams) and the
hero placeholder in physical-ai-project/PhysicalAI.png. Facts and numbers come from
physical-ai-project/context/ (see figures/README.md for per-figure sources).
"""

from __future__ import annotations

import sys


import math

from figkit import (  # noqa: F401
    BASELINE, DATA, GRID, HAIR, INK, INK2, MONO, MUTED, ORD, PANEL, PHYS, PT2PX, REAL, SURFACE, WORLD,
    accent_bar, arrow, box, canvas, chip, dot, flow_chips, footer, legend_row, measure, para,
    para_height, save, text, tint, title_block, wrap,
)

W = 1200  # blog column width at 1x


# =============================================================================== 1. stack
def fig_stack():
    H = 840
    fig, ax = canvas(W, H)
    title_block(
        ax,
        40,
        34,
        "The Physical AI stack, layer by layer",
        "Each layer reads and writes the one below it. The three computers on the right are where the work runs.",
    )

    X0, X1 = 40, 860  # stack column
    LABEL_W = 170
    CX = X0 + LABEL_W + 14
    CW = X1 - CX - 16
    star = dict(weight="semibold", ec=INK, lw=1.3)

    layers = [
        (
            "Agents & skills",
            "the new interface",
            [
                "Claude Code / Codex",
                "NVIDIA/skills",
                "content & physics agents",
                "NuRec agent skills on Brev",
            ],
        ),
        (
            "Learned worlds",
            "Cosmos & NuRec",
            [
                dict(text="Cosmos 3  ·  Reasoner | Generator", **star),
                "Cosmos Predict / Transfer / Reason (2025)",
                "NuRec 3D Gaussian reconstruction",
                "Cosmos Dreams (real-time driving)",
            ],
        ),
        (
            "Robotics apps",
            "Isaac & GR00T",
            [
                "Isaac Sim  (ROS 2 bridge, sensors)",
                "Isaac Lab  (RL / IL, 1k–16k envs)",
                "Isaac Lab Arena",
                "Isaac ROS",
                "GR00T N1 / N1.5",
                "MobilityGen · X-Mobility · COMPASS",
            ],
        ),
        (
            "Physics",
            "PhysX & Newton",
            [
                dict(text="Newton  (robot learning)", **star),
                "→",
                "MuJoCo Warp",
                "→",
                "Warp",
                "→",
                "CUDA",
                "PhysX  (industrial digital twins)",
            ],
        ),
        (
            "Platform",
            "Omniverse",
            [
                "Omniverse Kit",
                "→",
                "OV Stage",
                "OVRTX",
                "OV Physics",
                "OV Stream",
                "OVUI",
                "Sensor RTX",
                "Replicator",
                "Nucleus",
            ],
        ),
        (
            "Data model",
            "OpenUSD & SimReady",
            [
                "OpenUSD",
                "AOUSD schemas  (UsdPhysics, deformables)",
                "SimReady content specs",
            ],
        ),
    ]

    computers = [
        (
            "Train",
            DATA,
            "DGX · GB200",
            "“compresses tokens”",
            "Trains GR00T, Cosmos, and RL / IL policies.",
        ),
        (
            "Simulate",
            PHYS,
            "OVX · RTX PRO 6000 · DGX Spark",
            "“generates tokens”",
            "Runs Omniverse, Isaac, Newton, and Cosmos to make data.",
        ),
        (
            "Deploy",
            REAL,
            "Jetson Thor · Drive AGX",
            "“deploys tokens”",
            "Runs the distilled policy. Cosmos 3 Edge (4B) targets Jetson Thor.",
        ),
    ]

    y = 120
    gap = 10
    chip_size = 10
    for name, sub, items in layers:
        ch = flow_chips(ax, CX, 0, CW, items, size=chip_size, draw=False)
        h = max(ch + 28, 64)
        box(ax, X0, y, X1 - X0, h, fc=PANEL, ec="none", r=12)
        text(
            ax,
            X0 + 18,
            y + h / 2 - 9,
            name,
            size=12,
            weight="bold",
            va="center_baseline",
        )
        text(
            ax, X0 + 18, y + h / 2 + 10, sub, size=9.5, color=INK2, va="center_baseline"
        )
        flow_chips(ax, CX, y + (h - ch) / 2, CW, items, size=chip_size)
        y += h + gap

    # hardware layer: one cell per computer
    hh = 70
    box(ax, X0, y, X1 - X0, hh, fc=PANEL, ec="none", r=12)
    text(
        ax,
        X0 + 18,
        y + hh / 2 - 9,
        "Hardware",
        size=12,
        weight="bold",
        va="center_baseline",
    )
    text(
        ax,
        X0 + 18,
        y + hh / 2 + 10,
        "three computers",
        size=9.5,
        color=INK2,
        va="center_baseline",
    )
    cw = (CW - 2 * 8) / 3
    for i, (nm, col, hw, _, _) in enumerate(computers):
        cx = CX + i * (cw + 8)
        box(ax, cx, y + 12, cw, hh - 24, fc=tint(col, 0.14), ec="none", r=10)
        dot(ax, cx + 16, y + hh / 2, 5, col, ring=tint(col, 0.14))
        text(
            ax,
            cx + 28,
            y + hh / 2 - 8,
            nm,
            size=10,
            weight="semibold",
            va="center_baseline",
        )
        text(
            ax, cx + 28, y + hh / 2 + 9, hw.replace(" · DGX Spark", ""), size=8.8, color=INK2, va="center_baseline"
        )
    stack_bottom = y + hh

    # emphasis key
    ky = stack_bottom + 22
    chip(ax, X0, ky - 11, "Newton", size=9, weight="semibold", ec=INK, lw=1.3)
    text(
        ax,
        X0 + chip_width_("Newton") + 8,
        ky,
        "outlined = where this series digs in (Newton and Cosmos 3)",
        size=9,
        color=INK2,
        va="center_baseline",
    )

    # three computers column
    RX, RW = 890, 270
    text(ax, RX, 120, "Three computers", size=13, weight="bold")
    para(
        ax,
        RX,
        146,
        "NVIDIA’s framing: every Physical AI system needs all three.",
        RW,
        size=9.5,
        color=INK2,
    )
    cy = 196
    ch_ = (stack_bottom - cy - 2 * 14) / 3
    for nm, col, hw, quote, desc in computers:
        box(ax, RX, cy, RW, ch_, fc=tint(col, 0.10), ec="none", r=12)
        accent_bar(ax, RX, cy + 14, ch_ - 28, col, w=5)
        text(ax, RX + 20, cy + 18, nm, size=14, weight="bold")
        text(ax, RX + 20, cy + 44, hw, size=9.5, weight="semibold", color=INK2)
        text(ax, RX + 20, cy + 64, quote, size=9.5, color=INK2)
        para(ax, RX + 20, cy + 92, desc, RW - 36, size=10, color=INK)
        cy += ch_ + 14

    footer(
        ax,
        40,
        H - 34,
        "Sources: NVIDIA GTC 2025 Cosmos talks; NVIDIA Physical AI Day, SIGGRAPH 2026; Newton and Cosmos repositories. "
        "Simplified: not every product is shown.",
    )
    return save(fig, "stack")



# ===================================================================== 2. three generations
def fig_generations():
    H = 650
    fig, ax = canvas(W, H)
    title_block(
        ax, 40, 34, "Three generations of simulation",
        "Positioned as complementary, not successive: each covers what the others cannot.",
    )
    cols = [
        ("Simulation 1.0", "Graphics"),
        ("Simulation 2.0", "Neural reconstruction"),
        ("Simulation 3.0", "Generative world models"),
    ]
    rows = [
        ("Method", [
            "Hand-authored 3D assets, physics, and ray tracing (Omniverse, Isaac Sim).",
            "Rebuild real places from sensor logs: NeRF, now 3D Gaussian splats (NuRec).",
            "A learned model generates the world as video (Cosmos).",
        ]),
        ("Strength", [
            "Deterministic, exact ground truth; occlusion, LiDAR multipath, radar, hardware-in-the-loop.",
            "Scales content from fleet data; real places, smaller sim-to-real gap.",
            "Scales beyond fleet data: weather, lens effects, the long tail.",
        ]),
        ("Weakness", [
            "Bottlenecked on asset creation.",
            "Quality degrades as the trajectory deviates from the recorded one.",
            "Hallucination; slow unless distilled.",
        ]),
        ("Evidence", [
            "Mcity hand-built twin: about $150k and 6 months for 30 acres.",
            "Mcity NuRec rebuild: about $2,310 and 2 days.",
            "Cosmos Dreams keeps policy stack-ranking vs. NuRec at 68 fps (GB300).",
        ]),
    ]
    X0, LW = 40, 130
    CW = (W - 80 - LW - 3 * 12) / 3
    top = 120
    head_h = 78
    for i, (t, sub) in enumerate(cols):
        x = X0 + LW + 12 + i * (CW + 12)
        box(ax, x, top, CW, head_h, fc=tint(ORD[i], 0.16), ec="none", r=12)
        box(ax, x + 16, top + 14, 36, 6, fc=ORD[i], ec="none", r=3)
        text(ax, x + 16, top + 30, t, size=14, weight="bold")
        text(ax, x + 16, top + 54, sub, size=10.5, color=INK2)
    y = top + head_h + 10
    pad = 16
    for r_i, (rname, cells) in enumerate(rows):
        hts = [para_height(c, CW - 2 * pad, 10.5) for c in cells]
        h = max(hts) + 2 * pad - 4
        box(ax, X0, y, W - 80, h, fc="#f6f5f1" if r_i % 2 == 0 else SURFACE, ec="none", r=10)
        text(ax, X0 + 16, y + pad, rname, size=11, weight="bold")
        for i, c in enumerate(cells):
            x = X0 + LW + 12 + i * (CW + 12)
            weight = "semibold" if rname == "Evidence" else "normal"
            para(ax, x + pad, y + pad, c, CW - 2 * pad, size=10.5, weight=weight)
        y += h + 6
    y += 10
    para(ax, X0, y,
         "“The sweet spot is somewhere in the middle and will keep shifting.” "
         "Hand-built for exactness, reconstructed for real places, generated for the long tail.",
         W - 80, size=10.5, color=INK2)
    footer(ax, 40, H - 34,
           "Sources: Mcity slide and talk, NVIDIA Physical AI Day, SIGGRAPH 2026; Cosmos Dreams talk, SIGGRAPH 2026; "
           "NVIDIA AV simulation livestream (auto-captions). Costs are as presented by Mcity.")
    return save(fig, "three-generations", svg=False)


# ===================================================================== 3. newton step loop
def code(ax, x, y, s, size=9.5, color=INK, **kw):
    return text(ax, x, y, s, size=size, color=color, family=MONO, **kw)


def fig_newton_loop():
    H = 690
    fig, ax = canvas(W, H)
    title_block(
        ax, 40, 34, "Newton’s step loop",
        "Five explicit classes: Model, State, Control, Contacts, Solver. Build once, then step; the solver is a plug-in.",
    )
    # ---- build column
    BX, BW = 40, 228
    text(ax, BX, 116, "BUILD ONCE", size=9, weight="bold", color=MUTED)
    y = 140
    box(ax, BX, y, BW, 124, fc=tint(PHYS, 0.08), ec="none", r=12)
    text(ax, BX + 16, y + 16, "ModelBuilder", size=12.5, weight="bold")
    for i, s in enumerate(["add_urdf() / add_mjcf()", "add_usd()", "replicate(template,", "  world_count=N)"]):
        code(ax, BX + 16, y + 46 + i * 18, s, size=8.6, color=INK2)
    y2 = y + 124 + 30
    box(ax, BX + 54, y2, BW - 108, 32, fc=SURFACE, ec=INK2, r=16, lw=1.2)
    code(ax, BX + BW / 2, y2 + 16, "finalize()", size=9.5, ha="center", va="center_baseline")
    arrow(ax, (BX + BW / 2, y + 124), (BX + BW / 2, y2), color=INK2)
    y3 = y2 + 32 + 30
    box(ax, BX, y3, BW, 112, fc=tint(PHYS, 0.08), ec="none", r=12)
    text(ax, BX + 16, y3 + 16, "Model", size=12.5, weight="bold")
    para(ax, BX + 16, y3 + 44, "Time-invariant: bodies, joints, shapes, masses. One model, N worlds inside.",
         BW - 32, size=9.3, color=INK2)
    arrow(ax, (BX + BW / 2, y2 + 32), (BX + BW / 2, y3), color=INK2)
    y4 = y3 + 112 + 30
    box(ax, BX, y4, BW, 108, fc=SURFACE, ec=HAIR, r=12)
    text(ax, BX + 16, y4 + 14, "Allocate", size=11, weight="bold")
    for i, s in enumerate(["state_0 = model.state()", "state_1 = model.state()", "control = model.control()",
                           "contacts = pipe.contacts()"]):
        code(ax, BX + 16, y4 + 40 + i * 16, s, size=8.2, color=INK2)
    arrow(ax, (BX + BW / 2, y3 + 112), (BX + BW / 2, y4), color=INK2)

    # ---- loop
    LX0, LX1 = 310, 830
    text(ax, LX0, 116, "STEP, EVERY SUBSTEP", size=9, weight="bold", color=MUTED)
    nw, nh = 196, 92
    TY, BY = 150, 450
    pos = {"state": (LX0, TY), "collide": (LX1 - nw, TY), "step": (LX1 - nw, BY), "swap": (LX0, BY)}
    content = {
        "state": ("State · Control", [("t", "state: poses, velocities"), ("t", "control: joint targets")]),
        "collide": ("CollisionPipeline", [("c", "collide(state_0,"), ("c", "        contacts)"), ("t", "broad → narrow → reduce")]),
        "step": ("solver.step()", [("c", "(state_0, state_1,"), ("c", " control, contacts,"), ("c", " dt)")]),
        "swap": ("Swap states", [("c", "state_0, state_1 ="), ("c", "  state_1, state_0")]),
    }
    for k, (x, y_) in pos.items():
        ttl, lines = content[k]
        is_step = k == "step"
        box(ax, x, y_, nw, nh, fc=tint(PHYS, 0.20 if is_step else 0.08), ec=PHYS if is_step else "none",
            r=12, lw=1.6 if is_step else 0, z=3)
        text(ax, x + 14, y_ + 13, ttl, size=11.5, weight="bold", family=MONO if is_step else None)
        for i, (kind, s) in enumerate(lines):
            if kind == "t":
                text(ax, x + 14, y_ + 40 + i * 16, s, size=8.8, color=INK2)
            else:
                code(ax, x + 14, y_ + 40 + i * 16, s, size=8.2, color=INK2)
    a_kw = dict(color=INK2, lw=1.8, ms=15)
    arrow(ax, (LX0 + nw, TY + nh / 2), (LX1 - nw, TY + nh / 2), **a_kw)
    arrow(ax, (LX1 - nw / 2, TY + nh), (LX1 - nw / 2, BY), **a_kw)
    arrow(ax, (LX1 - nw, BY + nh / 2), (LX0 + nw, BY + nh / 2), **a_kw)
    arrow(ax, (LX0 + nw / 2, BY), (LX0 + nw / 2, TY + nh), **a_kw)
    mid = (LX0 + LX1) / 2
    code(ax, mid, TY + nh / 2 - 9, "state_0", size=8.6, color=INK2, ha="center", va="bottom")
    chip(ax, LX1 - nw / 2 + 12, (TY + nh + BY) / 2 - 14, "Contacts", size=9.5, weight="semibold")
    code(ax, mid, BY + nh / 2 - 9, "state_1", size=8.6, color=INK2, ha="center", va="bottom")
    text(ax, mid, BY + nh / 2 + 9, "(t + dt)", size=8.6, color=INK2, ha="center", va="top")
    text(ax, LX0 + nw / 2 + 12, (TY + nh + BY) / 2, "next substep", size=9, color=INK2, va="center_baseline")
    para(ax, mid - 70, TY + nh + 40, "The whole loop is captured once as a CUDA graph and replayed.", 170, size=9, color=MUTED)
    # allocate feeds the first state
    arrow(ax, (BX + BW, y4 + 24), (LX0, TY + nh - 16), color=MUTED, rad=0, lw=1.2, ms=11)

    para(ax, LX0, BY + nh + 34,
         "Conventions that bite: Z-up by default · quaternions are XYZW (MuJoCo and Isaac use WXYZ) · "
         "world −1 holds shared entities such as the ground plane.",
         LX1 - LX0, size=9.5, color=INK2)

    # ---- solver zoo
    ZX, ZW = 880, 280
    text(ax, ZX, 116, "PLUG-IN SOLVERS  (any SolverBase)", size=9, weight="bold", color=MUTED)
    solvers = [
        ("SolverMuJoCo", "Articulated robots at scale (default)"),
        ("SolverXPBD", "Fast position-based rigid bodies"),
        ("SolverVBD", "Cloth, soft bodies, cables (experimental)"),
        ("SolverFeatherstone · SemiImplicit", "Differentiable simulation via wp.Tape"),
        ("SolverImplicitMPM", "Sand, snow, mud, fluids"),
        ("SolverKamino", "Closed loops, hard contact (beta)"),
        ("solvers.experimental.coupled", "Mix solvers, e.g. robot + MPM terrain"),
    ]
    zy = 140
    zh = 60
    zg = 8
    for name, desc in solvers:
        box(ax, ZX, zy, ZW, zh, fc=SURFACE, ec=HAIR, r=10, z=3)
        accent_bar(ax, ZX + 8, zy + 12, zh - 24, PHYS, w=4, z=4)
        code(ax, ZX + 22, zy + 11, name, size=fit_size(name, ZW - 34, 9, "bold", family=MONO), weight="bold")
        text(ax, ZX + 22, zy + 34, desc, size=fit_size(desc, ZW - 34, 9.2), color=INK2)
        zy += zh + zg
    rail = ZX - 16
    ax.plot([rail, rail], [150, zy - zg - 10], color=PHYS, lw=2, solid_capstyle="round", zorder=2)
    arrow(ax, (rail, BY + nh / 2), (LX1, BY + nh / 2), color=PHYS, lw=2, ms=15, shrinkA=0)

    footer(ax, 40, H - 34,
           "Source: Newton repository (newton/examples/basic, newton/_src/solvers) and docs, v1.7.0.dev0 snapshot; "
           "Newton talk, NVIDIA Physical AI Day, SIGGRAPH 2026. Maturity labels as stated in the repo.")
    return save(fig, "newton-loop")


# ===================================================================== 4. newton memory (public ranges only)
def fig_newton_memory():
    H = 560
    fig, ax = canvas(W, H)
    title_block(
        ax, 40, 34, "Newton vs. Isaac Lab 2 on PhysX: what the public talk states",
        "Ranges across Isaac Lab RL tasks (MuJoCo solver) and a tiled-camera benchmark. "
        "Higher is better; 1× is the PhysX baseline.",
    )
    PX0, PX1 = 260, 800
    top = 150
    rows = [
        ("RL physics", "Steps per second", 2, 3, "2–3× faster"),
        ("RL physics", "GPU memory", 3, 4, "3–4× less"),
        ("Tiled camera", "Render speed", 2, 3, "2–3× faster"),
        ("Tiled camera", "GPU memory", 5, 10, "5–10× less"),
    ]
    xmax = 11

    def sx(v):
        return PX0 + (PX1 - PX0) * v / xmax

    row_h = 66
    ybot = top + len(rows) * row_h + 14
    for v in [0, 2, 4, 6, 8, 10]:
        ax.plot([sx(v), sx(v)], [top - 8, ybot], color=GRID, lw=1, zorder=1)
        text(ax, sx(v), ybot + 8, f"{v}×", size=9, color=MUTED, ha="center")
    ax.plot([sx(1), sx(1)], [top - 8, ybot], color=BASELINE, lw=1.4, zorder=1)
    text(ax, sx(1), top - 14, "PhysX = 1×", size=9, color=INK2, ha="center", va="bottom")
    text(ax, (PX0 + PX1) / 2, ybot + 30, "Multiple of the Isaac Lab 2 + PhysX baseline", size=9.5, color=INK2, ha="center")
    last_group = None
    for i, (grp, lab, lo, hi, val) in enumerate(rows):
        cy = top + i * row_h + row_h / 2
        if grp != last_group:
            text(ax, 40, cy - 22, grp.upper(), size=8.5, weight="bold", color=MUTED)
            last_group = grp
        text(ax, 40, cy, lab, size=11, weight="semibold", va="center_baseline")
        bh = 20
        ax.plot([sx(1), sx(lo)], [cy, cy], color=tint(PHYS, 0.35), lw=2, solid_capstyle="round", zorder=2)
        box(ax, sx(lo), cy - bh / 2, sx(hi) - sx(lo), bh, fc=PHYS, ec="none", r=4, z=3)
        text(ax, sx(hi) + 10, cy, val, size=10.5, weight="semibold", va="center_baseline")

    RX0, RX1 = 900, 1160
    text(ax, RX0 - 20, 120, "Environments on the same GPU", size=12, weight="bold")
    text(ax, RX0 - 20, 142, "as stated: “previously 4,000, now 16,000”", size=9.5, color=INK2)
    cb, ct = ybot, 200

    def sy(v):
        return cb - (cb - ct) * v / 16000

    for v in [0, 4000, 8000, 12000, 16000]:
        ax.plot([RX0, RX1], [sy(v), sy(v)], color=GRID if v else BASELINE, lw=1, zorder=1)
        text(ax, RX0 - 8, sy(v), f"{v:,}", size=9, color=MUTED, ha="right", va="center_baseline")
    bars = [("PhysX", 4000, "#a9a79f"), ("Newton", 16000, PHYS)]
    bw = 48
    for j, (lab, v, col) in enumerate(bars):
        cx = RX0 + (RX1 - RX0) * (0.3 + 0.42 * j)
        box(ax, cx - bw / 2, sy(v), bw, cb - sy(v), fc=col, ec="none", r=4, z=3)
        box(ax, cx - bw / 2, cb - 6, bw, 6, fc=col, ec="none", r=0, z=3)
        text(ax, cx, sy(v) - 8, f"{v:,}", size=10.5, weight="semibold", ha="center", va="bottom")
        text(ax, cx, cb + 8, lab, size=10, color=INK2, ha="center")

    footer(ax, 40, H - 52,
           "Source: Newton talk, NVIDIA Physical AI Day, SIGGRAPH 2026 (public recording). Bars span the ranges the "
           "speaker states; per-task measurements are not reproduced here. Tiled-camera figures credit Flexion "
           "Robotics and compare against Isaac Lab 2.")
    return save(fig, "newton-memory", svg=False)


# ===================================================================== 5. flywheel
def plt_circle(x, y, r, fc, ec):
    from matplotlib.patches import Circle

    return Circle((x, y), r, fc=fc, ec=ec, lw=2, zorder=6)


def para_z(ax, x, y, s, max_px, size=10, color=INK, lh=1.35):
    lines = wrap(s, max_px, size)
    for i, ln in enumerate(lines):
        text(ax, x, y + i * size * PT2PX * lh, ln, size=size, color=color, z=7)


def para_center(ax, x, y, s, max_px, size=10, color=INK, lh=1.4, weight="normal"):
    lines = wrap(s, max_px, size, weight)
    for i, ln in enumerate(lines):
        text(ax, x, y + i * size * PT2PX * lh, ln, size=size, color=color, ha="center", weight=weight)
    return y + len(lines) * size * PT2PX * lh


def fig_flywheel():
    H = 1040
    fig, ax = canvas(W, H)
    title_block(
        ax, 40, 34, "The sim-to-real data flywheel",
        "Ten steps, assembled from the GTC 2025 and SIGGRAPH 2026 talks. Each lap turns a little real data into a lot of training data.",
    )
    groups = {"real": (REAL, "Real world"), "phys": (PHYS, "Physics simulation"),
              "world": (WORLD, "Cosmos world models"), "data": (DATA, "Data & training")}
    legend_row(ax, 40, 122, [groups[g] for g in ("real", "phys", "world", "data")])
    steps = [
        ("Seed", "Teleop demos, fleet drives, archived video", "real"),
        ("Reconstruct / compose", "NuRec splats · Isaac Sim · CAD-to-SimReady", "phys"),
        ("Multiply trajectories", "Isaac Lab RL · DexMimicGen · MobilityGen", "phys"),
        ("Multiply appearance", "Cosmos Transfer: seg / depth / edges → photoreal", "world"),
        ("Dream", "Cosmos Predict / Cosmos 3 + inverse dynamics", "world"),
        ("Filter", "Cosmos Reason grades plausibility", "world"),
        ("Curate", "NeMo Curator · Cosmos video search", "data"),
        ("Train", "Co-train real + synthetic (GR00T, policies)", "data"),
        ("Evaluate", "Isaac Lab Arena · RoboFinals · NuRec scenes, then real", "phys"),
        ("Deploy & feed back", "Jetson Thor; failures become the next seed", "real"),
    ]
    # a rounded-rectangle ring: 2 on top, 3 down the right, 2 on the bottom, 3 up the left
    cw, ch = 252, 92
    xl, xm1, xm2, xr = 40 + cw / 2, 440, 760, W - 40 - cw / 2
    yt, ys1, ys2, ys3, yb = 220, 330, 560, 790, 900
    centers = [(xm1, yt), (xm2, yt), (xr, ys1), (xr, ys2), (xr, ys3),
               (xm2, yb), (xm1, yb), (xl, ys3), (xl, ys2), (xl, ys1)]
    patches = []
    for i, ((ttl, tool, g), (mx, my)) in enumerate(zip(steps, centers)):
        x, y = mx - cw / 2, my - ch / 2
        col = groups[g][0]
        p = box(ax, x, y, cw, ch, fc=tint(col, 0.10), ec="none", r=14, z=3)
        patches.append(p)
        accent_bar(ax, x + 8, y + 14, ch - 28, col, w=4, z=4)
        bx_, by_ = x + 34, y + 26
        ax.add_patch(plt_circle(bx_, by_, 13, SURFACE, col))
        text(ax, bx_, by_, str(i + 1), size=10, weight="bold", ha="center", va="center_baseline", z=7)
        text(ax, x + 56, y + 26, ttl, size=fit_size(ttl, cw - 66, 11.5, "bold"), weight="bold",
             va="center_baseline", z=7)
        para_z(ax, x + 22, y + 48, tool, cw - 36, size=9.3, color=INK2)
    for i in range(len(steps)):
        a, b = centers[i], centers[(i + 1) % len(steps)]
        arrow(ax, a, b, patchA=patches[i], patchB=patches[(i + 1) % len(steps)], color=INK2,
              rad=-0.12, lw=1.6, ms=13, shrinkA=5, shrinkB=5, z=2)
    cx, cy = W / 2, 540
    text(ax, cx, cy - 96, "Data you compute,", size=22, weight="bold", ha="center")
    text(ax, cx, cy - 62, "not collect", size=22, weight="bold", ha="center")
    para_center(ax, cx, cy - 6,
                "Lesson from X-Mobility: training on Isaac Sim + Cosmos Transfer data made no visible difference "
                "in the simulator, yet the real robot went from hitting a transparent box every time to avoiding it "
                "every time. Evaluate somewhere other than where you generated.",
                420, size=10, color=INK2)
    footer(ax, 40, H - 52,
           "Sources: NVIDIA GTC 2025 Cosmos and GR00T talks; NVIDIA Physical AI Day, SIGGRAPH 2026; X-Mobility result as "
           "reported in an NVIDIA livestream (auto-captions). Tools per step are examples, not an exhaustive list.")
    return save(fig, "flywheel")


# ===================================================================== 6. project architecture
def bullets(ax, x, y, items, max_px, size=10, color=INK, gap=5):
    from matplotlib.patches import Circle

    for it in items:
        ax.add_patch(Circle((x + 3, y + size * PT2PX * 0.62), 2.5, fc=MUTED, ec="none", zorder=5))
        y = para(ax, x + 14, y, it, max_px - 14, size=size, color=color) + gap
    return y


def fig_architecture():
    H = 750
    fig, ax = canvas(W, H)
    title_block(
        ax, 40, 34, "The project: can a world model grade a physics engine?",
        "Three phases. Physics produces rollouts with exact ground truth; Cosmos 3 judges and re-renders them; "
        "we score the judge against the ground truth.",
    )
    phases = [
        (40, 330, PHYS, "PHASE 0", "Solver bake-off", "Mac laptop · CPU only", [
            "newton[examples] on macOS",
            "One quadruped URDF, four solvers, 1–4 worlds",
            "Headless (--viewer null) or GL viewer",
            "Metrics + ViewerFile recordings",
        ]),
        (430, 350, PHYS, "PHASE 1", "Quadruped on granular terrain", "Cloud · 1× 24–48 GB GPU", [
            "ANYmal C URDF + ONNX locomotion policy",
            "SolverMuJoCo (robot) coupled with SolverImplicitMPM (terrain)",
            "Material sweep: sand · snow · clay",
            "N worlds via replicate()",
            "SensorTiledCamera: RGB / depth / segmentation",
            "ViewerUSD export; deterministic mode",
        ]),
        (840, 320, WORLD, "PHASE 2", "Cosmos 3 as judge and renderer", "Cloud · 24 GB judge, 80 GB transfer", None),
    ]
    top = 142
    ph = 392
    exits = {
        "PHASE 0": "Exit: all four solvers run the same asset on CPU, numbers explained.",
        "PHASE 1": "Exit: a reproducible sweep and a “terrain hurts this policy by X” statement with error bars.",
    }
    for x, w, col, tag, ttl, hw, items in phases:
        box(ax, x, top, w, ph, fc=tint(col, 0.07), ec="none", r=14)
        box(ax, x, top, w, 6, fc=col, ec="none", r=3)
        text(ax, x + 20, top + 22, tag, size=9, weight="bold", color=INK2)
        text(ax, x + 20, top + 42, ttl, size=13.5, weight="bold")
        text(ax, x + 20, top + 68, hw, size=9.8, color=INK2)
        if items:
            bullets(ax, x + 20, top + 104, items, w - 40, size=10)
        if tag in exits:
            eh = para_height(exits[tag], w - 64, 9.5) + 22
            box(ax, x + 16, top + ph - eh - 16, w - 32, eh, fc=SURFACE, ec=HAIR, r=10)
            para(ax, x + 32, top + ph - eh - 5, exits[tag], w - 64, size=9.5, color=INK2)
    x, w = 840, 320
    sb_y = top + 100
    box(ax, x + 16, sb_y, w - 32, 132, fc=SURFACE, ec=HAIR, r=10)
    accent_bar(ax, x + 24, sb_y + 14, 104, DATA, w=4)
    text(ax, x + 38, sb_y + 14, "Cosmos3-Nano Reasoner", size=11, weight="bold")
    text(ax, x + 38, sb_y + 34, "served with vLLM · the judge", size=9.3, color=INK2)
    bullets(ax, x + 38, sb_y + 62, ["Task success: A / B / C", "Physical plausibility: A / B", "Temporal segments"],
            w - 70, size=9.8, gap=3)
    sb2 = sb_y + 132 + 12
    box(ax, x + 16, sb2, w - 32, 132, fc=SURFACE, ec=HAIR, r=10)
    accent_bar(ax, x + 24, sb2 + 14, 104, WORLD, w=4)
    text(ax, x + 38, sb2 + 14, "Cosmos3-Nano Generator", size=11, weight="bold")
    text(ax, x + 38, sb2 + 34, "transfer mode · the renderer", size=9.3, color=INK2)
    bullets(ax, x + 38, sb2 + 62, ["Depth + segmentation hints → photoreal variants", "Scored with PAI-Bench-C metrics"],
            w - 70, size=9.8, gap=3)
    ay = top + 190
    arrow(ax, (370, ay), (430, ay), color=INK2, lw=1.8, ms=15)
    text(ax, 400, ay - 10, "verify", size=9.5, weight="semibold", color=INK2, ha="center", va="bottom")
    arrow(ax, (780, ay), (840, ay), color=INK2, lw=1.8, ms=15)
    text(ax, 810, ay - 26, "renders", size=9.5, weight="semibold", color=INK2, ha="center", va="bottom")
    text(ax, 810, ay - 10, "+ GT logs", size=9.5, weight="semibold", color=INK2, ha="center", va="bottom")

    gy = top + ph + 34
    box(ax, 430, gy, 350, 104, fc=SURFACE, ec=INK2, r=12, lw=1.2)
    text(ax, 450, gy + 16, "Ground truth from the simulator", size=11, weight="bold")
    para(ax, 450, gy + 42, "Fall flag · distance travelled · sinkage · contact forces · per-frame poses", 310, size=9.8, color=INK2)
    box(ax, 840, gy, 320, 104, fc=tint(DATA, 0.10), ec="none", r=12)
    text(ax, 860, gy + 16, "Score the judge", size=11, weight="bold")
    para(ax, 860, gy + 42, "Reasoner verdicts vs. ground truth → precision · recall · agreement", 280, size=9.8, color=INK)
    arrow(ax, (605, top + ph), (605, gy), color=INK2, lw=1.6)
    arrow(ax, (780, gy + 52), (840, gy + 52), color=INK2, lw=1.8, ms=15)
    arrow(ax, (1000, top + ph), (1000, gy), color=INK2, lw=1.6)
    text(ax, 1008, top + ph + 16, "verdicts", size=9, color=INK2, va="center_baseline")
    text(ax, 40, gy + 16, "Status", size=11, weight="bold")
    para(ax, 40, gy + 42, "Planned. No results yet: Parts 2, 3, and 4 of the series report each phase.", 330,
         size=9.8, color=INK2)
    footer(ax, 40, H - 34, "Source: planning/02-project-plan.md, section 4. Hardware sizes are targets, not measurements.")
    return save(fig, "project-architecture")


# ===================================================================== 7. cosmos 3 modes
def fig_cosmos3():
    H = 800
    fig, ax = canvas(W, H)
    title_block(
        ax, 40, 34, "Cosmos 3: one model, two towers, many jobs",
        "An omnimodal Mixture-of-Transformers. The Reasoner reads and writes text; the Generator denoises video, "
        "audio, and action. Which tokens are given and which are generated decides the job.",
    )
    top = 138
    box(ax, 40, top, W - 80, 44, fc=PANEL, ec="none", r=12)
    text(ax, 60, top + 22, "Cosmos 3", size=12.5, weight="bold", va="center_baseline")
    text(ax, 152, top + 22,
         "one checkpoint · shared attention · 3D mRoPE over absolute time · Edge 4B · Nano 16B · Super 64B",
         size=10, color=INK2, va="center_baseline")
    ty = top + 64
    th = 210
    towers = [
        (40, 500, DATA, "Reasoner tower", "autoregressive · causal attention",
         ["text", "image", "video"], ["text"],
         "Captions, timestamps, grounding boxes, plans, next actions, A/B physical-plausibility verdicts."),
        (660, 500, WORLD, "Generator tower", "diffusion · full attention over both towers",
         ["text", "image", "video", "action"], ["video", "audio", "action"],
         "World simulation, future prediction, synthetic data, policy learning."),
    ]
    for x, w, col, name, sub, ins, outs, desc in towers:
        box(ax, x, ty, w, th, fc=tint(col, 0.08), ec="none", r=14)
        accent_bar(ax, x + 10, ty + 16, th - 32, col, w=5)
        text(ax, x + 28, ty + 18, name, size=13.5, weight="bold")
        text(ax, x + 28, ty + 44, sub, size=9.8, color=INK2)
        text(ax, x + 28, ty + 80, "IN", size=8.5, weight="bold", color=MUTED, va="center_baseline")
        cx_ = x + 64
        for s in ins:
            cx_ += chip(ax, cx_, ty + 68, s, size=10) + 6
        text(ax, x + 28, ty + 118, "OUT", size=8.5, weight="bold", color=MUTED, va="center_baseline")
        cx_ = x + 64
        for s in outs:
            cx_ += chip(ax, cx_, ty + 106, s, size=10, weight="semibold", fc=tint(col, 0.22), ec="none") + 6
        para(ax, x + 28, ty + 148, desc, w - 56, size=10, color=INK)
    arrow(ax, (548, ty + 96), (652, ty + 96), color=INK2, lw=1.8, ms=15)
    para_center(ax, 600, ty + 110, "Generator attends to Reasoner tokens; never the reverse", 100, size=8.8, color=INK2, lh=1.3)

    my = ty + th + 40
    text(ax, 40, my, "GENERATOR MODES", size=9, weight="bold", color=MUTED)
    text(ax, 200, my, "Action is a modality, so one checkpoint covers all of these.", size=9.5, color=INK2)
    modes = [
        ("Forward dynamics", ["image", "actions"], ["video"], "What happens if the robot does this?"),
        ("Inverse dynamics", ["video", "(instruction)"], ["actions"], "Which actions explain this video? Turns human video into robot data."),
        ("Policy", ["image", "instruction", "(state)"], ["video", "action chunk"], "What should the robot do next, and what will it look like?"),
        ("Transfer", ["depth / seg / edge / blur hints", "text"], ["photoreal video"], "Keep the geometry, change the look. No ControlNets."),
    ]
    mw = (W - 80 - 3 * 14) / 4
    mh = 240
    for i, (name, given, gen, q) in enumerate(modes):
        x = 40 + i * (mw + 14)
        y = my + 26
        box(ax, x, y, mw, mh, fc=SURFACE, ec=HAIR, r=12)
        text(ax, x + 16, y + 16, name, size=12, weight="bold")
        text(ax, x + 16, y + 48, "given", size=8.5, weight="bold", color=MUTED)
        hgt = flow_chips(ax, x + 16, y + 66, mw - 32, given, size=9.5, gap=6, row_gap=6)
        yy = y + 66 + hgt + 10
        text(ax, x + 16, yy, "generated", size=8.5, weight="bold", color=MUTED)
        g_items = [dict(text=s, weight="semibold", fc=tint(WORLD, 0.22), ec="none") for s in gen]
        hgt2 = flow_chips(ax, x + 16, yy + 18, mw - 32, g_items, size=9.5, gap=6, row_gap=6)
        para(ax, x + 16, yy + 18 + hgt2 + 14, q, mw - 32, size=9.5, color=INK2)
    footer(ax, 40, H - 34,
           "Sources: Cosmos 3 technical report (June 2026) and Cosmos repository README / generator docs. "
           "Edge also runs on Jetson Thor; transfer is not available on Edge.")
    return save(fig, "cosmos3-modes", svg=False)


def fit_size(s, max_px, size, weight="normal", min_size=7.5, family=None):
    while size > min_size and measure(s, size, weight, family) > max_px:
        size -= 0.2
    return size


def chip_width_(s, size=9, weight="semibold"):
    return measure(s, size, weight) + 18


FIGS = {
    "stack": fig_stack,
    "three-generations": fig_generations,
    "newton-loop": fig_newton_loop,
    "newton-memory": fig_newton_memory,
    "flywheel": fig_flywheel,
    "project-architecture": fig_architecture,
    "cosmos3-modes": fig_cosmos3,
    "hero": lambda: __import__("make_hero").make_hero(),
}


def main(argv):
    names = argv[1:] or list(FIGS)
    for n in names:
        p = FIGS[n]()
        print("wrote", p)


if __name__ == "__main__":
    main(sys.argv)
