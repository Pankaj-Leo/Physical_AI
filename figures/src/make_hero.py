"""Placeholder hero infographic: physical-ai-project/PhysicalAI.png (2752 x 1536).

Mirrors the composition of sample/Finetuning.png (title, two rounded panels, cards,
a connecting pipe) in a flat "clay" style: warm beige, rounded shapes, soft offset
shadows and highlights. Icons are generic shapes only (no logos or product art).
It is meant to be replaced by a NotebookLM image generated from INFOGRAPHIC_BRIEF.md.

    python physical-ai-project/figures/src/make_hero.py
"""

from __future__ import annotations

import numpy as np
from figkit import FIG_DIR, ROUND, SANS, canvas, measure, save, text, wrap
from matplotlib import patheffects as pe
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle

W, H = 2752, 1536

BG = "#ecdcc3"
BG_DARK = "#e2cfb0"
RIM = "#c4875f"
RIM_DARK = "#a86a45"
PANEL_FILL = "#e8c99c"
CARD = "#f7ecd9"
INK = "#3b332c"
INK2 = "#5a4c40"
BLUE = "#4f7fb8"
TEAL = "#3f9c86"
TERRA = "#cf6f48"
MUSTARD = "#dfae45"
STEEL = "#8ea4b5"
STEEL_DARK = "#6f8798"
CHAR = "#4a4642"
SAND = "#dbb57d"
WHITE = "#fffaf1"

SH = "#6b4a2e"  # shadow tone


# ----------------------------------------------------------------------------- clay helpers
def rbox(x, y, w, h, r, **kw):
    return FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", **kw)


def clay(ax, make, fc, z=2, depth=10, hi=True, ec="none", lw=0):
    """Draw a shape with a soft drop shadow (stacked offsets) and a top-left highlight."""
    for k, a in ((1.0, 0.10), (0.7, 0.10), (0.4, 0.12)):
        ax.add_patch(
            make(depth * k * 0.6, depth * k, fc=SH, ec="none", alpha=a, zorder=z - 0.3)
        )
    ax.add_patch(make(0, 0, fc=fc, ec=ec, lw=lw, zorder=z))
    if hi:
        ax.add_patch(
            make(-2, -2, fc="none", ec=WHITE, lw=3, alpha=0.35, zorder=z + 0.05)
        )


def clay_box(ax, x, y, w, h, r, fc, z=2, depth=10, hi=True, ec="none", lw=0):
    clay(
        ax,
        lambda dx, dy, **k: rbox(x + dx, y + dy, w, h, r, **k),
        fc,
        z,
        depth,
        hi,
        ec,
        lw,
    )


def clay_circle(ax, x, y, r, fc, z=3, depth=6, hi=True):
    clay(ax, lambda dx, dy, **k: Circle((x + dx, y + dy), r, **k), fc, z, depth, hi)


def clay_poly(ax, pts, fc, z=3, depth=6, hi=False):
    pts = np.asarray(pts, float)
    clay(
        ax,
        lambda dx, dy, **k: Polygon(
            pts + [dx, dy], closed=True, joinstyle="round", **k
        ),
        fc,
        z,
        depth,
        hi,
    )


def snake(ax, xs, ys, color, lw=26, z=1):
    """A rolled clay coil: shadow, body, highlight."""
    ax.plot(
        np.asarray(xs) + 6,
        np.asarray(ys) + 10,
        color=SH,
        alpha=0.18,
        lw=lw,
        solid_capstyle="round",
        zorder=z - 0.2,
    )
    ax.plot(
        xs,
        ys,
        color=color,
        lw=lw,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=z,
    )
    ax.plot(
        np.asarray(xs) - 3,
        np.asarray(ys) - 4,
        color=WHITE,
        alpha=0.30,
        lw=lw * 0.28,
        solid_capstyle="round",
        zorder=z + 0.05,
    )


def gear(ax, cx, cy, r, color, teeth=8, z=1.5):
    th = np.linspace(0, 2 * np.pi, teeth * 4, endpoint=False)
    rad = np.where((np.arange(len(th)) // 2) % 2 == 0, r, r * 0.78)
    pts = np.c_[cx + rad * np.cos(th), cy + rad * np.sin(th)]
    clay_poly(ax, pts, color, z=z, depth=8, hi=True)
    ax.add_patch(Circle((cx, cy), r * 0.32, fc=BG, ec="none", zorder=z + 0.1))


def word_width(t, size, weight=900, family=ROUND):
    """Advance width including trailing spaces (ink extents drop them)."""
    space = measure("a a", size, weight, family) - measure("aa", size, weight, family)
    stripped = t.rstrip(" ")
    return measure(stripped, size, weight, family) + space * (len(t) - len(stripped))


def word_title(ax, x, y, parts, size):
    """Multi-colored title, each word with a clay outline and drop shadow."""
    total = sum(word_width(t, size) for t, _ in parts)
    cx = x - total / 2
    for t, col in parts:
        tx = text(
            ax,
            cx,
            y,
            t.rstrip(" "),
            size=size,
            weight=900,
            color=col,
            family=ROUND,
            va="center_baseline",
            z=20,
        )
        tx.set_path_effects(
            [
                pe.withSimplePatchShadow(offset=(4, -6), shadow_rgbFace=SH, alpha=0.30),
                pe.withStroke(linewidth=10, foreground="#f4e6cf"),
                pe.Normal(),
            ]
        )
        cx += word_width(t, size)


def icon_usd(ax, cx, cy):
    cols = [TEAL, MUSTARD, BLUE]
    for i, c in enumerate(cols):
        oy = cy + 50 - i * 42
        pts = [(cx - 130, oy), (cx + 40, oy), (cx + 130, oy - 52), (cx - 40, oy - 52)]
        clay_poly(ax, pts, c, z=4 + i, depth=8, hi=True)
    text(
        ax,
        cx + 2,
        cy - 54,
        "USD",
        size=26,
        weight=900,
        color=WHITE,
        family=ROUND,
        ha="center",
        va="center_baseline",
        z=8,
        rotation=0,
    )


def quadruped(ax, cx, cy, s=1.0, body=STEEL, z=6):
    # legs
    for lx in (-48, -22, 26, 50):
        clay_box(
            ax,
            cx + lx * s - 7 * s,
            cy,
            14 * s,
            46 * s,
            6 * s,
            STEEL_DARK,
            z=z,
            depth=4,
            hi=False,
        )
    clay_box(
        ax, cx - 66 * s, cy - 30 * s, 132 * s, 44 * s, 20 * s, body, z=z + 0.5, depth=5
    )
    clay_box(
        ax, cx + 52 * s, cy - 52 * s, 40 * s, 34 * s, 12 * s, body, z=z + 0.6, depth=4
    )
    ax.add_patch(Circle((cx + 80 * s, cy - 36 * s), 5 * s, fc=CHAR, zorder=z + 1))


def icon_newton(ax, cx, cy):
    # sand pile with grains, quadruped on top
    ax.add_patch(Ellipse((cx + 8, cy + 84), 330, 120, fc=SH, alpha=0.15, zorder=3))
    clay(
        ax,
        lambda dx, dy, **k: Ellipse((cx + dx, cy + 70 + dy), 320, 130, **k),
        SAND,
        z=4,
        depth=6,
    )
    rng = np.random.default_rng(3)
    for _ in range(70):
        a = rng.uniform(0, 2 * np.pi)
        rr = np.sqrt(rng.uniform(0, 1))
        gx, gy = cx + 150 * rr * np.cos(a), cy + 70 + 56 * rr * np.sin(a)
        if gy < cy + 40:
            continue
        ax.add_patch(
            Circle((gx, gy), rng.uniform(3, 6), fc="#c69a5c", ec="none", zorder=4.5)
        )
    quadruped(ax, cx - 6, cy - 6, s=1.0)


def icon_isaac(ax, cx, cy):
    cols = [STEEL, BLUE, STEEL, TEAL]
    for r in range(3):
        for c in range(5):
            x = cx - 170 + c * 72
            y = cy - 80 + r * 62
            col = cols[(r + c) % len(cols)]
            clay_box(ax, x, y, 52, 44, 12, col, z=4, depth=5)
            ax.add_patch(Circle((x + 17, y + 20), 4.5, fc=WHITE, zorder=5))
            ax.add_patch(Circle((x + 35, y + 20), 4.5, fc=WHITE, zorder=5))


def icon_replicator(ax, cx, cy):
    # camera
    clay_box(ax, cx - 150, cy - 50, 150, 104, 18, CHAR, z=4, depth=7)
    clay_box(ax, cx - 120, cy - 74, 54, 28, 8, CHAR, z=3.9, depth=4)
    clay_circle(ax, cx - 75, cy + 2, 34, STEEL, z=5, depth=3)
    ax.add_patch(Circle((cx - 75, cy + 2), 18, fc=BLUE, zorder=5.5))
    # randomized cubes
    for dx, dy, c, s in [
        (40, -50, TERRA, 52),
        (100, -10, MUSTARD, 46),
        (50, 30, TEAL, 56),
        (120, 50, BLUE, 40),
    ]:
        clay_box(ax, cx + dx, cy + dy, s, s, 10, c, z=4, depth=6)


def icon_nurec(ax, cx, cy):
    # gaussian splats scene
    rng = np.random.default_rng(7)
    cols = [BLUE, TEAL, MUSTARD, TERRA, STEEL]
    for _ in range(26):
        ex, ey = cx + rng.uniform(-160, 160), cy + rng.uniform(-95, -10)
        ax.add_patch(
            Ellipse(
                (ex, ey),
                rng.uniform(26, 60),
                rng.uniform(14, 30),
                angle=rng.uniform(0, 180),
                fc=cols[rng.integers(0, len(cols))],
                alpha=0.75,
                ec="none",
                zorder=3.5,
            )
        )
    # road
    clay_box(ax, cx - 175, cy + 40, 350, 30, 14, "#b9a98f", z=3.6, depth=4, hi=False)
    # car with roof camera
    clay_box(ax, cx - 90, cy - 4, 180, 56, 22, TERRA, z=5, depth=6)
    clay_box(ax, cx - 52, cy - 38, 100, 42, 16, TERRA, z=4.9, depth=4)
    clay_box(ax, cx - 10, cy - 58, 26, 22, 6, CHAR, z=5.2, depth=3)
    for wx in (cx - 52, cx + 52):
        clay_circle(ax, wx, cy + 54, 20, CHAR, z=5.5, depth=3)
    # scan rays
    for ang in (-40, -15, 15, 40):
        a = np.radians(-90 + ang)
        ax.plot(
            [cx + 3, cx + 3 + 120 * np.cos(a)],
            [cy - 58, cy - 58 + 120 * np.sin(a) * 0.5],
            color=WHITE,
            lw=3,
            alpha=0.8,
            solid_capstyle="round",
            zorder=5.3,
        )


def icon_cosmos(ax, cx, cy):
    # wireframe frame -> photoreal frame -> check badge
    x0, y0, w, h = cx - 190, cy - 70, 150, 120
    clay_box(ax, x0, y0, w, h, 14, "#f2e3c8", z=4, depth=6)
    for i in range(1, 5):
        ax.plot(
            [x0 + 10, x0 + w - 10],
            [y0 + i * h / 5] * 2,
            color=STEEL_DARK,
            lw=2,
            zorder=5,
        )
        ax.plot(
            [x0 + i * w / 5] * 2,
            [y0 + 10, y0 + h - 10],
            color=STEEL_DARK,
            lw=2,
            zorder=5,
        )
    ax.plot(
        [x0 + 20, x0 + 70, x0 + 130],
        [y0 + h - 20, y0 + 30, y0 + h - 20],
        color=BLUE,
        lw=4,
        zorder=5.2,
    )
    # arrow
    clay_poly(
        ax,
        [
            (cx - 30, cy - 18),
            (cx + 4, cy - 18),
            (cx + 4, cy - 34),
            (cx + 34, cy - 8),
            (cx + 4, cy + 18),
            (cx + 4, cy + 2),
            (cx - 30, cy + 2),
        ],
        MUSTARD,
        z=5,
        depth=4,
    )
    x1 = cx + 46
    clay_box(ax, x1, y0, w, h, 14, "#9cc3e6", z=4, depth=6)
    ax.add_patch(
        rbox(x1 + 8, y0 + 64, w - 16, h - 72, 10, fc=TEAL, ec="none", zorder=4.5)
    )
    ax.add_patch(
        Polygon(
            [(x1 + 20, y0 + 70), (x1 + 70, y0 + 28), (x1 + 120, y0 + 70)],
            fc="#6a8a6a",
            zorder=4.6,
        )
    )
    ax.add_patch(Circle((x1 + 118, y0 + 30), 13, fc=MUSTARD, zorder=4.6))
    # check badge
    clay_circle(ax, x1 + w, y0 + h - 4, 30, TEAL, z=6, depth=5)
    ax.plot(
        [x1 + w - 14, x1 + w - 3, x1 + w + 15],
        [y0 + h - 4, y0 + h + 8, y0 + h - 16],
        color=WHITE,
        lw=6,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=6.5,
    )


def icon_groot(ax, cx, cy):
    # friendly robot head and shoulders (nudged down so the antenna stays inside the card)
    cy += 24
    clay_box(ax, cx - 130, cy + 30, 260, 70, 30, STEEL, z=4, depth=6)
    clay_box(ax, cx - 30, cy + 8, 60, 30, 8, STEEL_DARK, z=4.2, depth=3, hi=False)
    clay_box(ax, cx - 100, cy - 110, 200, 128, 46, "#c9d3da", z=5, depth=7)
    clay_box(ax, cx - 80, cy - 80, 160, 64, 28, CHAR, z=5.5, depth=3, hi=False)
    ax.add_patch(Circle((cx - 36, cy - 48), 13, fc="#7fc4ff", zorder=6))
    ax.add_patch(Circle((cx + 36, cy - 48), 13, fc="#7fc4ff", zorder=6))
    for sx in (-1, 1):
        clay_circle(ax, cx + sx * 108, cy - 48, 18, BLUE, z=4.8, depth=3)
    ax.plot(
        [cx, cx],
        [cy - 110, cy - 140],
        color=STEEL_DARK,
        lw=6,
        zorder=4.9,
        solid_capstyle="round",
    )
    clay_circle(ax, cx, cy - 146, 12, TERRA, z=5, depth=3)


def icon_thor(ax, cx, cy):
    # edge module: a chip with pins, mounted on a small robot
    quadruped(ax, cx + 60, cy + 40, s=0.9, body=STEEL)
    x0, y0 = cx - 190, cy - 100
    for i in range(6):
        for px, py, w_, h_ in [
            (x0 + 22 + i * 22, y0 - 16, 10, 18),
            (x0 + 22 + i * 22, y0 + 150, 10, 18),
        ]:
            ax.add_patch(rbox(px, py, w_, h_, 3, fc="#b8a27a", ec="none", zorder=4))
        for px, py, w_, h_ in [
            (x0 - 16, y0 + 22 + i * 22, 18, 10),
            (x0 + 150, y0 + 22 + i * 22, 18, 10),
        ]:
            ax.add_patch(rbox(px, py, w_, h_, 3, fc="#b8a27a", ec="none", zorder=4))
    clay_box(ax, x0, y0, 152, 152, 18, CHAR, z=5, depth=7)
    clay_box(ax, x0 + 30, y0 + 30, 92, 92, 12, "#6e6a64", z=5.5, depth=2, hi=True)
    text(
        ax,
        x0 + 76,
        y0 + 76,
        "EDGE",
        size=17,
        weight=900,
        color=WHITE,
        family=ROUND,
        ha="center",
        va="center_baseline",
        z=7,
    )


# ----------------------------------------------------------------------------- composition
def card(ax, x, y, w, h, title, caption, icon):
    clay_box(ax, x, y, w, h, 34, CARD, z=3, depth=12)
    icon(ax, x + w / 2, y + 128)
    ty = y + 262
    tsize = 33
    while measure(title, tsize, 900, ROUND) > w - 60 and tsize > 29:
        tsize -= 1
    tlines = (
        [title]
        if measure(title, tsize, 900, ROUND) <= w - 60
        else wrap(title, w - 60, tsize, 900, ROUND)
    )
    for i, ln in enumerate(tlines):
        text(
            ax,
            x + w / 2,
            ty + i * tsize * 1.35,
            ln,
            size=tsize,
            weight=900,
            color=INK,
            family=ROUND,
            ha="center",
            z=10,
        )
    cy = ty + len(tlines) * tsize * 1.35 + 18
    lines = wrap(caption, w - 70, 19.5, 700, ROUND)
    for i, ln in enumerate(lines):
        text(
            ax,
            x + w / 2,
            cy + i * 36,
            ln,
            size=19.5,
            weight=700,
            color=INK2,
            family=ROUND,
            ha="center",
            z=10,
        )


def panel(ax, x, y, w, h, title, cards):
    # rim + fill
    clay_box(ax, x, y, w, h, 60, RIM, z=2, depth=16)
    ax.add_patch(
        rbox(x + 26, y + 26, w - 52, h - 52, 40, fc=PANEL_FILL, ec="none", zorder=2.2)
    )
    ax.add_patch(
        rbox(
            x + 26,
            y + 26,
            w - 52,
            h - 52,
            40,
            fc="none",
            ec=RIM_DARK,
            lw=3,
            alpha=0.5,
            zorder=2.25,
        )
    )
    # title tab
    tw = measure(title, 40, 900, ROUND) + 110
    clay_box(ax, x + w / 2 - tw / 2, y - 52, tw, 96, 48, RIM, z=8, depth=10)
    t = text(
        ax,
        x + w / 2,
        y - 4,
        title,
        size=40,
        weight=900,
        color="#fff4e3",
        family=ROUND,
        ha="center",
        va="center_baseline",
        z=9,
    )
    t.set_path_effects(
        [
            pe.withSimplePatchShadow(offset=(2, -3), shadow_rgbFace=SH, alpha=0.35),
            pe.Normal(),
        ]
    )
    # 2x2 cards
    pad, gap = 62, 34
    cw = (w - 2 * pad - gap) / 2
    ch = (h - 2 * pad - gap - 20) / 2
    for i, (ttl, cap, ico) in enumerate(cards):
        cx = x + pad + (i % 2) * (cw + gap)
        cy = y + pad + 20 + (i // 2) * (ch + gap)
        card(ax, cx, cy, cw, ch, ttl, cap, ico)


def background(ax):
    ax.add_patch(Rectangle((0, 0), W, H, fc=BG, ec="none", zorder=0))
    rng = np.random.default_rng(11)
    for _ in range(60):
        ax.add_patch(
            Ellipse(
                (rng.uniform(0, W), rng.uniform(0, H)),
                rng.uniform(150, 600),
                rng.uniform(60, 220),
                angle=rng.uniform(0, 180),
                fc=rng.choice([BG_DARK, "#f3e7d3"]),
                alpha=0.35,
                ec="none",
                zorder=0.1,
            )
        )
    t = np.linspace(0, 1, 120)
    snake(ax, 40 + 60 * np.sin(t * 5), 250 + 1150 * t, "#d9a77c", lw=30, z=0.6)
    snake(ax, W - 50 - 50 * np.sin(t * 4 + 1), 120 + 1250 * t, "#b9c4cc", lw=28, z=0.6)
    snake(ax, 300 + 900 * t, H - 30 + 18 * np.sin(t * 9), "#d9a77c", lw=24, z=0.6)
    snake(ax, 1500 + 1000 * t, 40 + 16 * np.sin(t * 8), "#c98a64", lw=22, z=0.6)
    gear(ax, 95, 145, 50, STEEL)
    gear(ax, W - 120, H - 150, 56, STEEL)
    gear(ax, 1376, H - 70, 40, "#a7b3bc")
    gear(ax, W - 230, 300, 34, "#d0a07a")


def pipe(ax, x0, x1, y):
    # tube between the panels, with collars at each end, and a label tag
    ax.plot(
        [x0, x1],
        [y + 12, y + 12],
        color=SH,
        alpha=0.2,
        lw=64,
        solid_capstyle="butt",
        zorder=1.5,
    )
    ax.plot([x0, x1], [y, y], color=STEEL, lw=60, solid_capstyle="butt", zorder=1.6)
    ax.plot(
        [x0, x1],
        [y - 12, y - 12],
        color=WHITE,
        lw=10,
        alpha=0.35,
        solid_capstyle="butt",
        zorder=1.7,
    )
    for cx in (x0 + 20, x1 - 20):
        clay_box(ax, cx - 22, y - 46, 44, 92, 14, STEEL_DARK, z=1.8, depth=5)
    mid = (x0 + x1) / 2
    a_w = measure("→", 26, 800, SANS)
    sw, rw, gap = measure("sim", 26, 900, ROUND), measure("real", 26, 900, ROUND), 14
    tw = sw + rw + a_w + 2 * gap + 60
    clay_box(ax, mid - tw / 2, y - 140, tw, 64, 32, CARD, z=9, depth=8)
    x = mid - (sw + rw + a_w + 2 * gap) / 2
    for t, fam, wt, ww in (
        ("sim", ROUND, 900, sw),
        ("→", SANS, 800, a_w),
        ("real", ROUND, 900, rw),
    ):
        text(
            ax,
            x,
            y - 108,
            t,
            size=26,
            weight=wt,
            color=INK,
            family=fam,
            va="center_baseline",
            z=10,
        )
        x += ww + gap
    ax.plot(
        [mid, mid],
        [y - 76, y - 32],
        color=STEEL_DARK,
        lw=6,
        zorder=8.5,
        solid_capstyle="round",
    )


def make_hero():
    fig, ax = canvas(W, H, bg=BG)
    background(ax)
    word_title(
        ax, W / 2, 128, [("The ", INK), ("Physical AI ", BLUE), ("Toolbox", TERRA)], 92
    )

    left = [
        (
            "OpenUSD & SimReady",
            "Describes robots and scenes in layers, with the joints, colliders, and mass a simulator needs.",
            icon_usd,
        ),
        (
            "Newton",
            "Open-source GPU physics for robot learning: rigid bodies, cloth, cables, and sand.",
            icon_newton,
        ),
        (
            "Isaac Sim & Isaac Lab",
            "Runs the robot with real ROS 2 software and trains policies in thousands of worlds.",
            icon_isaac,
        ),
        (
            "Omniverse & Replicator",
            "Renders physically based sensors and randomizes scenes into labeled data.",
            icon_replicator,
        ),
    ]
    right = [
        (
            "NuRec",
            "Rebuilds real places from sensor logs as 3D scenes you can drive through again.",
            icon_nurec,
        ),
        (
            "Cosmos Transfer & Reasoner",
            "Turns simulator depth and segmentation into photoreal video, then judges it.",
            icon_cosmos,
        ),
        (
            "GR00T & robot foundation models",
            "Generalist policies trained on real, simulated, and dreamed data.",
            icon_groot,
        ),
        (
            "Jetson Thor edge",
            "Runs the distilled policy on the robot; big world models stay in the data center.",
            icon_thor,
        ),
    ]
    py, ph = 300, 1140
    lx, pw = 150, 1170
    rx = W - 150 - pw
    pipe(ax, lx + pw - 30, rx + 30, py + ph / 2 + 10)
    panel(ax, lx, py, pw, ph, "Physics & Simulation", left)
    panel(ax, rx, py, pw, ph, "World Models & Deployment", right)

    text(
        ax,
        W - 40,
        H - 22,
        "placeholder — regenerate with INFOGRAPHIC_BRIEF.md",
        size=11,
        weight=700,
        color="#8a7660",
        family=ROUND,
        ha="right",
        va="bottom",
        z=20,
    )
    return save(fig, "PhysicalAI", svg=False, out_dir=FIG_DIR.parent, dpi=100)


if __name__ == "__main__":
    print("wrote", make_hero())
