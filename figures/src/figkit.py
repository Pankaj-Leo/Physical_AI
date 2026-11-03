"""Shared drawing toolkit for the Physical AI blog figures.

Every figure is drawn on a pixel canvas: the axes span 0..W by 0..H "CSS pixels"
(y grows downward, like a web page), and the PNG is saved at 2x. Font sizes are in
points; at 1x one point is 100/72 px, so 11 pt body text renders at ~15 px on a
1200 px blog column.

Palette: the dataviz reference categorical slots, used with a fixed meaning in
every figure (validated all-pairs for CVD with scripts/validate_palette.js):

    PHYS   blue    #2a78d6   physics & simulation (Newton, Isaac, Omniverse)
    WORLD  orange  #eb6834   learned world models (Cosmos, NuRec)
    REAL   aqua    #1baf7a   real world, robots, deployment
    DATA   violet  #4a3aa7   data, training, judging

Text is always ink (primary / secondary / muted), never the series color.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent
FONT_DIR = HERE / "fonts"

# --------------------------------------------------------------------------- fonts
SANS = "DejaVu Sans"
ROUND = "DejaVu Sans"
MONO = "DejaVu Sans Mono"
if FONT_DIR.is_dir():
    for f in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
    names = {
        font_manager.FontProperties(fname=str(f)).get_name()
        for f in FONT_DIR.glob("*.ttf")
    }
    if "Inter" in names:
        SANS = "Inter"
    if "Nunito" in names:
        ROUND = "Nunito"

plt.rcParams.update(
    {
        "font.family": SANS,
        "svg.fonttype": "path",  # SVGs render identically without the fonts installed
        "svg.hashsalt": "physical-ai",  # deterministic ids -> stable diffs
        "path.simplify": True,
    }
)

# --------------------------------------------------------------------------- colors
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
PANEL = "#f3f2ee"  # one step off the surface, for grouped regions
HAIR = "#d9d7cf"

PHYS = "#2a78d6"
WORLD = "#eb6834"
REAL = "#1baf7a"
DATA = "#4a3aa7"

# ordinal blue ramp (sequential steps 250 / 450 / 650) for ordered categories
ORD = ["#86b6ef", "#2a78d6", "#104281"]

PT2PX = 100 / 72  # 1 pt in 1x pixels


def tint(hex_color: str, amount: float = 0.12, base: str = SURFACE) -> str:
    """Mix `amount` of hex_color into base (a wash for card fills)."""
    c = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(base[i : i + 2], 16) for i in (1, 3, 5)]
    m = [round(b[k] + (c[k] - b[k]) * amount) for k in range(3)]
    return "#" + "".join(f"{v:02x}" for v in m)


# --------------------------------------------------------------------------- canvas
def canvas(w: int, h: int, bg: str = SURFACE):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_facecolor(bg)
    return fig, ax


def save(fig, name: str, svg: bool = True, out_dir: Path | None = None, dpi: int = 200):
    out = out_dir or FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=dpi, facecolor=fig.get_facecolor())
    if svg:
        fig.savefig(out / f"{name}.svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out / f"{name}.png"


# --------------------------------------------------------------------------- text
_FP_CACHE: dict = {}


def fp(weight="normal", family=None, style="normal") -> FontProperties:
    key = (weight, family or SANS, style)
    if key not in _FP_CACHE:
        _FP_CACHE[key] = FontProperties(
            family=family or SANS, weight=weight, style=style
        )
    return _FP_CACHE[key]


def measure(s: str, size: float, weight="normal", family=None) -> float:
    """Advance width of a single-line string, in 1x pixels."""
    if not s:
        return 0.0
    tp = TextPath((0, 0), s.replace(" ", " "), size=size, prop=fp(weight, family))
    ext = tp.get_extents()
    return ext.x1 * PT2PX


def wrap(s: str, max_px: float, size: float, weight="normal", family=None) -> list[str]:
    lines: list[str] = []
    for para in s.split("\n"):
        words = para.split(" ")
        cur = ""
        for w_ in words:
            trial = (cur + " " + w_).strip()
            if measure(trial, size, weight, family) <= max_px or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w_
        lines.append(cur)
    return lines


def text(
    ax,
    x,
    y,
    s,
    size=11,
    weight="normal",
    color=INK,
    ha="left",
    va="top",
    family=None,
    **kw,
):
    prop = fp(weight, family).copy()
    prop.set_size(size)
    if "z" in kw:
        kw["zorder"] = kw.pop("z")
    kw.setdefault("zorder", 5)
    return ax.text(x, y, s, fontproperties=prop, color=color, ha=ha, va=va, **kw)


def para(
    ax,
    x,
    y,
    s,
    max_px,
    size=11,
    weight="normal",
    color=INK,
    lh=1.38,
    ha="left",
    family=None,
):
    """Wrapped paragraph; returns the y just below the last line."""
    lines = wrap(s, max_px, size, weight, family)
    step = size * PT2PX * lh
    for i, ln in enumerate(lines):
        text(
            ax,
            x,
            y + i * step,
            ln,
            size=size,
            weight=weight,
            color=color,
            ha=ha,
            family=family,
        )
    return y + len(lines) * step


def para_height(s, max_px, size=11, weight="normal", lh=1.38, family=None):
    return len(wrap(s, max_px, size, weight, family)) * size * PT2PX * lh


# --------------------------------------------------------------------------- shapes
def box(ax, x, y, w, h, fc=SURFACE, ec=HAIR, r=10, lw=1.0, z=1, ls="-", alpha=1.0):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc,
        ec=ec,
        lw=lw,
        zorder=z,
        ls=ls,
        alpha=alpha,
        joinstyle="round",
    )
    ax.add_patch(p)
    return p


def accent_bar(ax, x, y, h, color, w=5, z=3):
    """A rounded vertical accent stripe at the left edge of a card."""
    return box(ax, x, y, w, h, fc=color, ec="none", r=w / 2, z=z)


def dot(ax, x, y, r, color, ring=SURFACE, z=4):
    ax.add_patch(Circle((x, y), r + 2, fc=ring, ec="none", zorder=z))
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=z + 0.1))


def arrow(
    ax,
    p0,
    p1,
    color=INK2,
    lw=1.6,
    rad=0.0,
    ms=13,
    z=2,
    style="-|>",
    patchA=None,
    patchB=None,
    shrinkA=2,
    shrinkB=2,
    ls="-",
):
    a = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=ms,
        color=color,
        lw=lw,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
        patchA=patchA,
        patchB=patchB,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        joinstyle="round",
        capstyle="round",
        linestyle=ls,
    )
    ax.add_patch(a)
    return a


def chip(
    ax,
    x,
    y,
    s,
    size=10,
    weight="normal",
    fc=SURFACE,
    ec=HAIR,
    color=INK,
    padx=9,
    h=None,
    lw=1.0,
    z=3,
    family=None,
):
    """Pill with text; returns its width."""
    tw = measure(s, size, weight, family)
    h = h or size * PT2PX + 12
    w = tw + 2 * padx
    box(ax, x, y, w, h, fc=fc, ec=ec, r=h / 2, lw=lw, z=z)
    text(
        ax,
        x + padx,
        y + h / 2,
        s,
        size=size,
        weight=weight,
        color=color,
        va="center_baseline",
        z=z + 1,
        family=family,
    )
    return w


def chip_width(s, size=10, weight="normal", padx=9, family=None):
    return measure(s, size, weight, family) + 2 * padx


def flow_chips(ax, x, y, max_w, items, size=10, gap=8, row_gap=8, h=None, draw=True):
    """Lay out chips left-to-right with wrapping. Items: str, '→', or dict(text=..., **chip kwargs).
    Returns total height used."""
    h = h or size * PT2PX + 12
    cx, cy = x, y
    for it in items:
        if it == "→":
            aw = measure("→", size + 1) + gap
            if cx + aw > x + max_w:
                cx, cy = x, cy + h + row_gap
            if draw:
                text(
                    ax,
                    cx + aw / 2 - gap / 2,
                    cy + h / 2,
                    "→",
                    size=size + 1,
                    color=MUTED,
                    ha="center",
                    va="center_baseline",
                )
            cx += aw
            continue
        d = dict(text=it) if isinstance(it, str) else dict(it)
        s = d.pop("text")
        w = chip_width(s, size, d.get("weight", "normal"), family=d.get("family"))
        if cx + w > x + max_w and cx > x:
            cx, cy = x, cy + h + row_gap
        if draw:
            chip(ax, cx, cy, s, size=size, h=h, **d)
        cx += w + gap
    return cy + h - y


def title_block(ax, x, y, title, subtitle=None, size=20, sub_size=11.5, max_px=1100):
    text(ax, x, y, title, size=size, weight="bold", color=INK)
    if subtitle:
        return para(
            ax, x, y + size * PT2PX * 1.45, subtitle, max_px, size=sub_size, color=INK2
        )
    return y + size * PT2PX * 1.4


def footer(ax, x, y, s, max_px=1120, size=8.5):
    return para(ax, x, y, s, max_px, size=size, color=MUTED, lh=1.35)


def legend_row(ax, x, y, entries, size=9.5, gap=22):
    """entries: list of (color, label). Swatch is a dot, label is ink."""
    cx = x
    for color, label in entries:
        dot(ax, cx + 6, y, 6, color)
        text(ax, cx + 18, y, label, size=size, color=INK2, va="center_baseline")
        cx += 18 + measure(label, size) + gap
    return cx


def wrapped(s, width):
    return "\n".join(textwrap.wrap(s, width))


if os.environ.get("FIGKIT_DEBUG"):
    print("fonts:", SANS, ROUND, MONO)
