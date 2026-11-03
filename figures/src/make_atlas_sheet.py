#!/usr/bin/env python3
"""Contact sheet that binds the workflow atlas together: one card per workflow, with the source screenshots it
reproduces as thumbnails. Reads only the vetted copies in physical-ai-project/assets/screens/ (decision D-11).

Usage:  python physical-ai-project/figures/src/make_atlas_sheet.py
Writes: physical-ai-project/figures/atlas-contact-sheet.jpg
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
SCREENS = PROJ / "assets" / "screens"
OUT = PROJ / "figures" / "atlas-contact-sheet.jpg"
FONTS = HERE / "fonts"

# (folder id, title, layer label, accent, evidence files (max 3), source line)
CARDS = [
    (
        "01",
        "OpenUSD & SimReady asset",
        "Data model",
        "#3b6fb6",
        [
            "simready-robot-asset-structure.jpg",
            "usd-content-journey.jpg",
            "simready-playbook.png",
        ],
        "Omniverse/OpenUSD deck · OpenUSD tutorials",
    ),
    (
        "02",
        "Isaac Sim + ROS 2 Nav2",
        "Robotics apps",
        "#3b6fb6",
        [
            "nav2-omnigraph-inspector.jpg",
            "nav2-wizard-step5.jpg",
            "isaac-sim-turtlebot3.jpg",
        ],
        "Nav2 wizard · Isaac Sim tutorials #6 to #11, #15",
    ),
    (
        "03",
        "MoveIt 2 + Panda",
        "Robotics apps",
        "#3b6fb6",
        ["moveit-rviz-panda.jpg", "moveit-isaac-sim-panda.jpg"],
        "Isaac Sim tutorial #13",
    ),
    (
        "04",
        "Isaac Lab RL",
        "Robot learning",
        "#7a5cc4",
        ["isaac-lab-rl-loop.jpg"],
        "NVIDIA Developer Blog diagram · tutorials #12, #14",
    ),
    (
        "05",
        "Replicator SDG → detector",
        "Synthetic data",
        "#7a5cc4",
        ["physical-ai-flywheel.jpg"],
        "Omniverse Basic Training 05, 06",
    ),
    (
        "06",
        "Newton at GPU scale",
        "Physics",
        "#3b6fb6",
        [],
        "Public SIGGRAPH 2026 Newton talk · newton-main code",
    ),
    (
        "07",
        "Cosmos 3 & Cosmos-Dreams",
        "World models",
        "#d0743c",
        [
            "cosmos-dreams-loop.jpg",
            "cosmos-dreams-recipe.jpg",
            "cosmos-dreams-eval.jpg",
        ],
        "SIGGRAPH 2026 AV and World Models talks · cosmos-main",
    ),
    (
        "08",
        "NuRec for AV",
        "Neural reconstruction",
        "#d0743c",
        [
            "nurec-reference-pipeline.jpg",
            "mcity-cost-comparison.jpg",
            "av-data-pipeline.jpg",
        ],
        "NuRec deck · Mcity · AV simulation talk",
    ),
    (
        "09",
        "cuOpt forklift fleet",
        "Industrial ops",
        "#2b9a8a",
        ["forklift-fleet-live.jpg", "forklift-fleet-complete.jpg"],
        "Forklift Fleet wizard (Isaac Sim + cuOpt)",
    ),
    (
        "10",
        "Omniverse Kit extension",
        "Platform",
        "#3b6fb6",
        [
            "omniverse-overview.jpg",
            "omniverse-skills.jpg",
            "omniverse-libraries-launch.jpg",
        ],
        "Omniverse tutorials · SIGGRAPH 2026 agentic talk",
    ),
    (
        "11",
        "Real-time engineering twin",
        "AI physics",
        "#2b9a8a",
        [
            "ai-enhanced-simulation.jpg",
            "physicsnemo.jpg",
            "industrial-engineering-twins.jpg",
        ],
        "SIGGRAPH 2026 real-time digital twin talk",
    ),
]

W, COLS = 2400, 3
PAD, GAP = 48, 28
CARD_W = (W - 2 * PAD - (COLS - 1) * GAP) // COLS
THUMB_H = 150
CARD_H = 92 + THUMB_H + 64
BG, CARD_BG, INK, MUTED, LINE = "#faf8f4", "#ffffff", "#1f2328", "#5b6470", "#e3ded5"


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    files = {"Regular": "Inter_400Regular.ttf", "SemiBold": "Inter_600SemiBold.ttf", "Bold": "Inter_700Bold.ttf"}
    for name in (files.get(weight, "Inter_400Regular.ttf"), "Inter_400Regular.ttf"):
        p = FONTS / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def thumb(path: Path, w: int, h: int) -> Image.Image:
    """Fit the whole slide inside w x h (letterboxed), so titles are never cropped."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), "#f1eee8")
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def main() -> None:
    rows = -(-len(CARDS) // COLS)
    header = 170
    H = header + rows * CARD_H + (rows - 1) * GAP + PAD + 40
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text(
        (PAD, 44),
        "Workflow atlas: from screenshots to code",
        font=font("Bold", 56),
        fill=INK,
    )
    d.text(
        (PAD, 118),
        "Each card is one workflow folder. Thumbnails are the source slides and screens it reproduces.",
        font=font("Regular", 28),
        fill=MUTED,
    )
    f_id, f_title, f_layer, f_src = (
        font("Bold", 30),
        font("SemiBold", 30),
        font("SemiBold", 20),
        font("Regular", 20),
    )
    for i, (fid, title, layer, accent, files, src) in enumerate(CARDS):
        r, c = divmod(i, COLS)
        x, y = PAD + c * (CARD_W + GAP), header + r * (CARD_H + GAP)
        d.rounded_rectangle(
            (x, y, x + CARD_W, y + CARD_H),
            radius=18,
            fill=CARD_BG,
            outline=LINE,
            width=2,
        )
        d.rounded_rectangle((x, y, x + 10, y + CARD_H), radius=6, fill=accent)
        d.text((x + 30, y + 22), fid, font=f_id, fill=accent)
        d.text((x + 80, y + 22), title, font=f_title, fill=INK)
        d.text((x + 30, y + 62), layer.upper(), font=f_layer, fill=MUTED)
        tx, ty = x + 30, y + 92
        avail = CARD_W - 60
        if files:
            n = len(files)
            tw = (avail - (n - 1) * 12) // n
            for k, f in enumerate(files):
                im = thumb(SCREENS / f, tw, THUMB_H)
                img.paste(im, (tx + k * (tw + 12), ty))
                d.rectangle(
                    (tx + k * (tw + 12), ty, tx + k * (tw + 12) + tw, ty + THUMB_H),
                    outline=LINE,
                    width=1,
                )
        else:
            d.rounded_rectangle(
                (tx, ty, tx + avail, ty + THUMB_H),
                radius=10,
                fill="#eef2f8",
                outline=LINE,
            )
            msg = "Newton deck slides are confidential: not shown.\nCode traced to newton-main + the public talk."
            d.multiline_text(
                (tx + 24, ty + 40),
                msg,
                font=font("Regular", 22),
                fill=MUTED,
                spacing=10,
            )
        d.text((x + 30, y + 92 + THUMB_H + 20), src, font=f_src, fill=MUTED)
    d.text(
        (PAD, H - 52),
        "Slides and screens remain the work of their presenters (NVIDIA, Mcity, Wizardverse tutorials); "
        "shown for commentary. Index: workflows/README.md · manifest: assets/screens/MANIFEST.md",
        font=font("Regular", 20),
        fill=MUTED,
    )
    img.save(OUT, quality=85, optimize=True)
    print(f"wrote {OUT.relative_to(PROJ)} ({OUT.stat().st_size // 1024} KB, {W}x{H})")


if __name__ == "__main__":
    main()
