"""Write the two tutorial #13 fixes as unified diffs against the Setup Assistant output (runs on CPU).

- ``moveit_controllers.yaml``: the arm controller lacks ``action_ns`` and ``default``, so Plan & Execute fails.
- ``panda.ros2_control.xacro``: swap the mock hardware for topic-based ros2_control on the Isaac Sim topics.

The ``*.generated.*`` files are reconstructions of MoveIt Setup Assistant (Humble) output; diff them against your own
generated package before applying. Apply from inside your package:  patch -p1 < patches/<file>.patch

    python make_patches.py          # (re)write patches/
    python make_patches.py --check  # exit 1 if patches/ is stale
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "moveit_config" / "config"
PAIRS = {
    "moveit_controllers.yaml.patch": ("moveit_controllers.generated.yaml", "moveit_controllers.yaml"),
    "panda.ros2_control.xacro.patch": ("panda.ros2_control.generated.xacro", "panda.ros2_control.xacro"),
}


def make_patch(generated: str, fixed: str) -> str:
    a = (CONFIG / generated).read_text().splitlines(keepends=True)
    b = (CONFIG / fixed).read_text().splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, fromfile=f"a/config/{fixed}", tofile=f"b/config/{fixed}"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    stale = []
    for name, (generated, fixed) in PAIRS.items():
        text = make_patch(generated, fixed)
        path = HERE / "patches" / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(name)
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(HERE)}")
    if stale:
        print(f"stale patches: {stale}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
