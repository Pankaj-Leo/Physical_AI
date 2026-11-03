#!/usr/bin/env python3
"""Static and CPU checks for every workflow folder.

1. Every .py file compiles (catches syntax errors in GPU-only code we cannot run here).
2. Every workflow folder has a README with the required sections.
3. Every evidence image a README links to exists.
Run from anywhere:  python physical-ai-project/workflows/check_workflows.py
Then:               cd physical-ai-project/workflows && ruff check . && pytest
"""

from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED = ["## Source evidence", "## Code", "## Run it", "## Source step", "## What we verified here"]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def main() -> int:
    problems: list[str] = []
    folders = sorted(p for p in HERE.iterdir() if p.is_dir() and re.match(r"\d\d-", p.name))
    for py in HERE.rglob("*.py"):
        try:
            py_compile.compile(str(py), doraise=True)
        except py_compile.PyCompileError as exc:
            problems.append(f"compile: {py.relative_to(HERE)}: {exc.msg}")
    for folder in folders:
        readme = folder / "README.md"
        if not readme.exists():
            problems.append(f"{folder.name}: missing README.md")
            continue
        text = readme.read_text(encoding="utf-8")
        problems += [f"{folder.name}: README lacks '{s}'" for s in REQUIRED if s not in text]
        for target in LINK.findall(text):
            if re.match(r"^(https?:|mailto:|#)", target):
                continue
            if not (folder / target.split("#", 1)[0]).exists():
                problems.append(f"{folder.name}: broken link {target}")
    print(f"checked {len(folders)} workflow folders, {sum(1 for _ in HERE.rglob('*.py'))} Python files")
    for p in problems:
        print("PROBLEM", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
