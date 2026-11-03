"""Make the workflow folder importable (pytest runs with --import-mode=importlib)."""

import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1]
if str(FOLDER) not in sys.path:
    sys.path.insert(0, str(FOLDER))
