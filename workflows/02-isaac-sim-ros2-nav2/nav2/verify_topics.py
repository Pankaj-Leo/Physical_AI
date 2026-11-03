"""Wizard step 4: check that Isaac Sim publishes the topics Nav2 needs.

The talk: the NVIDIA sample publishes over 100 topics, but "you only care about" the sensors, the clock and the
command velocity. The expected set comes from graph_spec.py. ``missing_topics`` runs on CPU; ``main`` needs ROS 2.

    python3 /ws/nav2/verify_topics.py            # inside the nav2 container, while Isaac Sim is playing
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import graph_spec as gs

# /cmd_vel shows up in `ros2 topic list` because Isaac Sim subscribes to it.
NAV2_NEEDS = {"/clock", "/tf", "/odom", "/scan", f"/{gs.CMD_VEL_TOPIC}"}


def expected_topics(profile_key: str = "nova_carter") -> set[str]:
    return gs.published_topics(gs.PROFILES[profile_key]) | {f"/{gs.CMD_VEL_TOPIC}"}


def missing_topics(expected: set[str], listed: str) -> list[str]:
    present = {line.strip() for line in listed.splitlines() if line.strip()}
    return sorted(expected - present)


def main() -> int:
    listed = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True, check=True).stdout
    missing = missing_topics(expected_topics(), listed)
    critical = sorted(set(missing) & NAV2_NEEDS)
    print(f"{len(listed.split())} topics listed; missing: {missing or 'none'}")
    if critical:
        print(f"Nav2 cannot start without: {critical}. Is Isaac Sim playing? Same ROS_DOMAIN_ID?")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
