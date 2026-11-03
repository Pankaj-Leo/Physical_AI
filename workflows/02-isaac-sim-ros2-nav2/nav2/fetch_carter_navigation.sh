#!/usr/bin/env bash
# Wizard step 5: fetch NVIDIA's carter_navigation package (occupancy map + Nav2 params) and fix the map paths.
# Not executed here. Run on the host or inside the nav2 container from the workflow folder:
#   bash nav2/fetch_carter_navigation.sh
#
# The wizard infographic shows "git clone https://github.com/NVIDIA/carter_navigation.git". The package is
# published inside NVIDIA's IsaacSim-ros_workspaces repository (verify: repo URL, branch, and the paths below for
# your Isaac Sim release). Override with REPO=... BRANCH=... SUBDIR=... if they differ.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-https://github.com/isaac-sim/IsaacSim-ros_workspaces.git}"
BRANCH="${BRANCH:-main}"
SUBDIR="${SUBDIR:-jazzy_ws/src/navigation/carter_navigation}"
WORK="${WORK:-/tmp/isaac_ros_ws_src}"

if [ ! -d "$WORK/.git" ]; then
  git clone --depth 1 --branch "$BRANCH" "$REPO" "$WORK"
fi
PKG="$WORK/$SUBDIR"
ls "$PKG/maps" "$PKG/params"

# Rewrite the map YAML so map_server finds the image inside the container (/ws is this folder).
python3 "$HERE/fix_map_yaml.py" "$PKG/maps/carter_warehouse_navigation.yaml" \
  --out-dir "$HERE/maps" --container-dir /ws/nav2/maps

# Keep NVIDIA's params next to ours for comparison; bringup.launch.py uses nav2_params.yaml unless params:= is given.
cp "$PKG"/params/*.yaml "$HERE/maps/" 2>/dev/null || true
echo "map ready: $HERE/maps/carter_warehouse_navigation.yaml"
