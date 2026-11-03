"""pytest config for this folder: skip the Kit-side tests, which need ``omni.kit.test`` and run inside Kit only."""

collect_ignore_glob = ["exts/*/physicalai/atlas/tools/tests/*"]
