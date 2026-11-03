"""Build the cuOpt routing request for a forklift scenario (runs on CPU).

The sources show the cuOpt microservice in its own container on port 8000 and the Isaac Sim cuOpt extension's
"Simple cost matrix" example (Wizard #1 talk), and they describe the inputs in words: a waypoint network, the fleet,
the targets, no-entry zones (Wizard #2 talk). **They do not show the request schema.** The field names below follow
the self-hosted cuOpt server's waypoint-graph request as we know it; verify against your cuOpt server version before
sending. ``greedy_solver.solve`` reads exactly this dict, so the CPU baseline and cuOpt see the same problem.

Edge weights are travel times in seconds (edge length / forklift speed); service time is per task.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from warehouse_graph import Scenario, WaypointGraph, build_graph

SCHEMA_NOTE = "verify against your cuOpt server version: field names are not shown in the sources"


def build_payload(sc: Scenario, g: WaypointGraph | None = None, *, time_limit_s: float = 5.0) -> dict[str, Any]:
    g = g or build_graph(sc.layout)
    weights = [round(length / sc.speed_mps, 6) for length in g.lengths]
    depots = [g.node(d) for d in sc.depots]
    tasks = [g.node(t) for t in sc.targets]
    return {
        "cost_waypoint_graph_data": {
            "waypoint_graph": {"0": {"offsets": list(g.offsets), "edges": list(g.edges), "weights": weights}}
        },
        "fleet_data": {
            "vehicle_ids": list(sc.vehicle_ids),
            "vehicle_locations": [[d, d] for d in depots],  # start and end at the depot
            "capacities": [list(sc.capacities)],  # one capacity dimension: jobs
        },
        "task_data": {
            "task_ids": [f"job_{i + 1}" for i in range(len(tasks))],
            "task_locations": tasks,
            "demand": [[1] * len(tasks)],
            "service_times": [sc.service_time_s] * len(tasks),
        },
        "solver_config": {"time_limit": time_limit_s},
    }


def check_payload(p: dict[str, Any]) -> list[str]:
    """Structural checks that catch the usual mistakes before a request reaches the server."""
    problems: list[str] = []
    wg = p["cost_waypoint_graph_data"]["waypoint_graph"]["0"]
    offsets, edges, weights = wg["offsets"], wg["edges"], wg["weights"]
    n = len(offsets) - 1
    if offsets[0] != 0 or offsets[-1] != len(edges) or any(a > b for a, b in pairwise(offsets)):
        problems.append("offsets must start at 0, end at len(edges), and not decrease")
    if len(weights) != len(edges):
        problems.append("weights and edges differ in length")
    if any(not 0 <= e < n for e in edges):
        problems.append("edge target out of range")
    if any(w <= 0 for w in weights):
        problems.append("non-positive edge weight")
    fleet, tasks = p["fleet_data"], p["task_data"]
    nv = len(fleet["vehicle_locations"])
    if any(not 0 <= x < n for pair in fleet["vehicle_locations"] for x in pair):
        problems.append("vehicle location out of range")
    if any(not 0 <= t < n for t in tasks["task_locations"]):
        problems.append("task location out of range")
    if any(len(c) != nv for c in fleet["capacities"]):
        problems.append("capacities must have one entry per vehicle per dimension")
    nt = len(tasks["task_locations"])
    if any(len(d) != nt for d in tasks["demand"]) or len(tasks["service_times"]) != nt:
        problems.append("demand / service_times must have one entry per task")
    if len(fleet["capacities"]) != len(tasks["demand"]):
        problems.append("capacity and demand dimensions differ")
    if fleet["capacities"] and tasks["demand"] and sum(fleet["capacities"][0]) < sum(tasks["demand"][0]):
        problems.append("total capacity is below total demand: some tasks cannot be served")
    return problems
