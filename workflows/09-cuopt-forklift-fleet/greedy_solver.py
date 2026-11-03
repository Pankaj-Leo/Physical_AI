"""CPU baseline for the cuOpt request: shortest paths + cheapest insertion + local search (runs on CPU, tested).

It reads the same payload ``cuopt_payload.build_payload`` writes for cuOpt and answers in a cuOpt-like shape, so the
pipeline (scene -> waypoint graph -> request -> routes -> per-forklift waypoint lists) runs end to end without a GPU.
It is a baseline, not cuOpt: cuOpt is a GPU solver with its own heuristics and objectives.

Model: a forklift leaves its depot, visits its jobs (service time each), returns to the depot. Route time is the sum
of shortest-path travel times between stops plus service times. Two objectives:
- ``makespan``: minimise the time the last forklift gets home, then total time (what the dashboard's "duration" shows);
- ``total``: minimise the sum of route times, then makespan (a cost-minimising objective; it may leave a forklift idle,
  as the wizard talk observed cuOpt do in the 20-target run).
``exact_solve`` enumerates every assignment and order; use it for small cases to measure the heuristic's gap.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

from warehouse_graph import dijkstra, walk_back


@dataclass
class Problem:
    n_nodes: int
    adj: list[list[tuple[int, float]]]
    vehicle_ids: list[str]
    starts: list[int]
    ends: list[int]
    capacity: list[int]
    task_ids: list[str]
    task_nodes: list[int]
    demand: list[int]
    service: list[float]
    # travel time and predecessor trees from every key node (depots and task nodes)
    dist: dict[int, list[float]]
    prev: dict[int, list[int]]

    def travel(self, a: int, b: int) -> float:
        return self.dist[a][b]


def parse(payload: dict[str, Any]) -> Problem:
    wg = payload["cost_waypoint_graph_data"]["waypoint_graph"]["0"]
    offsets, edges, weights = wg["offsets"], wg["edges"], wg["weights"]
    n = len(offsets) - 1
    adj = [
        list(zip(edges[offsets[i] : offsets[i + 1]], weights[offsets[i] : offsets[i + 1]], strict=True))
        for i in range(n)
    ]
    fleet, tasks = payload["fleet_data"], payload["task_data"]
    locs = fleet["vehicle_locations"]
    starts, ends = [a for a, _ in locs], [b for _, b in locs]
    task_nodes = list(tasks["task_locations"])
    keys = set(starts) | set(ends) | set(task_nodes)
    dist, prev = {}, {}
    for k in keys:
        dist[k], prev[k] = dijkstra(lambda i: adj[i], n, k)
    unreachable = [t for t in task_nodes for s in starts if math.isinf(dist[s][t])]
    if unreachable:
        raise ValueError(f"tasks unreachable from some depot: {sorted(set(unreachable))}")
    return Problem(
        n_nodes=n,
        adj=adj,
        vehicle_ids=list(fleet.get("vehicle_ids") or [str(i) for i in range(len(locs))]),
        starts=starts,
        ends=ends,
        capacity=list(fleet["capacities"][0]) if fleet.get("capacities") else [math.inf] * len(locs),
        task_ids=list(tasks.get("task_ids") or [str(i) for i in range(len(task_nodes))]),
        task_nodes=task_nodes,
        demand=list(tasks["demand"][0]) if tasks.get("demand") else [0] * len(task_nodes),
        service=list(tasks.get("service_times") or [0.0] * len(task_nodes)),
        dist=dist,
        prev=prev,
    )


def route_time(p: Problem, v: int, route: list[int]) -> float:
    if not route:
        return 0.0  # an unused forklift stays at its depot
    t, at = 0.0, p.starts[v]
    for task in route:
        t += p.travel(at, p.task_nodes[task]) + p.service[task]
        at = p.task_nodes[task]
    return t + p.travel(at, p.ends[v])


def score(times: list[float], objective: str) -> tuple[float, float]:
    mk, tot = max(times, default=0.0), sum(times)
    return (round(mk, 9), round(tot, 9)) if objective == "makespan" else (round(tot, 9), round(mk, 9))


def _load(p: Problem, route: list[int]) -> int:
    return sum(p.demand[t] for t in route)


def cheapest_insertion(p: Problem, objective: str) -> list[list[int]]:
    routes: list[list[int]] = [[] for _ in p.starts]
    times = [0.0] * len(routes)
    todo = list(range(len(p.task_nodes)))
    while todo:
        best = None
        for task in todo:
            for v, route in enumerate(routes):
                if _load(p, route) + p.demand[task] > p.capacity[v]:
                    continue
                for pos in range(len(route) + 1):
                    cand = [*route[:pos], task, *route[pos:]]
                    t_new = route_time(p, v, cand)
                    key = score([*times[:v], t_new, *times[v + 1 :]], objective)
                    if best is None or key < best[0]:
                        best = (key, task, v, cand, t_new)
        if best is None:
            raise ValueError("capacity too small for the remaining tasks")
        _, task, v, cand, t_new = best
        routes[v], times[v] = cand, t_new
        todo.remove(task)
    return routes


def improve(p: Problem, routes: list[list[int]], objective: str, max_rounds: int = 200) -> list[list[int]]:
    """First-improvement local search: relocate a job (within or across forklifts) and 2-opt within a route."""
    routes = [list(r) for r in routes]
    times = [route_time(p, v, r) for v, r in enumerate(routes)]
    current = score(times, objective)
    for _ in range(max_rounds):
        improved = False
        # relocate
        for a, b in itertools.product(range(len(routes)), repeat=2):
            for i, task in enumerate(routes[a]):
                if a != b and _load(p, routes[b]) + p.demand[task] > p.capacity[b]:
                    continue
                ra = routes[a][:i] + routes[a][i + 1 :]
                for pos in range(len(routes[b]) + (0 if a == b else 1)):
                    base = ra if a == b else routes[b]
                    rb = [*base[:pos], task, *base[pos:]]
                    trial = [list(r) for r in routes]
                    trial[a] = ra
                    trial[b] = rb
                    t = [route_time(p, v, r) for v, r in enumerate(trial)]
                    if score(t, objective) < current:
                        routes, times, current, improved = trial, t, score(t, objective), True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # 2-opt inside each route
        for v, r in enumerate(routes):
            for i, j in itertools.combinations(range(len(r)), 2):
                cand = r[:i] + r[i : j + 1][::-1] + r[j + 1 :]
                t = [*times[:v], route_time(p, v, cand), *times[v + 1 :]]
                if score(t, objective) < current:
                    routes[v], times, current, improved = cand, t, score(t, objective), True
                    break
            if improved:
                break
        if not improved:
            break
    return routes


def exact_solve(p: Problem, objective: str) -> list[list[int]]:
    """Brute force over assignments and orders. Only for small cases (about 8 jobs, 2-3 forklifts)."""
    n_t, n_v = len(p.task_nodes), len(p.starts)
    if n_v**n_t > 20000:
        raise ValueError("too large for brute force")
    best_order: dict[tuple[int, tuple[int, ...]], tuple[float, list[int]]] = {}

    def best_route(v: int, tasks: tuple[int, ...]) -> tuple[float, list[int]]:
        key = (v, tasks)
        if key not in best_order:
            best_order[key] = min(
                ((route_time(p, v, list(perm)), list(perm)) for perm in itertools.permutations(tasks)),
                key=lambda x: x[0],
            )
        return best_order[key]

    best = None
    for assign in itertools.product(range(n_v), repeat=n_t):
        groups = [tuple(t for t in range(n_t) if assign[t] == v) for v in range(n_v)]
        if any(sum(p.demand[t] for t in g) > p.capacity[v] for v, g in enumerate(groups)):
            continue
        rs = [best_route(v, g) for v, g in enumerate(groups)]
        key = score([t for t, _ in rs], objective)
        if best is None or key < best[0]:
            best = (key, [r for _, r in rs])
    if best is None:
        raise ValueError("no feasible assignment")
    return best[1]


def expand(p: Problem, v: int, route: list[int]) -> dict[str, Any]:
    """Waypoint-level route with arrival times, in a cuOpt-like layout (verify field names against cuOpt)."""
    if not route:
        return {"task_id": [], "route": [p.starts[v]], "type": ["Depot"], "arrival_stamp": [0.0]}
    nodes, types, stamps, task_ids = [p.starts[v]], ["Depot"], [0.0], []
    t, at = 0.0, p.starts[v]
    stops = [(p.task_nodes[k], k) for k in route] + [(p.ends[v], None)]
    for node, k in stops:
        path = walk_back(p.prev[at], at, node)
        for a, b in itertools.pairwise(path):
            t += next(w for j, w in p.adj[a] if j == b)
            nodes.append(b)
            types.append("w")
            stamps.append(round(t, 3))
        types[-1] = "Task" if k is not None else "Depot"
        if k is not None:
            task_ids.append(p.task_ids[k])
            t += p.service[k]
        at = node
    return {"task_id": task_ids, "route": nodes, "type": types, "arrival_stamp": stamps}


def solve(payload: dict[str, Any], objective: str = "makespan", local_search: bool = True) -> dict[str, Any]:
    p = parse(payload)
    routes = cheapest_insertion(p, objective)
    if local_search:
        routes = improve(p, routes, objective)
    return response(p, routes, objective)


def response(p: Problem, routes: list[list[int]], objective: str) -> dict[str, Any]:
    times = [route_time(p, v, r) for v, r in enumerate(routes)]
    vehicle_data = {p.vehicle_ids[v]: expand(p, v, r) for v, r in enumerate(routes) if r}
    return {
        "response": {
            "solver_response": {
                "status": 0,
                "num_vehicles": len(vehicle_data),
                "solution_cost": round(sum(times), 3),
                "vehicle_data": vehicle_data,
                # extras for this baseline (not cuOpt fields)
                "objective": objective,
                "makespan_s": round(max(times), 3),
                "route_times_s": {p.vehicle_ids[v]: round(t, 3) for v, t in enumerate(times)},
                "jobs_per_vehicle": {p.vehicle_ids[v]: len(r) for v, r in enumerate(routes)},
            }
        }
    }
