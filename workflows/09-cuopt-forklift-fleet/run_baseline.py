"""Run the CPU baseline on the two scenarios and write results/cpu_baseline.json (runs on CPU).

    python run_baseline.py            # writes results/cpu_baseline.json and results/payload_screenshot.json

Numbers come from our time model (edge length / 1.5 m/s + 10 s per job), not from Isaac Sim physics, so they are not
comparable to the wizard's measured durations (1:05 for 2 forklifts / 6 jobs in physics mode; ~5 min estimated for
5 forklifts / 20 targets in kinematic mode). Solver wall times are from this container and vary by machine.
"""

from __future__ import annotations

import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cuopt_payload
import greedy_solver as gs
import warehouse_graph as wg

RESULTS = Path(__file__).resolve().parent / "results"


def lower_bound_makespan(p: gs.Problem) -> float:
    """No schedule can beat the longest single job done alone by its best-placed forklift."""
    return max(
        min(p.travel(p.starts[v], t) + p.service[k] + p.travel(t, p.ends[v]) for v in range(len(p.starts)))
        for k, t in enumerate(p.task_nodes)
    )


def run(payload: dict, objective: str, local_search: bool) -> dict:
    t0 = time.perf_counter()
    r = gs.solve(payload, objective, local_search)["response"]["solver_response"]
    ms = (time.perf_counter() - t0) * 1000
    return {
        "makespan_s": r["makespan_s"],
        "total_route_time_s": r["solution_cost"],
        "jobs_per_vehicle": r["jobs_per_vehicle"],
        "idle_vehicles": sum(1 for n in r["jobs_per_vehicle"].values() if n == 0),
        "solver_wall_ms": round(ms, 1),
    }


def scenario_block(sc: wg.Scenario, exact: bool) -> dict:
    g = wg.build_graph(sc.layout)
    payload = cuopt_payload.build_payload(sc, g)
    assert cuopt_payload.check_payload(payload) == []
    p = gs.parse(payload)
    block = {
        "scenario": sc.name,
        "notes": sc.notes,
        "grid": {"cols": sc.layout.cols, "rows": sc.layout.rows, "spacing_m": sc.layout.spacing},
        "waypoints": g.n,
        "directed_edges": len(g.edges),
        "obstacle_cells": len(sc.layout.obstacles),
        "no_entry_zones": len(sc.layout.no_entry),
        "forklifts": len(sc.vehicle_ids),
        "jobs": len(sc.targets),
        "capacity_per_forklift": sc.capacities[0],
        "makespan_lower_bound_s": round(lower_bound_makespan(p), 3),
        "results": {
            f"{obj}{'+local_search' if ls else '_insertion_only'}": run(payload, obj, ls)
            for obj in ("makespan", "total")
            for ls in (False, True)
        },
    }
    if exact:
        for obj in ("makespan", "total"):
            r = gs.response(p, gs.exact_solve(p, obj), obj)["response"]["solver_response"]
            block["results"][f"{obj}_exact"] = {
                "makespan_s": r["makespan_s"],
                "total_route_time_s": r["solution_cost"],
                "jobs_per_vehicle": r["jobs_per_vehicle"],
            }
    return block


def gap_study(n_instances: int = 20, n_jobs: int = 7, n_vehicles: int = 2) -> dict:
    """Heuristic vs brute-force optimum on small random floors (makespan objective)."""
    gaps = {"insertion_only": [], "insertion+local_search": []}
    for seed in range(n_instances):
        sc = wg.random_scenario(
            f"gap_{seed}",
            cols=11,
            rows=9,
            n_obstacles=8,
            n_targets=n_jobs,
            n_vehicles=n_vehicles,
            capacity=n_jobs,
            seed=1000 + seed,
        )
        payload = cuopt_payload.build_payload(sc)
        p = gs.parse(payload)
        best = gs.response(p, gs.exact_solve(p, "makespan"), "makespan")["response"]["solver_response"]["makespan_s"]
        for key, ls in (("insertion_only", False), ("insertion+local_search", True)):
            got = gs.solve(payload, "makespan", ls)["response"]["solver_response"]["makespan_s"]
            gaps[key].append(100 * (got - best) / best)
    return {
        "instances": n_instances,
        "jobs": n_jobs,
        "forklifts": n_vehicles,
        "grid": "11x9 at 4 m, 8 random obstacle cells",
        **{
            k: {
                "mean_gap_pct": round(statistics.mean(v), 2),
                "max_gap_pct": round(max(v), 2),
                "optimal_count": sum(1 for x in v if x < 1e-6),
            }
            for k, v in gaps.items()
        },
    }


def main() -> int:
    random.seed(0)
    RESULTS.mkdir(exist_ok=True)
    small, large = wg.screenshot_scenario(), wg.large_scenario()
    out = {
        "what": "CPU baseline for the cuOpt forklift routing request (not cuOpt, not Isaac Sim)",
        "time_model": {
            "speed_mps": small.speed_mps,
            "service_time_s_per_job": small.service_time_s,
            "route": "depot -> jobs -> depot over shortest waypoint paths",
            "source_of_speed": "forklift-fleet-live.jpg (Forklift_C DRIVING 1.50 m/s)",
            "service_time": "our assumption",
        },
        "environment": {"python": platform.python_version(), "machine": platform.machine()},
        "scenarios": [scenario_block(small, exact=True), scenario_block(large, exact=False)],
        "gap_study": gap_study(),
        "wizard_reference_not_comparable": {
            "2_forklifts_6_jobs_physics_duration": "1:05 (forklift-fleet-complete.jpg; Wizard #2 promo)",
            "5_forklifts_20_targets_kinematic_estimate": "about 5 min (Wizard #2 promo, caption)",
        },
    }
    (RESULTS / "cpu_baseline.json").write_text(json.dumps(out, indent=2) + "\n")
    (RESULTS / "payload_screenshot.json").write_text(json.dumps(cuopt_payload.build_payload(small)) + "\n")
    for s in out["scenarios"]:
        print(f"\n{s['scenario']}  (lower bound on makespan {s['makespan_lower_bound_s']} s)")
        for name, r in s["results"].items():
            print(
                f"  {name:28s} makespan {r['makespan_s']:8.1f} s  total {r['total_route_time_s']:8.1f} s  "
                f"jobs {list(r['jobs_per_vehicle'].values())}"
            )
    print("\ngap study:", json.dumps(out["gap_study"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
