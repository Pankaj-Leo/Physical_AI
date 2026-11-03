"""CPU tests: waypoint graph, cuOpt payload, baseline solver, forklift graph, and the committed results."""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import cuopt_payload  # noqa: E402
import forklift_graph_spec as fk  # noqa: E402
import greedy_solver as gs  # noqa: E402
import ogspec  # noqa: E402
import run_baseline  # noqa: E402
import warehouse_graph as wg  # noqa: E402


@pytest.fixture(scope="module")
def small():
    sc = wg.screenshot_scenario()
    return sc, wg.build_graph(sc.layout), cuopt_payload.build_payload(sc)


# --- waypoint graph -------------------------------------------------------------------------------------------------


def test_graph_is_symmetric_and_grid_spaced(small):
    _, g, _ = small
    pairs = {(i, j) for i in range(g.n) for j, _ in g.neighbors(i)}
    assert all((j, i) in pairs for i, j in pairs)
    for i in range(g.n):
        for j, length in g.neighbors(i):
            assert math.dist(g.xy[i], g.xy[j]) == pytest.approx(length) == pytest.approx(4.0)


def test_no_entry_zone_and_obstacles_have_no_nodes(small):
    sc, g, _ = small
    zone = sc.layout.no_entry[0]
    assert not any(zone.contains(x, y) for x, y in g.xy)
    assert not set(sc.layout.obstacles) & set(g.cells)
    assert g.n == sc.layout.cols * sc.layout.rows - len(sc.layout.obstacles) - sum(
        1 for r in range(sc.layout.rows) for c in range(sc.layout.cols) if zone.contains(*sc.layout.xy((c, r)))
    )


def test_screenshot_coordinates_lie_on_the_grid():
    layout = wg.screenshot_scenario().layout
    xs = {layout.xy((c, 0))[0] for c in range(layout.cols)}
    ys = {layout.xy((0, r))[1] for r in range(layout.rows)}
    for x, y in [(-16, 4), (12, -12)]:  # targets printed in forklift-fleet-live.jpg
        assert x in xs and y in ys


@pytest.mark.parametrize("make", [wg.screenshot_scenario, wg.large_scenario])
def test_every_depot_and_target_is_reachable(make):
    sc = make()
    g = wg.build_graph(sc.layout)
    reach = wg.component(g, g.node(sc.depots[0]))
    assert all(g.node(c) in reach for c in (*sc.depots, *sc.targets))


def test_large_scenario_matches_the_talk():
    sc = wg.large_scenario()
    assert (sc.layout.cols, len(sc.layout.obstacles), len(sc.targets), len(sc.vehicle_ids)) == (30, 20, 20, 5)


# --- payload ---------------------------------------------------------------------------------------------------------


def test_payload_is_well_formed(small):
    sc, g, p = small
    assert cuopt_payload.check_payload(p) == []
    wgd = p["cost_waypoint_graph_data"]["waypoint_graph"]["0"]
    assert len(wgd["offsets"]) == g.n + 1
    assert wgd["weights"][0] == pytest.approx(4.0 / sc.speed_mps)
    assert p["fleet_data"]["capacities"] == [[3, 3]]
    assert len(p["task_data"]["task_locations"]) == 6
    json.dumps(p)  # serialisable as the request body


def test_check_payload_catches_errors(small):
    _, _, p = small
    bad = json.loads(json.dumps(p))
    bad["cost_waypoint_graph_data"]["waypoint_graph"]["0"]["edges"][0] = 10_000
    bad["task_data"]["service_times"].pop()
    bad["fleet_data"]["capacities"] = [[1, 1]]
    problems = " | ".join(cuopt_payload.check_payload(bad))
    assert "edge target out of range" in problems
    assert "service_times" in problems
    assert "below total demand" in problems


# --- solver ----------------------------------------------------------------------------------------------------------


def _check_solution(payload, resp):
    p = gs.parse(payload)
    r = resp["response"]["solver_response"]
    served = list(itertools.chain.from_iterable(v["task_id"] for v in r["vehicle_data"].values()))
    assert sorted(served) == sorted(p.task_ids)  # every job exactly once
    for vid, data in r["vehicle_data"].items():
        v = p.vehicle_ids.index(vid)
        assert len(data["task_id"]) <= p.capacity[v]
        route = data["route"]
        assert route[0] == p.starts[v] and route[-1] == p.ends[v]
        for a, b in itertools.pairwise(route):
            assert b in {j for j, _ in p.adj[a]}, "consecutive waypoints must share an edge"
        assert data["type"].count("Task") == len(data["task_id"])
        assert data["arrival_stamp"] == sorted(data["arrival_stamp"])
    return r


@pytest.mark.parametrize("objective", ["makespan", "total"])
@pytest.mark.parametrize("make", [wg.screenshot_scenario, wg.large_scenario])
def test_solutions_are_feasible(make, objective):
    payload = cuopt_payload.build_payload(make())
    r = _check_solution(payload, gs.solve(payload, objective))
    assert r["makespan_s"] == pytest.approx(max(r["route_times_s"].values()))


def test_route_times_are_consistent_with_arrival_stamps(small):
    _, _, payload = small
    p = gs.parse(payload)
    r = gs.solve(payload)["response"]["solver_response"]
    for vid, data in r["vehicle_data"].items():
        # The last stamp is the return to the depot; stamps after a job include its service time.
        assert data["arrival_stamp"][-1] == pytest.approx(r["route_times_s"][vid], abs=1e-3)
        travel = sum(next(w for j, w in p.adj[a] if j == b) for a, b in itertools.pairwise(data["route"]))
        service = sum(p.service[p.task_ids.index(t)] for t in data["task_id"])
        assert travel + service == pytest.approx(r["route_times_s"][vid], abs=1e-3)


def test_heuristic_matches_brute_force_on_screenshot_case(small):
    _, _, payload = small
    p = gs.parse(payload)
    exact = gs.response(p, gs.exact_solve(p, "makespan"), "makespan")["response"]["solver_response"]
    heur = gs.solve(payload, "makespan")["response"]["solver_response"]
    assert heur["makespan_s"] >= exact["makespan_s"] - 1e-6  # brute force is a true optimum
    assert heur["makespan_s"] == pytest.approx(exact["makespan_s"])


def test_local_search_never_hurts():
    payload = cuopt_payload.build_payload(wg.large_scenario())
    for obj in ("makespan", "total"):
        base = gs.score(
            list(gs.solve(payload, obj, False)["response"]["solver_response"]["route_times_s"].values()), obj
        )
        ls = gs.score(list(gs.solve(payload, obj, True)["response"]["solver_response"]["route_times_s"].values()), obj)
        assert ls <= base


def test_total_cost_objective_can_idle_a_forklift():
    # The wizard talk: in the 20-target run cuOpt "decided that these forklifts ... we don't need to use".
    payload = cuopt_payload.build_payload(wg.large_scenario())
    r = gs.solve(payload, "total")["response"]["solver_response"]
    assert 0 in r["jobs_per_vehicle"].values()
    mk = gs.solve(payload, "makespan")["response"]["solver_response"]
    assert mk["makespan_s"] < r["makespan_s"] and mk["solution_cost"] > r["solution_cost"]


def test_makespan_respects_lower_bound():
    payload = cuopt_payload.build_payload(wg.large_scenario())
    p = gs.parse(payload)
    r = gs.solve(payload, "makespan")["response"]["solver_response"]
    assert r["makespan_s"] >= run_baseline.lower_bound_makespan(p) - 1e-6


def test_committed_results_reproduce():
    committed = json.loads((HERE / "results" / "cpu_baseline.json").read_text())
    for block in committed["scenarios"]:
        sc = wg.screenshot_scenario() if block["scenario"].startswith("screenshot") else wg.large_scenario()
        fresh = run_baseline.scenario_block(sc, exact=False)
        for name, r in fresh["results"].items():
            assert r["makespan_s"] == block["results"][name]["makespan_s"], name
            assert r["total_route_time_s"] == block["results"][name]["total_route_time_s"], name


# --- forklift graph --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ros", [False, True])
def test_forklift_graph_is_consistent(ros):
    assert ogspec.validate(fk.ackermann_graph("/World/Forklift_B", ros_input=ros)) == []


def test_forklift_graph_matches_dashboard():
    g = fk.ackermann_graph("/World/Forklift_B")
    ctl = g.node("ackermann_controller").values
    assert ctl["invertSteering"] is True and ctl["maxWheelRotation"] == pytest.approx(1.2)
    into = {(e.dst, e.dst_attr): (e.src, e.src_attr) for e in g.edges}
    assert into["ackermann_controller", "inputs:dt"] == ("on_playback_tick", "outputs:deltaSeconds")
    assert into["steering_controller", "inputs:positionCommand"] == ("ackermann_controller", "outputs:wheelAngles")
    assert into["drive_controller", "inputs:velocityCommand"] == (
        "ackermann_controller",
        "outputs:wheelRotationVelocity",
    )
    # Two steering values ([1.20, 0.87]) and four wheel velocities ([0.00, 0.00, 0.00, 0.00]) in the screenshot.
    assert g.node("steer_names").values["arraySize"] == 2
    assert g.node("drive_names").values["arraySize"] == 4


def test_ackermann_geometry_reproduces_the_screenshot_pair():
    ratio = fk.implied_track_ratio(1.20, 0.87)
    assert ratio == pytest.approx(0.455, abs=1e-3)
    # Find the bicycle angle whose inner wheel is 1.20 rad, then check the outer wheel comes out at 0.87 rad.
    lo, hi = 0.01, 1.2
    for _ in range(60):
        mid = (lo + hi) / 2
        inner, _ = fk.ackermann_wheel_angles(mid, 1.0, ratio)
        lo, hi = (mid, hi) if inner < 1.20 else (lo, mid)
    _, outer = fk.ackermann_wheel_angles(lo, 1.0, ratio)
    assert outer == pytest.approx(0.87, abs=1e-6)
    assert ratio <= fk.TRACK_WIDTH_M / fk.WHEEL_BASE_M


def test_ogspec_copy_matches_workflow_02():
    other = HERE.parent / "02-isaac-sim-ros2-nav2" / "ogspec.py"
    if not other.exists():
        pytest.skip("workflow 02 not present")
    assert other.read_text() == (HERE / "ogspec.py").read_text()


def test_client_imports_on_cpu():
    import cuopt_client

    assert cuopt_client.DEFAULT_URL.endswith(":8000")  # Wizard #1: "I think the port was 8,000"
