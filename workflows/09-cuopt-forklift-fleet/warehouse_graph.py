"""Waypoint graph for a forklift floor: a grid of nodes, minus obstacle cells and no-entry zones.

This is the "network of nodes and connections that cuOpt is going to use to calculate the route" from the Forklift
Fleet wizard (talk), drawn as white dots on the floor in ``forklift-fleet-live.jpg``. Obstacles (boxes) delete nodes;
a no-entry zone (the wizard's red box) deletes every node inside it, so no edge can cross it. Runs on CPU.

Coordinates: meters, Z-up floor plane, grid centered on the origin. The screenshot shows targets such as (-16, 4)
and (12, -12), consistent with a 4 m spacing; the grid sizes below are our reading of the screenshot, not given in
the source.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

Cell = tuple[int, int]  # (col, row)


@dataclass(frozen=True)
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


@dataclass(frozen=True)
class Layout:
    cols: int
    rows: int
    spacing: float = 4.0
    obstacles: frozenset[Cell] = frozenset()
    no_entry: tuple[Rect, ...] = ()

    def xy(self, cell: Cell) -> tuple[float, float]:
        c, r = cell
        return ((c - (self.cols - 1) / 2) * self.spacing, (r - (self.rows - 1) / 2) * self.spacing)

    def free(self, cell: Cell) -> bool:
        c, r = cell
        if not (0 <= c < self.cols and 0 <= r < self.rows) or cell in self.obstacles:
            return False
        x, y = self.xy(cell)
        return not any(z.contains(x, y) for z in self.no_entry)


@dataclass
class WaypointGraph:
    """CSR adjacency (the layout cuOpt's waypoint graph uses): edges of node i are edges[offsets[i]:offsets[i+1]]."""

    cells: list[Cell]
    xy: list[tuple[float, float]]
    offsets: list[int]
    edges: list[int]
    lengths: list[float]  # meters
    index: dict[Cell, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.cells)

    def neighbors(self, i: int) -> list[tuple[int, float]]:
        lo, hi = self.offsets[i], self.offsets[i + 1]
        return list(zip(self.edges[lo:hi], self.lengths[lo:hi], strict=True))

    def node(self, cell: Cell) -> int:
        return self.index[cell]


def build_graph(layout: Layout) -> WaypointGraph:
    cells = [(c, r) for r in range(layout.rows) for c in range(layout.cols) if layout.free((c, r))]
    index = {cell: i for i, cell in enumerate(cells)}
    offsets, edges, lengths = [0], [], []
    for c, r in cells:
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (c + dc, r + dr)
            if nb in index:
                edges.append(index[nb])
                lengths.append(layout.spacing)
        offsets.append(len(edges))
    return WaypointGraph(cells, [layout.xy(cell) for cell in cells], offsets, edges, lengths, index)


def dijkstra(neighbors, n: int, src: int) -> tuple[list[float], list[int]]:
    """Shortest paths from ``src``; ``neighbors(i)`` yields (j, cost). Returns (dist, prev)."""
    dist = [math.inf] * n
    prev = [-1] * n
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue
        for j, w in neighbors(i):
            nd = d + w
            if nd < dist[j]:
                dist[j], prev[j] = nd, i
                heapq.heappush(heap, (nd, j))
    return dist, prev


def walk_back(prev: list[int], src: int, dst: int) -> list[int]:
    path = [dst]
    while path[-1] != src:
        if prev[path[-1]] < 0:
            raise ValueError(f"no path {src} -> {dst}")
        path.append(prev[path[-1]])
    return path[::-1]


def component(g: WaypointGraph, start: int) -> set[int]:
    seen, stack = {start}, [start]
    while stack:
        for j, _ in g.neighbors(stack.pop()):
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return seen


@dataclass(frozen=True)
class Scenario:
    name: str
    layout: Layout
    vehicle_ids: tuple[str, ...]
    depots: tuple[Cell, ...]  # start = end ("depot" in the dashboard table)
    capacities: tuple[int, ...]  # max jobs per forklift (demand 1 per job)
    targets: tuple[Cell, ...]
    speed_mps: float = 1.5  # forklift-fleet-live.jpg: Forklift_C DRIVING at 1.50 m/s
    service_time_s: float = 10.0  # our assumption: align (reversing at 0.40 m/s in the screenshot) + fork time
    notes: str = ""


def screenshot_scenario() -> Scenario:
    """Two forklifts, six jobs, one no-entry zone: our approximation of the default-plus-two-targets run."""
    layout = Layout(
        cols=11,
        rows=9,
        spacing=4.0,
        obstacles=frozenset({(4, 3), (4, 4), (4, 5), (3, 6), (7, 6), (7, 2)}),  # tan boxes, positions approximate
        no_entry=(Rect(-2.0, -2.0, 6.0, 6.0),),  # the large red box
    )
    return Scenario(
        name="screenshot_2_forklifts_6_jobs",
        layout=layout,
        vehicle_ids=("Forklift_B", "Forklift_C"),
        depots=((1, 4), (9, 1)),
        capacities=(3, 3),  # dashboard: JOBS 3/3 for each forklift
        targets=((1, 7), (2, 1), (3, 2), (6, 7), (9, 6), (8, 0)),
        notes="11x9 waypoints at 4 m read off forklift-fleet-*.jpg; box and target cells approximate",
    )


def random_scenario(
    name: str, *, cols: int, rows: int, n_obstacles: int, n_targets: int, n_vehicles: int, capacity: int, seed: int
) -> Scenario:
    """The wizard's random generator: columns, rows, obstacles, targets, forklifts (talk: 30 cols, 20, 20)."""
    rng = random.Random(seed)
    all_cells = [(c, r) for r in range(rows) for c in range(cols)]
    depots = tuple((0, round(r)) for r in [(i + 0.5) * rows / n_vehicles for i in range(n_vehicles)])
    for _ in range(100):
        pool = [cell for cell in all_cells if cell not in depots]
        obstacles = frozenset(rng.sample(pool, n_obstacles))
        layout = Layout(cols, rows, 4.0, obstacles)
        g = build_graph(layout)
        reach = component(g, g.node(depots[0]))
        if not all(g.index.get(d) in reach for d in depots):
            continue
        free = [g.cells[i] for i in sorted(reach) if g.cells[i] not in depots]
        targets = tuple(rng.sample(free, n_targets))
        return Scenario(
            name=name,
            layout=layout,
            vehicle_ids=tuple(f"Forklift_{i + 1}" for i in range(n_vehicles)),
            depots=depots,
            capacities=(capacity,) * n_vehicles,
            targets=targets,
            notes=f"random layout, seed {seed}; rows and depot placement are our choice",
        )
    raise RuntimeError("could not place a connected scenario")


def large_scenario() -> Scenario:
    # Promo transcript: "20 targets and five forklifts"; long talk: 30 columns, 20 obstacles, 20 targets.
    return random_scenario(
        "random_5_forklifts_20_targets",
        cols=30,
        rows=20,
        n_obstacles=20,
        n_targets=20,
        n_vehicles=5,
        capacity=10,
        seed=7,
    )
