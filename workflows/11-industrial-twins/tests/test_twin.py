"""CPU tests: the Warp solver against an analytic case and energy balance, the correlations, the surrogate."""

from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
wp = pytest.importorskip("warp")


def _load(name: str):
    key = f"wf11_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


hs = _load("heat_solver")
sur = _load("surrogate")
twin = _load("run_twin")


def test_slab_matches_analytic():
    """Uniform flux q into the bottom of a slab, convection h on top, adiabatic sides: theta is linear in y and
    the cell-centred FV solution is exact, theta_bottom = q (1/h + (tb - dx/2)/k)."""
    nx, ny, dx, k, h, q = 8, 20, 0.5e-3, hs.K_AL, 12.0, 4000.0
    kinds = np.full((ny, nx, 4), hs.SOLID, dtype=np.int32)
    kinds[:, 0, 0] = hs.ADIABATIC
    kinds[:, -1, 1] = hs.ADIABATIC
    kinds[0, :, 2] = hs.SOURCE
    kinds[-1, :, 3] = hs.OPEN
    n = nx * ny
    with wp.ScopedDevice("cpu"):
        m = wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32)
        kd = wp.array(kinds.reshape(n, 4), dtype=wp.int32)
        diag, g_conv, b, x = (wp.zeros(n, dtype=wp.float64) for _ in range(4))
        g_off = wp.zeros((n, 4), dtype=wp.float64)
        wp.launch(
            hs.assemble,
            n,
            inputs=[m, kd, wp.float64(k), wp.float64(dx), wp.float64(0.0), wp.float64(h), wp.float64(q)],
            outputs=[diag, g_off, g_conv, b],
        )
        hs.pcg(diag, g_off, b, x, nx, tol=1e-12)
        theta = x.numpy().reshape(ny, nx)
    tb = ny * dx
    expected_bottom = q * (1.0 / h + (tb - dx / 2.0) / k)
    assert theta[0].mean() == pytest.approx(expected_bottom, rel=1e-8)
    assert np.ptp(theta[0]) < 1e-9 * expected_bottom  # no x variation


def test_open_plate_limit_of_channel_correlation():
    """For very wide channels Bar-Cohen & Rohsenow reduces to the isolated-plate 0.59 Ra^0.25 law."""
    assert hs.h_channel(1.0, 0.1, 40.0) == pytest.approx(hs.h_open(0.1, 40.0), rel=1e-3)
    assert hs.h_channel(0.002, 0.1, 40.0) < 0.5 * hs.h_channel(0.008, 0.1, 40.0)  # narrow channels choke


def test_design_solve_energy_balance_and_optimum_shape():
    res = {n: hs.solve(hs.Design(n_fins=n, fin_height_mm=20.0, power_w=8.0)) for n in (4, 8, 20)}
    for r in res.values():
        assert abs(r.energy_balance_err) < 1e-6
        assert r.outer_iters < 30
    # Too few fins (little area) and too many fins (choked channels) are both worse than a middle spacing.
    assert res[8].t_max_rise_k < res[4].t_max_rise_k
    assert res[8].t_max_rise_k < res[20].t_max_rise_k


def test_mlp_fits_smooth_function():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, size=(300, 3))
    y = np.sin(2 * x[:, 0]) + 0.5 * x[:, 1] ** 2 - 0.3 * x[:, 2]
    mlp = sur.MLP(sizes=(3, 32, 32, 1), seed=0).fit(x[:250], y[:250], epochs=2000)
    err = mlp.predict(x[250:]) - y[250:]
    assert float(np.sqrt((err**2).mean())) < 0.05


def test_sampling_and_sweep_shapes():
    designs = twin.sample_designs(50, seed=0)
    n, h, q = (np.array(c) for c in zip(*designs, strict=True))
    assert n.min() >= 4 and n.max() <= 24
    assert h.min() >= 10 and h.max() <= 40 and q.min() >= 2 and q.max() <= 20
    rows = [{"spacing_mm": twin.spacing_mm(a), "fin_height_mm": b, "power_w": c} for a, b, c in designs]
    model = sur.QuadraticLS().fit(twin.features(rows), np.log(1.0 + q))
    optima, info = twin.sweep(model)
    assert len(optima) == len(twin.SWEEP_H) * len(twin.SWEEP_Q)
    assert info["designs_evaluated"] == 21 * len(twin.SWEEP_H) * len(twin.SWEEP_Q)


def test_physicsnemo_script_is_marked_and_importable(tmp_path):
    tp = _load("train_physicsnemo")  # importing must not need torch or physicsnemo
    assert "not executed: needs GPU" in (tp.__doc__ or "")
    assert "not executed: needs GPU" in inspect.getdoc(tp.train)
    csv_path = tmp_path / "d.csv"
    csv_path.write_text("spacing_mm,fin_height_mm,power_w,t_max_rise_k\n6.0,30,10,38.5\n")
    x, y = tp.load_xy(csv_path)
    assert x == [[6.0, 30.0, 10.0]]
    assert math.isclose(y[0], 38.5)
