"""Very short CPU simulations. Skipped when Newton/Warp are not installed.

First run compiles Warp kernels for the CPU (tens of seconds); later runs hit the kernel cache.
"""

import math

import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from ssw import bakeoff  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module", autouse=True)
def _quiet_warp():
    wp.config.log_level = wp.LOG_WARNING


@pytest.mark.parametrize("solver", ["xpbd", "mujoco_c"])
def test_few_frames_run_and_fall(solver):
    cfg = bakeoff.BakeoffConfig(frames=5, substeps=4, device="cpu")
    res = bakeoff.run_solver(solver, cfg, record_body_q=True)
    assert res.ran, res.reason
    assert res.stable, res.stability_reason
    assert len(res.t) == cfg.frames + 1
    # Free fall from 0.7 m during the first 50 ms: base must move down, energy stays finite.
    assert res.base_z[-1] < res.base_z[0]
    assert all(math.isfinite(e) for e in res.energy)
    assert res.body_q_history.shape[0] == cfg.frames + 1
    row = bakeoff.summarize(res, cfg, warmup_frames=1)
    assert row["ran"] and "ms_per_frame_median" in row


def test_xpbd_run_twice_bit_identical():
    out = bakeoff.determinism_check("xpbd", bakeoff.BakeoffConfig(substeps=4), frames=3)
    assert out["default_flags"]["bit_identical"]
    assert out["deterministic_flags"]["bit_identical"]


def test_unknown_solver_is_reported_not_raised():
    res = bakeoff.run_solver("nope", bakeoff.BakeoffConfig(frames=1))
    assert not res.ran
    assert "unknown solver" in res.reason


def test_terrain_refuses_full_res_mpm_on_cpu():
    terrain = pytest.importorskip("ssw.terrain")
    cfg = terrain.TerrainConfig(material="sand", device="cpu")
    assert terrain.estimate_particle_count(cfg) > 20_000
    with pytest.raises(SystemExit, match="GPU job"):
        terrain.run(cfg)
    assert terrain.estimate_particle_count(terrain.TerrainConfig.tiny(material="sand")) < 2_000
    assert terrain.estimate_particle_count(terrain.TerrainConfig(material="rigid")) == 0


def test_auto_device_falls_back_to_cpu_without_cuda():
    from ssw.device import resolve_device

    if wp.get_cuda_device_count() > 0:
        assert resolve_device("auto") == "cuda:0"
    else:
        with pytest.warns(RuntimeWarning, match="falling back"):
            assert resolve_device("auto") == "cpu"
        with pytest.raises(SystemExit, match="no CUDA device"):
            resolve_device("cuda:0")


@pytest.mark.gpu
@pytest.mark.skipif(wp.get_cuda_device_count() == 0, reason="needs a CUDA GPU")
@pytest.mark.parametrize("solver", ["xpbd", "mujoco_warp"])
def test_cuda_graph_run_matches_eager(solver):
    cfg = bakeoff.BakeoffConfig(frames=6, substeps=4, device="cuda:0")
    graphed = bakeoff.run_solver(solver, cfg)
    eager = bakeoff.run_solver(solver, bakeoff.BakeoffConfig(frames=6, substeps=4, device="cuda:0", graph=False))
    assert graphed.ran and eager.ran, (graphed.reason, eager.reason)
    assert graphed.graph and not eager.graph
    assert graphed.device == "cuda:0"
    assert graphed.base_z == pytest.approx(eager.base_z, abs=1e-5)
