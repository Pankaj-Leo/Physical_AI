"""Fast, pure-NumPy tests for ssw.metrics."""

import math

import numpy as np
import pytest

from ssw import metrics


def test_quat_identity_and_z_rotation():
    np.testing.assert_allclose(metrics.quat_xyzw_to_matrix([0, 0, 0, 1]), np.eye(3), atol=1e-12)
    s = math.sqrt(0.5)
    r = metrics.quat_xyzw_to_matrix([0, 0, s, s])  # +90 deg about Z, XYZW order
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_energy_of_body_at_rest_is_pure_potential():
    e = metrics.rigid_body_energy(
        body_q=[[0, 0, 2.0, 0, 0, 0, 1]],
        body_qd=[[0, 0, 0, 0, 0, 0]],
        mass=[3.0],
        inertia=[np.eye(3)],
        com=[[0, 0, 0.5]],
    )
    assert e.kinetic == 0.0
    assert e.potential == pytest.approx(3.0 * 9.81 * 2.5)


def test_energy_translation_and_rotation():
    # Rotated body: angular KE must be frame-consistent (omega in world, inertia in body).
    s = math.sqrt(0.5)
    inertia = np.diag([1.0, 2.0, 3.0])
    e = metrics.rigid_body_energy(
        body_q=[[0, 0, 0, 0, 0, s, s]],  # 90 deg about Z: body x -> world y
        body_qd=[[1.0, 0, 0, 0, 2.0, 0]],  # spin about world y == body x
        mass=[2.0],
        inertia=[inertia],
        com=[[0, 0, 0]],
    )
    assert e.kinetic == pytest.approx(0.5 * 2.0 * 1.0 + 0.5 * 1.0 * 4.0)
    assert e.total == pytest.approx(e.kinetic)


def test_settle_height_and_time():
    t = np.linspace(0, 1, 101)
    z = np.where(t < 0.5, 0.7 - t, 0.2)
    assert metrics.settle_height(z) == pytest.approx(0.2)
    assert metrics.settle_time(t, z, tol=0.005) == pytest.approx(0.5)
    assert math.isnan(metrics.settle_time(t, 0.2 + 0.1 * np.sin(40 * t), tol=1e-3))


def test_energy_drift_slope():
    t = np.linspace(0, 2, 201)
    assert metrics.energy_drift(t, 5.0 + 0.3 * t) == pytest.approx(0.3, rel=1e-9)
    assert metrics.energy_drift(t, np.full_like(t, 7.0)) == pytest.approx(0.0, abs=1e-12)


def test_is_stable():
    assert metrics.is_stable([0.5, 0.4, 0.45], [1.0, 1.0, 1.0]) == (True, "ok")
    assert not metrics.is_stable([0.5, float("nan")])[0]
    assert not metrics.is_stable([0.5, -2.0])[0]
    assert not metrics.is_stable([0.5, 0.4], [1.0, 1e9])[0]


def test_timing_stats_skips_warmup():
    ts = metrics.timing_stats([1000.0, 10.0, 12.0, 11.0], skip=1)
    assert ts.n == 3
    assert ts.median_ms == pytest.approx(11.0)
    assert ts.min_ms == 10.0
    with pytest.raises(ValueError):
        metrics.timing_stats([1.0], skip=1)


def test_bitwise_identical():
    a = np.array([1.0, np.nan], dtype=np.float32)
    assert metrics.bitwise_identical(a, a.copy())
    assert not metrics.bitwise_identical(a, np.array([1.0, 2.0], dtype=np.float32))
    assert not metrics.bitwise_identical(a, a.astype(np.float64))
    assert metrics.max_abs_diff([1.0, 2.0], [1.0, 2.5]) == 0.5


def test_binary_scores_with_abstention():
    y_true = ["failure", "failure", "success", "success", "failure"]
    y_pred = ["failure", "success", "failure", "success", None]
    bs = metrics.binary_scores(y_true, y_pred, positive="failure")
    assert (bs.tp, bs.fp, bs.fn, bs.tn, bs.abstained) == (1, 1, 1, 1, 1)
    assert bs.precision == 0.5
    assert bs.recall == 0.5
    assert bs.f1 == 0.5
    assert bs.coverage == pytest.approx(0.8)


def test_binary_scores_degenerate():
    bs = metrics.binary_scores(["success"], ["success"], positive="failure")
    assert math.isnan(bs.precision)
    assert math.isnan(bs.recall)


def test_confusion_matrix_and_agreement():
    y_true = ["success", "failure", "failure", "success"]
    y_pred = ["success", "failure", None, "failure"]
    cm = metrics.confusion_matrix(y_true, y_pred, ["success", "failure"], ["success", "failure", None])
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 1, 1]])
    assert metrics.agreement(y_true, y_pred) == 0.5


def test_si_rmse_is_scale_invariant():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.5, 5.0, size=(8, 8))
    assert metrics.scale_invariant_rmse(3.7 * gt, gt) == pytest.approx(0.0, abs=1e-9)
    noisy = gt * np.exp(rng.normal(0, 0.1, size=gt.shape))
    assert 0.05 < metrics.scale_invariant_rmse(noisy, gt) < 0.15
    gt_bad = gt.copy()
    gt_bad[0, 0] = 0.0  # invalid pixel ignored
    assert np.isfinite(metrics.scale_invariant_rmse(gt, gt_bad))


def test_mean_iou():
    gt = np.array([[0, 0, 1, 1], [2, 2, 2, 2]])
    assert metrics.mean_iou(gt, gt) == 1.0
    pred = np.array([[0, 1, 1, 1], [2, 2, 2, 2]])
    # class0: 1/2, class1: 2/3, class2: 1
    assert metrics.mean_iou(pred, gt) == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
