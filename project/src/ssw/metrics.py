"""Pure NumPy metrics shared by all phases.

Nothing in this module imports Newton or Warp, so it can be unit-tested and used for
analysis on any machine.

Conventions follow Newton: Z is up, quaternions are stored XYZW, a body transform is
``(px, py, pz, qx, qy, qz, qw)``, and a spatial velocity is ``(vx, vy, vz, wx, wy, wz)``
with the linear part being the world-frame velocity of the body's center of mass and
the angular part expressed in the world frame.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81

# ---------------------------------------------------------------------------
# Rigid-body kinematics and energy
# ---------------------------------------------------------------------------


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert XYZW quaternions to rotation matrices.

    Args:
        q: Array of shape ``(..., 4)`` in XYZW order. Need not be normalized.

    Returns:
        Array of shape ``(..., 3, 3)``.
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty((*q.shape[:-1], 3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


@dataclass(frozen=True)
class EnergyBreakdown:
    """Mechanical energy of a set of rigid bodies [J]."""

    kinetic: float
    potential: float

    @property
    def total(self) -> float:
        return self.kinetic + self.potential


def rigid_body_energy(
    body_q: np.ndarray,
    body_qd: np.ndarray,
    mass: np.ndarray,
    inertia: np.ndarray,
    com: np.ndarray,
    *,
    gravity: float = GRAVITY,
) -> EnergyBreakdown:
    """Kinetic plus gravitational potential energy of rigid bodies (Z-up).

    Args:
        body_q: ``(N, 7)`` body-origin transforms ``(p, q_xyzw)`` in world frame.
        body_qd: ``(N, 6)`` spatial velocities ``(v_com, omega)`` in world frame.
        mass: ``(N,)`` masses [kg].
        inertia: ``(N, 3, 3)`` inertia tensors about the COM in the body frame [kg m^2].
        com: ``(N, 3)`` COM offsets in the body frame [m].
        gravity: Magnitude of gravity along -Z [m/s^2].

    Returns:
        Kinetic and potential energy. The potential uses z = 0 as reference.
    """
    body_q = np.asarray(body_q, dtype=np.float64).reshape(-1, 7)
    body_qd = np.asarray(body_qd, dtype=np.float64).reshape(-1, 6)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    inertia = np.asarray(inertia, dtype=np.float64).reshape(-1, 3, 3)
    com = np.asarray(com, dtype=np.float64).reshape(-1, 3)

    rot = quat_xyzw_to_matrix(body_q[:, 3:7])
    v = body_qd[:, 0:3]
    w = body_qd[:, 3:6]
    # Inertia in world frame: R I R^T. omega^T (R I R^T) omega = (R^T omega)^T I (R^T omega).
    w_body = np.einsum("nji,nj->ni", rot, w)
    rot_ke = 0.5 * np.einsum("ni,nij,nj->n", w_body, inertia, w_body)
    lin_ke = 0.5 * mass * np.einsum("ni,ni->n", v, v)
    com_world = body_q[:, 0:3] + np.einsum("nij,nj->ni", rot, com)
    pe = mass * gravity * com_world[:, 2]
    return EnergyBreakdown(kinetic=float(np.sum(lin_ke + rot_ke)), potential=float(np.sum(pe)))


# ---------------------------------------------------------------------------
# Trajectory summaries
# ---------------------------------------------------------------------------


def _tail(x: np.ndarray, frac: float) -> np.ndarray:
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"window fraction must be in (0, 1], got {frac}")
    n = max(1, round(len(x) * frac))
    return x[-n:]


def settle_height(z: Sequence[float], window_frac: float = 0.2) -> float:
    """Mean of the last ``window_frac`` of a height trace [m]."""
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        raise ValueError("empty trace")
    return float(np.mean(_tail(z, window_frac)))


def settle_time(t: Sequence[float], z: Sequence[float], tol: float = 0.005, window_frac: float = 0.2) -> float:
    """First time after which ``|z - settle_height| <= tol`` holds for the rest of the trace.

    Returns ``nan`` if the trace never settles within ``tol`` (including at the last sample).
    """
    t = np.asarray(t, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if t.shape != z.shape or t.size == 0:
        raise ValueError("t and z must be non-empty and the same shape")
    final = settle_height(z, window_frac)
    outside = np.abs(z - final) > tol
    if outside[-1]:
        return float("nan")
    if not outside.any():
        return float(t[0])
    last_out = int(np.flatnonzero(outside)[-1])
    return float(t[last_out + 1])


def energy_drift(t: Sequence[float], e: Sequence[float], window_frac: float = 0.5) -> float:
    """Least-squares slope of energy over the last ``window_frac`` of the run [J/s].

    For a settled, actively held robot this should be near zero; a persistent positive
    slope means the integrator or contact model is injecting energy.
    """
    t = _tail(np.asarray(t, dtype=np.float64), window_frac)
    e = _tail(np.asarray(e, dtype=np.float64), window_frac)
    if t.size < 2:
        return 0.0
    slope, _ = np.polyfit(t - t[0], e, 1)
    return float(slope)


def is_stable(
    z: Sequence[float],
    e: Sequence[float] | None = None,
    z_min: float = -0.5,
    z_max: float = 5.0,
    e_max: float = 1.0e4,
) -> tuple[bool, str]:
    """Heuristic stability check: finite values, base height in a box, bounded energy.

    Returns:
        ``(stable, reason)`` where ``reason`` is ``"ok"`` or a short description.
    """
    z = np.asarray(z, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        return False, "non-finite base height"
    if z.min() < z_min:
        return False, f"base fell through ground (min z={z.min():.3f} m)"
    if z.max() > z_max:
        return False, f"base launched (max z={z.max():.3f} m)"
    if e is not None:
        e = np.asarray(e, dtype=np.float64)
        if not np.all(np.isfinite(e)):
            return False, "non-finite energy"
        if np.abs(e).max() > e_max:
            return False, f"energy blew up (max |E|={np.abs(e).max():.3g} J)"
    return True, "ok"


@dataclass(frozen=True)
class TimingStats:
    """Wall-clock statistics in milliseconds."""

    n: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    std_ms: float
    min_ms: float


def timing_stats(samples_ms: Iterable[float], skip: int = 0) -> TimingStats:
    """Summarize wall-clock samples, dropping the first ``skip`` (warm-up/compile)."""
    x = np.asarray(list(samples_ms), dtype=np.float64)[skip:]
    if x.size == 0:
        raise ValueError("no samples left after skip")
    return TimingStats(
        n=int(x.size),
        mean_ms=float(x.mean()),
        median_ms=float(np.median(x)),
        p90_ms=float(np.percentile(x, 90)),
        std_ms=float(x.std(ddof=1)) if x.size > 1 else 0.0,
        min_ms=float(x.min()),
    )


def bitwise_identical(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two arrays have the same shape, dtype and bytes (NaNs compare by bit pattern)."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute element-wise difference (``inf`` on shape mismatch)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


# ---------------------------------------------------------------------------
# Phase 2: judge verdicts vs. simulator ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryScores:
    """Binary classification scores where the positive class is ``positive``."""

    tp: int
    fp: int
    fn: int
    tn: int
    abstained: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if np.isnan(p) or np.isnan(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else float("nan")

    @property
    def coverage(self) -> float:
        """Fraction of items on which the judge committed to an answer."""
        total = self.n + self.abstained
        return self.n / total if total else float("nan")


def binary_scores(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    positive: object,
    abstain: Iterable[object] = (None,),
) -> BinaryScores:
    """Precision/recall bookkeeping with an explicit abstention class.

    Predictions whose value is in ``abstain`` (e.g. the Reasoner's "(B) Impossible to
    tell", or an unparseable answer mapped to ``None``) are counted separately and
    excluded from tp/fp/fn/tn.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    abstain = set(abstain)
    tp = fp = fn = tn = ab = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if p in abstain:
            ab += 1
            continue
        tpos = t == positive
        ppos = p == positive
        if tpos and ppos:
            tp += 1
        elif ppos:
            fp += 1
        elif tpos:
            fn += 1
        else:
            tn += 1
    return BinaryScores(tp=tp, fp=fp, fn=fn, tn=tn, abstained=ab)


def confusion_matrix(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    labels_true: Sequence[object],
    labels_pred: Sequence[object],
) -> np.ndarray:
    """Counts with rows = ground-truth label, columns = predicted label.

    Items whose label is not listed are ignored. ``labels_pred`` may include an
    abstention label (e.g. ``"B"`` or ``None``) that has no ground-truth row.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    ti = {lab: i for i, lab in enumerate(labels_true)}
    pi = {lab: i for i, lab in enumerate(labels_pred)}
    cm = np.zeros((len(labels_true), len(labels_pred)), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        if t in ti and p in pi:
            cm[ti[t], pi[p]] += 1
    return cm


def agreement(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Raw agreement rate over all items (abstentions count as disagreement)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return float("nan")
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


# ---------------------------------------------------------------------------
# Phase 2: transfer control fidelity (PAI-Bench-C style)
# ---------------------------------------------------------------------------


def scale_invariant_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-6) -> float:
    """Scale-invariant RMSE in log depth (Eigen et al. 2014): ``sqrt(mean(d^2) - mean(d)^2)``, ``d = log p - log g``.

    Invariant to a global scale of ``pred``. Pixels with non-positive or non-finite depth, or
    outside ``mask``, are ignored. Check the exact PAI-Bench-C definition before comparing
    against published numbers (Nano: 0.610).
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > eps) & (gt > eps)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
        return float("nan")
    d = np.log(pred[valid]) - np.log(gt[valid])
    return float(np.sqrt(max(np.mean(d * d) - np.mean(d) ** 2, 0.0)))


def mean_iou(
    pred: np.ndarray, gt: np.ndarray, num_classes: int | None = None, ignore_index: int | None = None
) -> float:
    """Mean intersection-over-union over classes present in ``gt`` or ``pred`` (integer label maps)."""
    pred = np.asarray(pred).astype(np.int64).ravel()
    gt = np.asarray(gt).astype(np.int64).ravel()
    if pred.shape != gt.shape:
        raise ValueError("shape mismatch")
    if ignore_index is not None:
        keep = gt != ignore_index
        pred, gt = pred[keep], gt[keep]
    n = num_classes if num_classes is not None else int(max(pred.max(initial=0), gt.max(initial=0))) + 1
    cm = np.bincount(gt * n + pred, minlength=n * n).reshape(n, n)
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    present = union > 0
    return float(np.mean(inter[present] / union[present])) if present.any() else float("nan")
