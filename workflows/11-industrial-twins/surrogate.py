"""Surrogates in plain numpy: a small MLP trained with Adam, and a quadratic least-squares baseline.

Inputs are (channel spacing [mm], fin height [mm], heat load [W]); the target is log(T_max rise [K]). The log
keeps the target well scaled across a 2-20 W load range, where natural convection makes T roughly Q^0.8.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scaler:
    """Affine map of each column to [-1, 1]."""

    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Scaler:
        return cls(x.min(axis=0), x.max(axis=0))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0


@dataclass
class MLP:
    """Fully connected tanh network. ``sizes`` = [inputs, hidden..., outputs]."""

    sizes: tuple[int, ...] = (3, 64, 64, 1)
    seed: int = 0
    params: list[np.ndarray] = field(default_factory=list)
    x_scaler: Scaler | None = None
    y_mean: float = 0.0
    y_std: float = 1.0
    history: list[float] = field(default_factory=list)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.params = []
        for a, b in zip(self.sizes[:-1], self.sizes[1:], strict=True):
            self.params.append(rng.normal(0.0, np.sqrt(1.0 / a), size=(a, b)))  # LeCun init for tanh
            self.params.append(np.zeros(b))

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        acts = [x]
        h = x
        n_layers = len(self.params) // 2
        for i in range(n_layers):
            w, b = self.params[2 * i], self.params[2 * i + 1]
            h = h @ w + b
            if i < n_layers - 1:
                h = np.tanh(h)
            acts.append(h)
        return h, acts

    def _grads(self, x: np.ndarray, y: np.ndarray, weight_decay: float) -> tuple[float, list[np.ndarray]]:
        out, acts = self._forward(x)
        err = out - y
        loss = float(np.mean(err**2))
        delta = 2.0 * err / len(x)
        grads = [None] * len(self.params)
        n_layers = len(self.params) // 2
        for i in reversed(range(n_layers)):
            w = self.params[2 * i]
            grads[2 * i] = acts[i].T @ delta + weight_decay * w
            grads[2 * i + 1] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ w.T) * (1.0 - acts[i] ** 2)
        return loss, grads

    def fit(
        self, x: np.ndarray, y: np.ndarray, epochs: int = 6000, lr: float = 3e-3, weight_decay: float = 1e-6
    ) -> MLP:
        """Full-batch Adam with a cosine learning-rate decay."""
        self.x_scaler = Scaler.fit(x)
        self.y_mean, self.y_std = float(y.mean()), float(y.std())
        xs = self.x_scaler(x)
        ys = ((y - self.y_mean) / self.y_std).reshape(-1, 1)
        m = [np.zeros_like(p) for p in self.params]
        v = [np.zeros_like(p) for p in self.params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, epochs + 1):
            step = lr * 0.5 * (1.0 + np.cos(np.pi * t / epochs))
            loss, grads = self._grads(xs, ys, weight_decay)
            for k, g in enumerate(grads):
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * g * g
                self.params[k] -= step * (m[k] / (1 - b1**t)) / (np.sqrt(v[k] / (1 - b2**t)) + eps)
            if t % 100 == 0:
                self.history.append(loss)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        out, _ = self._forward(self.x_scaler(np.atleast_2d(x)))
        return out[:, 0] * self.y_std + self.y_mean


def _quad_features(xs: np.ndarray) -> np.ndarray:
    cols = [np.ones(len(xs))]
    d = xs.shape[1]
    cols += [xs[:, i] for i in range(d)]
    cols += [xs[:, i] * xs[:, j] for i in range(d) for j in range(i, d)]
    return np.stack(cols, axis=1)


@dataclass
class QuadraticLS:
    """Second-order polynomial in the scaled inputs, fitted by least squares (the cheap baseline)."""

    coef: np.ndarray | None = None
    x_scaler: Scaler | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> QuadraticLS:
        self.x_scaler = Scaler.fit(x)
        self.coef, *_ = np.linalg.lstsq(_quad_features(self.x_scaler(x)), y, rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _quad_features(self.x_scaler(np.atleast_2d(x))) @ self.coef


def error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Errors on the physical target (T_max rise in K)."""
    err = y_pred - y_true
    ss_res = float((err**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "n": len(y_true),
        "mae_k": float(np.abs(err).mean()),
        "rmse_k": float(np.sqrt((err**2).mean())),
        "max_abs_k": float(np.abs(err).max()),
        "mape_pct": float((np.abs(err) / y_true).mean() * 100.0),
        "max_ape_pct": float((np.abs(err) / y_true).max() * 100.0),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }
