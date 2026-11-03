"""High-fidelity reference solver: 2D steady conduction in a finned heat-sink cross-section, in Warp kernels.

Physics (all SI, per unit depth along the fin length L):

* Aluminium sink (k = 167 W/m K, 6061-T6) of width W, base thickness tb, ``n`` fins of thickness t and height H,
  outer fins flush with the sink edges, so the channel spacing is s = (W - t)/(n - 1) - t.
* A chip of width ws, centred under the base, injects Q watts (uniform flux Q / (ws L)); the rest of the base back
  face is adiabatic.
* Every other surface loses heat to 25 C air by natural convection. Channel walls use the Bar-Cohen & Rohsenow
  (1984) correlation for symmetric isothermal vertical plates (as given in Incropera et al., Fundamentals of Heat
  and Mass Transfer, ch. 9): Nu_s = [576 / El^2 + 2.873 / El^0.5]^-0.5, El = Ra_s s / L. Open faces (outer
  fin sides, fin tips, base sides) use the laminar vertical-plate correlation Nu_L = 0.59 Ra_L^0.25 (Incropera,
  ch. 9).
  h depends on the surface temperature through Ra, so the solve is a fixed-point loop on h.
* Radiation is neglected. Air properties are constants at a 325 K film temperature (linear interpolation of
  Incropera Table A.4 between 300 K and 350 K).

Numerics: cell-centred finite volumes on a uniform grid (dx = 0.5 mm by default; 0.25 mm agrees to 0.02% on the
designs we checked), solid cells only, convective faces as a series conductance (h dx and half a cell of
conduction). The SPD system A theta = b is solved matrix-free with Jacobi-preconditioned conjugate gradients;
matvec, vector updates and dot products are Warp kernels with device-side scalars, replayed as a captured Warp
graph. The same code targets CUDA (the intended device; not yet run on a GPU) and the Warp CPU device (executed).

The narrow-channel penalty in h(s) against the area gain from more fins is what creates an optimal spacing. It is
the trade-off the SIGGRAPH 2026 real-time twin talk's agent demo explored (as reported in the talk, the agent
"finds the optimal channel spacing" for heat dissipation, 12:57-13:07, auto-captioned).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import warp as wp

K_AL = 167.0  # W/(m K), aluminium 6061-T6
T_AMB_C = 25.0
G = 9.81
T_FILM_K = 325.0
# Air at 325 K, linear between Incropera Table A.4 values at 300 K and 350 K.
NU_AIR = 0.5 * (15.89e-6 + 20.92e-6)  # m^2/s
ALPHA_AIR = 0.5 * (22.5e-6 + 29.9e-6)  # m^2/s
K_AIR = 0.5 * (26.3e-3 + 30.0e-3)  # W/(m K)
BETA = 1.0 / T_FILM_K

SOLID, CHANNEL, OPEN, ADIABATIC, SOURCE = 0, 1, 2, 3, 4  # face kinds


@dataclass(frozen=True)
class Design:
    n_fins: int = 10
    fin_height_mm: float = 30.0
    power_w: float = 20.0
    width_mm: float = 50.0
    base_mm: float = 4.0
    fin_t_mm: float = 1.0
    length_mm: float = 100.0  # fin length along the air flow (out of plane)
    source_mm: float = 20.0

    @property
    def spacing_mm(self) -> float:
        return (self.width_mm - self.fin_t_mm) / (self.n_fins - 1) - self.fin_t_mm


@dataclass
class Result:
    t_max_rise_k: float
    t_base_mean_rise_k: float
    r_th_k_per_w: float
    h_channel: float
    h_open: float
    energy_balance_err: float
    outer_iters: int
    cg_iters: int
    wall_s: float
    solid_cells: int


def ra_coeff() -> float:
    """g beta / (nu alpha): Ra = ra_coeff * dT * length^3."""
    return G * BETA / (NU_AIR * ALPHA_AIR)


def h_channel(spacing_m: float, length_m: float, dt_k: float) -> float:
    """Bar-Cohen & Rohsenow (1984), symmetric isothermal plates (Incropera, ch. 9)."""
    dt = max(dt_k, 1e-3)
    el = ra_coeff() * dt * spacing_m**3 * spacing_m / length_m
    nu = (576.0 / el**2 + 2.873 / math.sqrt(el)) ** -0.5
    return nu * K_AIR / spacing_m


def h_open(length_m: float, dt_k: float) -> float:
    """Laminar isolated vertical plate, Nu_L = 0.59 Ra_L^0.25 (Incropera, ch. 9)."""
    ra = ra_coeff() * max(dt_k, 1e-3) * length_m**3
    return 0.59 * ra**0.25 * K_AIR / length_m


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def rasterize(d: Design, dx_mm: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Solid mask (ny, nx) and face kinds (ny, nx, 4) in order W, E, S, N."""
    nx = round(d.width_mm / dx_mm)
    nb = round(d.base_mm / dx_mm)
    ny = nb + round(d.fin_height_mm / dx_mm)
    xc = (np.arange(nx) + 0.5) * dx_mm
    mask = np.zeros((ny, nx), dtype=bool)
    mask[:nb, :] = True
    pitch = (d.width_mm - d.fin_t_mm) / (d.n_fins - 1)
    for i in range(d.n_fins):
        x0 = i * pitch
        cols = (xc >= x0) & (xc < x0 + d.fin_t_mm)
        mask[nb:, cols] = True
    src = np.abs(xc - d.width_mm / 2) < d.source_mm / 2
    kinds = np.full((ny, nx, 4), SOLID, dtype=np.int32)
    pad = np.pad(mask, 1, constant_values=False)
    neighbours = {0: pad[1:-1, :-2], 1: pad[1:-1, 2:], 2: pad[:-2, 1:-1], 3: pad[2:, 1:-1]}
    for dirn, nb_solid in neighbours.items():
        k = np.where(nb_solid, SOLID, CHANNEL)
        if dirn == 0:
            k[:, 0] = OPEN
        elif dirn == 1:
            k[:, -1] = OPEN
        elif dirn == 2:
            k[0, :] = np.where(src, SOURCE, ADIABATIC)
        else:
            k[-1, :] = OPEN  # fin tips
        kinds[..., dirn] = k
    return mask, kinds


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def assemble(  # noqa: PLR0917 (Warp kernels take positional arguments)
    mask: wp.array[wp.int32],
    kinds: wp.array2d[wp.int32],
    k: wp.float64,
    dx: wp.float64,
    h_ch: wp.float64,
    h_op: wp.float64,
    q_flux: wp.float64,
    diag: wp.array[wp.float64],
    g_off: wp.array2d[wp.float64],
    g_conv: wp.array[wp.float64],
    b: wp.array[wp.float64],
):
    i = wp.tid()
    if mask[i] == 0:
        diag[i] = wp.float64(1.0)
        b[i] = wp.float64(0.0)
        g_conv[i] = wp.float64(0.0)
        for d in range(4):
            g_off[i, d] = wp.float64(0.0)
        return
    dg = wp.float64(0.0)
    gc = wp.float64(0.0)
    src = wp.float64(0.0)
    half = wp.float64(2.0) * k  # conductance of half a cell, per unit depth
    for d in range(4):
        kind = kinds[i, d]
        g_off[i, d] = wp.float64(0.0)
        if kind == 0:
            g_off[i, d] = k
            dg += k
        elif kind == 1:
            g = wp.float64(1.0) / (wp.float64(1.0) / (h_ch * dx) + wp.float64(1.0) / half)
            gc += g
        elif kind == 2:
            g = wp.float64(1.0) / (wp.float64(1.0) / (h_op * dx) + wp.float64(1.0) / half)
            gc += g
        elif kind == 4:
            src += q_flux * dx
    diag[i] = dg + gc
    g_conv[i] = gc
    b[i] = src


@wp.kernel
def matvec(
    x: wp.array[wp.float64],
    diag: wp.array[wp.float64],
    g_off: wp.array2d[wp.float64],
    nx: int,
    y: wp.array[wp.float64],
):
    i = wp.tid()
    acc = diag[i] * x[i]
    if g_off[i, 0] != wp.float64(0.0):
        acc -= g_off[i, 0] * x[i - 1]
    if g_off[i, 1] != wp.float64(0.0):
        acc -= g_off[i, 1] * x[i + 1]
    if g_off[i, 2] != wp.float64(0.0):
        acc -= g_off[i, 2] * x[i - nx]
    if g_off[i, 3] != wp.float64(0.0):
        acc -= g_off[i, 3] * x[i + nx]
    y[i] = acc


@wp.kernel
def cg_init(  # noqa: PLR0917
    b: wp.array[wp.float64],
    ax: wp.array[wp.float64],
    diag: wp.array[wp.float64],
    r: wp.array[wp.float64],
    z: wp.array[wp.float64],
    p: wp.array[wp.float64],
    acc: wp.array[wp.float64],
):
    i = wp.tid()
    ri = b[i] - ax[i]
    zi = ri / diag[i]
    r[i] = ri
    z[i] = zi
    p[i] = zi
    wp.atomic_add(acc, 1, ri * zi)
    wp.atomic_add(acc, 2, ri * ri)


@wp.kernel
def cg_matvec_dot(  # noqa: PLR0917
    p: wp.array[wp.float64],
    diag: wp.array[wp.float64],
    g_off: wp.array2d[wp.float64],
    nx: int,
    ap: wp.array[wp.float64],
    acc: wp.array[wp.float64],
):
    """ap = A p and acc[0] += p . ap (one fused pass)."""
    i = wp.tid()
    v = diag[i] * p[i]
    if g_off[i, 0] != wp.float64(0.0):
        v -= g_off[i, 0] * p[i - 1]
    if g_off[i, 1] != wp.float64(0.0):
        v -= g_off[i, 1] * p[i + 1]
    if g_off[i, 2] != wp.float64(0.0):
        v -= g_off[i, 2] * p[i - nx]
    if g_off[i, 3] != wp.float64(0.0):
        v -= g_off[i, 3] * p[i + nx]
    ap[i] = v
    wp.atomic_add(acc, 0, p[i] * v)


@wp.kernel
def cg_update_xr(  # noqa: PLR0917
    p: wp.array[wp.float64],
    ap: wp.array[wp.float64],
    diag: wp.array[wp.float64],
    scal: wp.array[wp.float64],
    acc: wp.array[wp.float64],
    x: wp.array[wp.float64],
    r: wp.array[wp.float64],
    z: wp.array[wp.float64],
):
    """alpha = rz / pAp computed on the device; x += alpha p, r -= alpha Ap, z = r / diag; accumulate r.z, r.r."""
    i = wp.tid()
    alpha = wp.float64(0.0)
    if acc[0] > wp.float64(0.0):
        alpha = scal[0] / acc[0]
    x[i] = x[i] + alpha * p[i]
    ri = r[i] - alpha * ap[i]
    zi = ri / diag[i]
    r[i] = ri
    z[i] = zi
    wp.atomic_add(acc, 1, ri * zi)
    wp.atomic_add(acc, 2, ri * ri)


@wp.kernel
def cg_update_p(
    z: wp.array[wp.float64],
    scal: wp.array[wp.float64],
    acc: wp.array[wp.float64],
    p: wp.array[wp.float64],
):
    i = wp.tid()
    beta = wp.float64(0.0)
    if scal[0] > wp.float64(0.0):
        beta = acc[1] / scal[0]
    p[i] = z[i] + beta * p[i]


@wp.kernel
def cg_rotate(acc: wp.array[wp.float64], scal: wp.array[wp.float64]):
    """Single thread: scal = (r.z, r.r) of this iteration; reset the accumulators."""
    scal[0] = acc[1]
    scal[1] = acc[2]
    acc[0] = wp.float64(0.0)
    acc[1] = wp.float64(0.0)
    acc[2] = wp.float64(0.0)


def pcg(diag, g_off, b, x, nx: int, *, tol: float = 1e-8, max_iter: int = 50000, chunk: int = 25) -> int:
    """Jacobi-preconditioned CG, all scalars on the device, ``chunk`` iterations per captured Warp graph.

    ``x`` holds the initial guess and receives the solution. Convergence (recursive residual
    ||r|| < tol ||b||) is checked on the host once per chunk, so up to ``chunk - 1`` extra iterations run;
    they are harmless. Returns the iteration count. Reductions use ``wp.atomic_add`` into one address: fine on
    the CPU device; on a GPU a tiled reduction would be faster (not yet measured on a GPU).
    """
    dev = x.device
    n = x.shape[0]
    r, z, p, ap = (wp.zeros(n, dtype=wp.float64, device=dev) for _ in range(4))
    acc = wp.zeros(3, dtype=wp.float64, device=dev)
    scal = wp.zeros(2, dtype=wp.float64, device=dev)
    wp.launch(matvec, n, inputs=[x, diag, g_off, nx], outputs=[ap], device=dev)
    wp.launch(cg_init, n, inputs=[b, ap, diag], outputs=[r, z, p, acc], device=dev)
    wp.launch(cg_rotate, 1, inputs=[acc, scal], device=dev)
    bnorm = math.sqrt(float(wp.utils.array_inner(b, b))) or 1.0
    if math.sqrt(max(float(scal.numpy()[1]), 0.0)) < tol * bnorm:
        return 0

    def iterations():
        for _ in range(chunk):
            wp.launch(cg_matvec_dot, n, inputs=[p, diag, g_off, nx], outputs=[ap, acc], device=dev)
            wp.launch(cg_update_xr, n, inputs=[p, ap, diag, scal], outputs=[acc, x, r, z], device=dev)
            wp.launch(cg_update_p, n, inputs=[z, scal, acc], outputs=[p], device=dev)
            wp.launch(cg_rotate, 1, inputs=[acc, scal], device=dev)

    use_graph = dev.is_cpu or dev.is_mempool_enabled
    graph = None
    if use_graph:
        with wp.ScopedCapture(device=dev) as cap:
            iterations()
        graph = cap.graph
    it = 0
    while it < max_iter:
        if graph is not None:
            wp.capture_launch(graph)
        else:
            iterations()
        it += chunk
        if math.sqrt(max(float(scal.numpy()[1]), 0.0)) < tol * bnorm:
            break
    return it


def _relax(dt: float, new: float, hist: tuple[float, float] | None) -> tuple[float, tuple[float, float]]:
    """One secant step on log(dT): the map dT -> dT_new behaves like C dT^-p, so with p estimated from the last
    two iterates the fixed point is log dT* = (log C) / (1 + p). Falls back to the geometric mean (p = 1)."""
    ld, ln = math.log(max(dt, 1e-6)), math.log(max(new, 1e-6))
    p = 1.0
    if hist is not None and abs(ld - hist[0]) > 1e-9:
        p = min(max(-(ln - hist[1]) / (ld - hist[0]), 0.0), 3.0)
    target = (ln + p * ld) / (1.0 + p)
    return math.exp(target), (ld, ln)


def solve(d: Design, dx_mm: float = 0.5, device: str = "cpu", tol: float = 1e-8, max_outer: int = 30) -> Result:
    """Steady temperature field for one design: fixed point on h(dT), one PCG solve per outer iteration.

    h depends on the surface excess temperature dT through Ra. In narrow channels h grows almost linearly with
    dT, so plain substitution oscillates; the update uses the geometric mean of the old and new dT, which
    converges in a few iterations in both the narrow-channel and the open-plate regimes.
    """
    t0 = time.perf_counter()
    mask, kinds = rasterize(d, dx_mm)
    ny, nx = mask.shape
    n = nx * ny
    dx = dx_mm * 1e-3
    length = d.length_mm * 1e-3
    q_flux = d.power_w / (d.source_mm * 1e-3 * length)
    ch_faces = ((kinds == CHANNEL).sum(axis=2) * mask).ravel()  # solid cells only: air cells hold theta = 0
    op_faces = ((kinds == OPEN).sum(axis=2) * mask).ravel()
    s_m = d.spacing_mm * 1e-3
    with wp.ScopedDevice(device):
        m = wp.array(mask.ravel().astype(np.int32), dtype=wp.int32)
        kd = wp.array(kinds.reshape(n, 4), dtype=wp.int32)
        diag = wp.zeros(n, dtype=wp.float64)
        g_off = wp.zeros((n, 4), dtype=wp.float64)
        g_conv = wp.zeros(n, dtype=wp.float64)
        b = wp.zeros(n, dtype=wp.float64)
        x = wp.zeros(n, dtype=wp.float64)
        dt_ch = dt_op = 30.0  # initial guess for the surface excess temperature [K]
        hist_ch: tuple[float, float] | None = None
        hist_op: tuple[float, float] | None = None
        cg_total, outer_iters = 0, 0
        for outer in range(1, max_outer + 1):
            outer_iters = outer
            hc, ho = h_channel(s_m, length, dt_ch), h_open(length, dt_op)
            wp.launch(
                assemble,
                n,
                inputs=[m, kd, wp.float64(K_AL), wp.float64(dx), wp.float64(hc), wp.float64(ho), wp.float64(q_flux)],
                outputs=[diag, g_off, g_conv, b],
            )
            cg_total += pcg(diag, g_off, b, x, nx, tol=tol)
            theta = x.numpy()
            new_ch = float((theta * ch_faces).sum() / max(ch_faces.sum(), 1))
            new_op = float((theta * op_faces).sum() / max(op_faces.sum(), 1))
            change = max(abs(new_ch / dt_ch - 1.0), abs(new_op / dt_op - 1.0))
            if change < 1e-4:
                break
            dt_ch, hist_ch = _relax(dt_ch, new_ch, hist_ch)
            dt_op, hist_op = _relax(dt_op, new_op, hist_op)
        q_out = float((g_conv.numpy() * theta).sum()) * length
        base = theta.reshape(ny, nx)[0]
        t_max = float(theta.max())
    return Result(
        t_max_rise_k=t_max,
        t_base_mean_rise_k=float(base.mean()),
        r_th_k_per_w=t_max / d.power_w,
        h_channel=hc,
        h_open=ho,
        energy_balance_err=(q_out - d.power_w) / d.power_w,
        outer_iters=outer_iters,
        cg_iters=cg_total,
        wall_s=time.perf_counter() - t0,
        solid_cells=int(mask.sum()),
    )


def temperature_field(d: Design, dx_mm: float = 0.5, device: str = "cpu") -> np.ndarray:
    """Converged excess-temperature field (ny, nx) [K], NaN in air cells (for plots)."""
    res = solve(d, dx_mm=dx_mm, device=device)
    mask, kinds = rasterize(d, dx_mm)
    ny, nx = mask.shape
    n = nx * ny
    length = d.length_mm * 1e-3
    with wp.ScopedDevice(device):
        m = wp.array(mask.ravel().astype(np.int32), dtype=wp.int32)
        kd = wp.array(kinds.reshape(n, 4), dtype=wp.int32)
        diag, g_conv, b, x = (wp.zeros(n, dtype=wp.float64) for _ in range(4))
        g_off = wp.zeros((n, 4), dtype=wp.float64)
        q_flux = d.power_w / (d.source_mm * 1e-3 * length)
        wp.launch(
            assemble,
            n,
            inputs=[
                m,
                kd,
                wp.float64(K_AL),
                wp.float64(dx_mm * 1e-3),
                wp.float64(res.h_channel),
                wp.float64(res.h_open),
                wp.float64(q_flux),
            ],
            outputs=[diag, g_off, g_conv, b],
        )
        pcg(diag, g_off, b, x, nx)
        theta = x.numpy().reshape(ny, nx)
    return np.where(mask, theta, np.nan)


def solve_row(d: Design, **kw) -> dict:
    """Flat dict of design + result, one CSV row."""
    r = solve(d, **kw)
    return {**asdict(d), "spacing_mm": round(d.spacing_mm, 4), **asdict(r)}


if __name__ == "__main__":
    wp.config.log_level = wp.LOG_WARNING
    for n in (4, 8, 12, 20):
        row = solve_row(Design(n_fins=n))
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
