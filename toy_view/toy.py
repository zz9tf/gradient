import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Config
# =========================

@dataclass
class ToyConfig:
    # geometry
    theta0: np.ndarray
    theta_star: np.ndarray
    u_common: np.ndarray

    # NEW: per-task optima (mismatch optima)
    mus: np.ndarray            # shape (3,2), mu_i for each task

    # NEW: anisotropic curvature (high curvature on u_common)
    lam_common: float = 8.0    # big curvature along u_common
    lam_perp: float = 0.3      # small curvature on u_perp

    # simulation
    steps: int = 40
    lr: float = 0.15

    # task gradient structure
    conflict_strength: float = 0.0  # how strong task-specific components are
    conflict_rotate_deg: float = 35.0  # rotate conflict direction away from u_common

    # noise / low-quality data
    sigma: float = 0.03
    heavy_tail: bool = True
    p_spike: float = 0.12
    spike_scale: float = 10.0

    # "bad batch" burst (domain shift)
    burst_prob: float = 0.15
    burst_scale: float = 2.0

    # common gate (task-space rho gating)
    gate_ema_beta: float = 0.95
    gate_tau: float = 0.0       # strict rho > 0 => tau=0.0; allow tolerance => tau>0

    # plotting / saving
    save_dir: str = "toy_figs_3task"
    xlim: Tuple[float, float] = (-2.0, 5.0)
    ylim: Tuple[float, float] = (-3.5, 4.5)
    grid_n: int = 140


# =========================
# Utilities
# =========================
def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + eps)

def make_anisotropic_H(u_common: np.ndarray, lam_common: float, lam_perp: float) -> np.ndarray:
    """
    H = lam_common * u u^T + lam_perp * v v^T
    where v is unit vector orthogonal to u in 2D.
    """
    u = unit(u_common)
    v = unit(np.array([-u[1], u[0]], dtype=float))  # rotate 90deg
    return lam_common * np.outer(u, u) + lam_perp * np.outer(v, v)

def grad_anisotropic_quadratic(theta: np.ndarray, mu: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Gradient of 0.5*(theta-mu)^T H (theta-mu) is H*(theta-mu)."""
    return H @ (theta - mu)

def rotate(v: np.ndarray, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a)],
                  [np.sin(a),  np.cos(a)]], dtype=float)
    return R @ v

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 1.0
    return float(np.dot(a, b) / (na * nb))

def proj_scalar(v: np.ndarray, u: np.ndarray) -> float:
    return float(np.dot(v, u))

def remove_component(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Remove the component of v along unit vector u."""
    return v - proj_scalar(v, u) * u


# =========================
# Base toy "loss" (only for contour + reporting)
# =========================
def task_loss(theta: np.ndarray, mu: np.ndarray, H: np.ndarray) -> float:
    d = theta - mu
    return 0.5 * float(d @ (H @ d))

def mtl_loss(theta: np.ndarray, cfg: ToyConfig) -> float:
    H = make_anisotropic_H(cfg.u_common, cfg.lam_common, cfg.lam_perp)
    return float(np.mean([task_loss(theta, cfg.mus[i], H) for i in range(3)]))


# =========================
# Pre-sampling noise/bursts for fair comparison
# =========================

def pre_sample_noise_and_bursts(cfg: ToyConfig, seed: int) -> Dict[str, np.ndarray]:
    """
    Returns:
      noise: shape (steps, 3, 2)
      spike_mask: shape (steps, 3) bool
      burst_mask: shape (steps, 3) bool
    """
    rng = np.random.default_rng(seed)

    noise = rng.normal(0.0, cfg.sigma, size=(cfg.steps, 3, 2))

    if cfg.heavy_tail:
        spike_mask = rng.random(size=(cfg.steps, 3)) < cfg.p_spike
        spikes = rng.normal(0.0, cfg.sigma * cfg.spike_scale, size=(cfg.steps, 3, 2))
        noise = noise + spikes * spike_mask[..., None]
    else:
        spike_mask = np.zeros((cfg.steps, 3), dtype=bool)

    burst_mask = rng.random(size=(cfg.steps, 3)) < cfg.burst_prob

    return {"noise": noise, "spike_mask": spike_mask, "burst_mask": burst_mask}


# =========================
# 3-task gradient generator (pure 1st-order directions)
# =========================

def task_gradients(theta: np.ndarray, t: int, cfg: ToyConfig, presamp: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    3-task gradients (pure 1st-order):
      g_i = H*(theta - mu_i)  +  (optional) extra conflict along u_conf  + noise/burst
    where H is anisotropic: high curvature along u_common.

    This makes:
      (2) common direction high-curvature -> small mismatch causes large gradients
      (4) task optima mismatch -> tasks pull to different mu_i -> persistent conflict
    """
    u = cfg.u_common
    u_conf = unit(rotate(u, cfg.conflict_rotate_deg))

    # anisotropic curvature matrix (shared across tasks)
    H = make_anisotropic_H(u, cfg.lam_common, cfg.lam_perp)

    # task-specific coefficients (still keep, but you can lower it if mismatch is already strong)
    coeffs = np.array([+1.0, -1.0, +0.6], dtype=float)

    gs = []
    for i in range(3):
        mu_i = cfg.mus[i]  # <-- mismatch optima lives here

        # base anisotropic quadratic gradient toward mu_i
        g_base = grad_anisotropic_quadratic(theta, mu_i, H)

        # optional explicit conflict component (can keep, or set conflict_strength small)
        g = g_base + (cfg.conflict_strength * coeffs[i]) * u_conf

        # burst: amplify conflict only
        if presamp["burst_mask"][t, i]:
            # burst hits common direction (backbone drift)
            g = g_base + (cfg.conflict_strength * coeffs[i]) * u_conf + (cfg.burst_scale * coeffs[i]) * u

        # add noise
        g = g + presamp["noise"][t, i]
        gs.append(g)

    return gs

# =========================
# Aggregation rules
# =========================

def agg_sum(gs: List[np.ndarray]) -> np.ndarray:
    return np.mean(gs, axis=0)   # <- 原来是 np.sum

def agg_pcgrad(gs: List[np.ndarray], eps: float = 1e-12) -> np.ndarray:
    """
    PCGrad for T tasks:
      for each i:
        for each j != i:
          if dot(g_i, g_j) < 0: project g_i to remove component along g_j
      return sum(adjusted g_i)

    Deterministic order (0..T-1) for reproducibility.
    """
    T = len(gs)
    g_adj = [g.copy() for g in gs]

    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            gij = float(np.dot(g_adj[i], g_adj[j]))
            if gij < 0.0:
                denom = float(np.dot(g_adj[j], g_adj[j])) + eps
                g_adj[i] = g_adj[i] - (gij / denom) * g_adj[j]

    return np.mean(g_adj, axis=0)    # <- 原来 sum

def agg_graddrop(gs: List[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """
    GradDrop-style (toy, coordinate-wise):
      For each coordinate k:
        - if all nonzero task components share the same sign -> sum them
        - else -> randomly choose one task's component for that coordinate
    """
    G = np.stack(gs, axis=0)  # (T, 2)
    out = np.zeros((2,), dtype=float)

    for k in range(2):
        vals = G[:, k]
        # ignore exact zeros in sign check
        nz = vals[np.abs(vals) > 1e-12]
        if nz.size == 0:
            out[k] = 0.0
            continue

        signs = np.sign(nz)
        if np.all(signs == signs[0]):
            out[k] = float(np.sum(vals))
        else:
            idx = int(rng.integers(0, len(gs)))
            out[k] = float(vals[idx])

    return out / len(gs)

class CommonGateState:
    def __init__(self):
        # EMA direction in common subspace (2D vector, aligned with u)
        self.gpop_common = None  # shape (2,)

def agg_common_gate_sum_then_freeze(gs: List[np.ndarray],
                                   u_common: np.ndarray,
                                   state: CommonGateState,
                                   ema_beta: float,
                                   tau: float) -> Tuple[np.ndarray, float, int]:
    """
    CommonGate (per-task directional gating in common subspace):

      For each task i:
        g_i^c = <g_i, u> * u          (2D vector in common subspace)

      Maintain EMA direction:
        Gpop^c <- EMA(sum_i g_i^c)    (2D)

      Compute per-task alignment:
        rho_i = cos(g_i^c, Gpop^c)

      Gate:
        if min_i rho_i <= tau: freeze common-direction update
          => remove component of aggregated gradient along u

    Returns:
      g_final, rho (min rho_i), gate (1=allow, 0=freeze)
    """
    u = unit(u_common)

    # common-subspace vectors for each task (2D, collinear with u)
    gci = [proj_scalar(g, u) * u for g in gs]   # list length 3, each (2,)
    gc_sum = agg_sum(gci)                       # (2,)

    # init EMA direction (don't normalize; cosine() handles scale)
    if state.gpop_common is None:
        state.gpop_common = gc_sum.copy()

    # compute per-task rhos wrt current (old) EMA direction
    rhos = np.array([cosine(gci[i], state.gpop_common) for i in range(len(gs))], dtype=float)
    rho = float(np.min(rhos))

    # start from raw summed gradient (no surgery except possible freeze on u)
    g_sum = agg_sum(gs)

    gate = 1
    if rho <= tau:
        gate = 0
        g_sum = remove_component(g_sum, u)  # freeze ONLY along u
        # IMPORTANT: do NOT update EMA when frozen (avoid contaminating gpop)
        return g_sum, rho, gate

    # if allowed, update EMA using current common-sum direction
    state.gpop_common = ema_beta * state.gpop_common + (1.0 - ema_beta) * gc_sum
    return g_sum, rho, gate

# =========================
# Simulation
# =========================

def simulate(method: str, cfg: ToyConfig, presamp: Dict[str, np.ndarray], seed: int) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """
    Returns:
      path: (steps+1, 2)
      logs: dict of curves (loss, rho, gate, alignment)
    """
    theta = cfg.theta0.astype(float).copy()
    path = [theta.copy()]

    logs = {
        "loss": [],
        "rho": [],
        "gate": [],
        "cos_common": [],
    }

    rng = np.random.default_rng(seed)
    state = CommonGateState()

    for t in range(cfg.steps):
        gs = task_gradients(theta, t, cfg, presamp)

        if method == "sum":
            g = agg_sum(gs)
            rho, gate = np.nan, 1

        elif method == "pcgrad":
            g = agg_pcgrad(gs)
            rho, gate = np.nan, 1

        elif method == "graddrop":
            g = agg_graddrop(gs, rng)
            rho, gate = np.nan, 1

        elif method == "common_gate":
            g, rho, gate = agg_common_gate_sum_then_freeze(
                gs, cfg.u_common, state,
                ema_beta=cfg.gate_ema_beta,
                tau=cfg.gate_tau,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # logs (before update)
        logs["loss"].append(mtl_loss(theta, cfg))
        logs["rho"].append(float(rho) if np.isfinite(rho) else np.nan)
        logs["gate"].append(float(gate))
        logs["cos_common"].append(cosine(g, cfg.u_common))

        # GD update
        theta = theta - cfg.lr * g
        path.append(theta.copy())

    return np.array(path), logs


# =========================
# Plotting (save only)
# =========================

def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_landscape_and_paths(cfg: ToyConfig, results: Dict[str, Tuple[np.ndarray, Dict[str, List[float]]]]) -> None:
    xs = np.linspace(cfg.xlim[0], cfg.xlim[1], cfg.grid_n)
    ys = np.linspace(cfg.ylim[0], cfg.ylim[1], cfg.grid_n)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.zeros_like(XX, dtype=float)

    for i in range(cfg.grid_n):
        for j in range(cfg.grid_n):
            ZZ[i, j] = mtl_loss(np.array([XX[i, j], YY[i, j]]), cfg)

    plt.figure(figsize=(7, 6))
    plt.contour(XX, YY, ZZ, levels=25)

    for name, (path, _) in results.items():
        plt.plot(path[:, 0], path[:, 1], marker="o", markersize=3, linewidth=1.5, label=name)

    plt.scatter([cfg.theta0[0]], [cfg.theta0[1]], marker="s", s=80, label="start")
    plt.scatter([cfg.theta_star[0]], [cfg.theta_star[1]], marker="*", s=140, label="target")

    plt.title("Toy landscape (proxy) + GD trajectories (3 tasks)")
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.legend()

    _savefig(os.path.join(cfg.save_dir, "landscape_paths.png"))

def plot_loss_curve(cfg: ToyConfig, results: Dict[str, Tuple[np.ndarray, Dict[str, List[float]]]]) -> None:
    plt.figure(figsize=(7, 4))
    for name, (_, logs) in results.items():
        plt.plot(logs["loss"], label=name)
    plt.title(r"Proxy loss  $0.5\|\theta-\theta^*\|^2$  vs step")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    _savefig(os.path.join(cfg.save_dir, "loss_curve.png"))

def plot_rho_and_gate(cfg: ToyConfig, results: Dict[str, Tuple[np.ndarray, Dict[str, List[float]]]]) -> None:
    if "common_gate" not in results:
        return
    _, logs = results["common_gate"]
    rho = np.array(logs["rho"], dtype=float)
    gate = np.array(logs["gate"], dtype=float)

    plt.figure(figsize=(7, 4))
    plt.plot(rho, label="rho = min_i cos(g_i^c, Gpop^c)")
    plt.plot(gate, label="gate (1=update common, 0=freeze)")
    plt.title("CommonGate: task-space rho + gate")
    plt.xlabel("step")
    plt.legend()
    _savefig(os.path.join(cfg.save_dir, "rho_and_gate.png"))

def plot_alignment_common(cfg: ToyConfig, results: Dict[str, Tuple[np.ndarray, Dict[str, List[float]]]]) -> None:
    plt.figure(figsize=(7, 4))
    for name, (_, logs) in results.items():
        plt.plot(logs["cos_common"], label=name)
    plt.title(r"Alignment with common direction $u$:  cos(g, u)")
    plt.xlabel("step")
    plt.ylabel("cos")
    plt.legend()
    _savefig(os.path.join(cfg.save_dir, "alignment_common.png"))

def make_plots(cfg: ToyConfig, results: Dict[str, Tuple[np.ndarray, Dict[str, List[float]]]]) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    plot_landscape_and_paths(cfg, results)
    plot_loss_curve(cfg, results)
    plot_rho_and_gate(cfg, results)
    plot_alignment_common(cfg, results)


# =========================
# Main
# =========================

def main():
    u_common = unit(np.array([1.0, 0.35], dtype=float))

    # (4) mismatch optima: three tasks want different points
    mus = np.array([
        [ 1.2,  1.0],   # task1 optimum
        [-1.0,  1.1],   # task2 optimum
        [ 0.2, -1.2],   # task3 optimum
    ], dtype=float)

    cfg = ToyConfig(
        theta0=np.array([3.5, 3.5], dtype=float),

        # "global target" for proxy plots only (you can keep it, or set to average of mus)
        theta_star=np.array([0.0, 0.0], dtype=float),
        u_common=u_common,
        mus=mus,

        # (2) high curvature on common direction
        lam_common=12.0,
        lam_perp=0.25,

        # make conflict more visible
        lr=0.10,                 # with high curvature, smaller lr is safer
        steps=60,
        conflict_strength=0.35,  # mismatch already provides conflict; keep this moderate
        conflict_rotate_deg=20.0,

        # noise (optional; to show common_gate advantage under bad data)
        sigma=0.04,
        heavy_tail=True,
        p_spike=0.18,
        spike_scale=12.0,
        burst_prob=0.18,
        burst_scale=2.5,

        gate_tau=0.0,
        gate_ema_beta=0.95,

        save_dir="toy_figs_3task_hcurv_mismatch",
        xlim=(-2.5, 4.5),
        ylim=(-3.0, 4.5),
        grid_n=160,
    )

    presamp = pre_sample_noise_and_bursts(cfg, seed=2026)

    methods = ["sum", "pcgrad", "graddrop", "common_gate"]
    results = {}
    for m in methods:
        path, logs = simulate(m, cfg, presamp, seed=7)
        results[m] = (path, logs)

    make_plots(cfg, results)

if __name__ == "__main__":
    main()