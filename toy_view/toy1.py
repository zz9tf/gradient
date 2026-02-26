import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
@dataclass
class Cfg:
    # problem size
    d: int = 2
    n_tasks: int = 3

    # quadratic curvature: Lambda eigenvalues (stiff + flat)
    lam1: float = 12.0
    lam2: float = 0.4

    # per-task rotation angles (deg). If None -> evenly spaced.
    angles_deg: Optional[List[float]] = None

    # per-task optima mu_i (shape n_tasks x d). If None -> random.
    mus: Optional[np.ndarray] = None
    mu_scale: float = 1.2

    # optimization
    steps: int = 80
    lr: float = 0.08
    theta0: np.ndarray = np.array([3.2, 3.0], dtype=float)

    # stochastic gradient noise
    sigma: float = 0.03
    heavy_tail: bool = True
    p_spike: float = 0.12
    spike_scale: float = 12.0

    # burst / domain shift (per-task occasional bias)
    burst_prob: float = 0.10
    burst_scale: float = 1.5

    # common direction for gating (unit vector in R^d)
    u_common: np.ndarray = np.array([1.0, 0.35], dtype=float)

    # gate
    gate_tau: float = 0.0      # require min cos > tau to update common
    gate_ema_beta: float = 0.99

    # plotting
    save_dir: str = "toy_rotquad"
    xlim: Tuple[float, float] = (-3.0, 4.8)
    ylim: Tuple[float, float] = (-3.5, 4.5)
    grid_n: int = 160


# -----------------------------
# Utils
# -----------------------------
def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + eps)

def rot2(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]], dtype=float)

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0  # IMPORTANT: neutral, don't auto-pass gate
    return float(np.dot(a, b) / (na * nb))

def remove_component(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    u = unit(u)
    return v - float(np.dot(v, u)) * u


# -----------------------------
# Build tasks: H_i = R_i^T Lambda R_i
# -----------------------------
def build_tasks(cfg: Cfg, seed: int = 0) -> Tuple[List[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    d = cfg.d
    assert d == 2, "This simple script is written for d=2 for plotting clarity."

    # angles
    if cfg.angles_deg is None:
        # evenly spread
        angles = np.linspace(-35, 35, cfg.n_tasks).tolist()
    else:
        angles = cfg.angles_deg
        assert len(angles) == cfg.n_tasks

    # Lambda
    Lam = np.diag([cfg.lam1, cfg.lam2]).astype(float)

    Hs = []
    for a in angles:
        R = rot2(a)
        H = R.T @ Lam @ R
        # symmetrize (numerical)
        H = 0.5 * (H + H.T)
        Hs.append(H)

    # mus
    if cfg.mus is None:
        mus = rng.normal(0.0, cfg.mu_scale, size=(cfg.n_tasks, d))
    else:
        mus = np.array(cfg.mus, dtype=float)
        assert mus.shape == (cfg.n_tasks, d)

    return Hs, mus


# -----------------------------
# Task loss + gradient
# -----------------------------
def task_loss(theta: np.ndarray, H: np.ndarray, mu: np.ndarray) -> float:
    d = theta - mu
    return 0.5 * float(d @ (H @ d))

def task_grad(theta: np.ndarray, H: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return H @ (theta - mu)

def avg_loss(theta: np.ndarray, Hs: List[np.ndarray], mus: np.ndarray) -> float:
    return float(np.mean([task_loss(theta, Hs[i], mus[i]) for i in range(len(Hs))]))


# -----------------------------
# Pre-sample noise for fairness
# -----------------------------
def presample_noise(cfg: Cfg, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, cfg.sigma, size=(cfg.steps, cfg.n_tasks, cfg.d))

    spike_mask = np.zeros((cfg.steps, cfg.n_tasks), dtype=bool)
    if cfg.heavy_tail:
        spike_mask = rng.random((cfg.steps, cfg.n_tasks)) < cfg.p_spike
        spikes = rng.normal(0.0, cfg.sigma * cfg.spike_scale, size=noise.shape)
        noise = noise + spikes * spike_mask[..., None]

    burst_mask = rng.random((cfg.steps, cfg.n_tasks)) < cfg.burst_prob
    burst_dir = unit(cfg.u_common)

    return {
        "noise": noise,
        "spike_mask": spike_mask,
        "burst_mask": burst_mask,
        "burst_dir": burst_dir,
    }


# -----------------------------
# Build stochastic per-task gradients g_i(theta,t)
# -----------------------------
def task_gradients(theta: np.ndarray, t: int, Hs: List[np.ndarray], mus: np.ndarray,
                   cfg: Cfg, pre: Dict[str, np.ndarray]) -> List[np.ndarray]:
    gs = []
    for i in range(cfg.n_tasks):
        g = task_grad(theta, Hs[i], mus[i])

        # burst: add a bias along u_common (domain shift / backbone drift)
        if pre["burst_mask"][t, i]:
            g = g + (cfg.burst_scale * (1.0 if i % 2 == 0 else -1.0)) * pre["burst_dir"]

        # add noise
        g = g + pre["noise"][t, i]
        gs.append(g)
    return gs


# -----------------------------
# Aggregators
# -----------------------------
def agg_mean(gs: List[np.ndarray]) -> np.ndarray:
    return np.mean(gs, axis=0)

def agg_pcgrad(gs: List[np.ndarray], eps: float = 1e-12) -> np.ndarray:
    T = len(gs)
    g_adj = [g.copy() for g in gs]
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            dot = float(np.dot(g_adj[i], g_adj[j]))
            if dot < 0.0:
                denom = float(np.dot(g_adj[j], g_adj[j])) + eps
                g_adj[i] = g_adj[i] - (dot / denom) * g_adj[j]
    return np.mean(g_adj, axis=0)

class GateState:
    def __init__(self):
        self.gpop_common: Optional[np.ndarray] = None  # EMA direction in common-subspace

def agg_common_gate(gs: List[np.ndarray], cfg: Cfg, st: GateState) -> Tuple[np.ndarray, float, int]:
    """
    CommonGate idea (simple):
      - Project each task gradient to common direction u (1D subspace)
      - Track EMA gpop_common in that 1D subspace (still 2D vector collinear with u)
      - If any task's projected grad disagrees (cos <= tau), freeze common component in update
    """
    u = unit(cfg.u_common)

    # per-task common projection (2D vectors collinear with u)
    gci = [float(np.dot(g, u)) * u for g in gs]
    gc = np.mean(gci, axis=0)

    if st.gpop_common is None:
        st.gpop_common = gc.copy()

    # min alignment across tasks
    rhos = np.array([cosine(gci[i], st.gpop_common) for i in range(len(gs))], dtype=float)
    rho = float(np.min(rhos))

    g = np.mean(gs, axis=0)
    # update EMA only when allowed
    st.gpop_common = cfg.gate_ema_beta * st.gpop_common + (1.0 - cfg.gate_ema_beta) * gc

    if rho <= cfg.gate_tau:
        # freeze ONLY common direction component
        g = remove_component(g, u)
        return g, rho, 0

    
    return g, rho, 1


# -----------------------------
# Simulation
# -----------------------------
def simulate(method: str, cfg: Cfg, Hs: List[np.ndarray], mus: np.ndarray, pre: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    theta = cfg.theta0.astype(float).copy()
    path = [theta.copy()]
    logs = {"loss": [], "rho": [], "gate": [], "minpair_cos": []}

    st = GateState()

    for t in range(cfg.steps):
        gs = task_gradients(theta, t, Hs, mus, cfg, pre)

        # conflict metric (pairwise min cosine)
        pair = []
        for i in range(cfg.n_tasks):
            for j in range(i + 1, cfg.n_tasks):
                pair.append(cosine(gs[i], gs[j]))
        logs["minpair_cos"].append(float(np.min(pair)) if len(pair) else 1.0)

        if method == "mean":
            g = agg_mean(gs)
            rho, gate = np.nan, 1
        elif method == "pcgrad":
            g = agg_pcgrad(gs)
            rho, gate = np.nan, 1
        elif method == "common_gate":
            g, rho, gate = agg_common_gate(gs, cfg, st)
        else:
            raise ValueError(method)

        logs["loss"].append(avg_loss(theta, Hs, mus))
        logs["rho"].append(float(rho) if np.isfinite(rho) else np.nan)
        logs["gate"].append(float(gate))

        theta = theta - cfg.lr * g
        path.append(theta.copy())

    return np.array(path), logs


# -----------------------------
# Plotting
# -----------------------------
def plot_all(cfg: Cfg, Hs: List[np.ndarray], mus: np.ndarray, results: Dict[str, Tuple[np.ndarray, Dict[str, List[float]]]]):
    os.makedirs(cfg.save_dir, exist_ok=True)

    # contour of average loss
    xs = np.linspace(cfg.xlim[0], cfg.xlim[1], cfg.grid_n)
    ys = np.linspace(cfg.ylim[0], cfg.ylim[1], cfg.grid_n)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.zeros_like(XX, dtype=float)
    for i in range(cfg.grid_n):
        for j in range(cfg.grid_n):
            ZZ[i, j] = avg_loss(np.array([XX[i, j], YY[i, j]]), Hs, mus)

    plt.figure(figsize=(7, 6))
    plt.contour(XX, YY, ZZ, levels=28)

    for name, (path, _) in results.items():
        plt.plot(path[:, 0], path[:, 1], marker="o", markersize=2.6, linewidth=1.4, label=name)

    plt.scatter([cfg.theta0[0]], [cfg.theta0[1]], s=80, marker="s", label="start")
    # show task optima
    plt.scatter(mus[:, 0], mus[:, 1], s=70, marker="x", label="mus (task optima)")

    plt.title("Rotated Quadratic MTL: contours + trajectories")
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "traj.png"), dpi=250)
    plt.close()

    # loss curve
    plt.figure(figsize=(7, 4))
    for name, (_, logs) in results.items():
        plt.plot(logs["loss"], label=name)
    plt.title("Average loss vs step")
    plt.xlabel("step")
    plt.ylabel("avg loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "loss.png"), dpi=250)
    plt.close()

    # min pairwise cosine (conflict)
    plt.figure(figsize=(7, 4))
    for name, (_, logs) in results.items():
        plt.plot(logs["minpair_cos"], label=name)
    plt.title("Conflict: min pairwise cosine(cos(g_i, g_j))")
    plt.xlabel("step")
    plt.ylabel("min cosine")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "conflict.png"), dpi=250)
    plt.close()

    # gate curve
    if "common_gate" in results:
        _, logs = results["common_gate"]
        plt.figure(figsize=(7, 4))
        plt.plot(logs["rho"], label="rho (min cos in common-subspace)")
        plt.plot(logs["gate"], label="gate (1=allow,0=freeze)")
        plt.title("CommonGate signals")
        plt.xlabel("step")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.save_dir, "gate.png"), dpi=250)
        plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = Cfg(
        n_tasks=3,
        angles_deg=[-25, 0, 25],     # rotate bowls to create conflict
        lam1=14.0, lam2=0.35,        # stiff + flat
        mu_scale=1.3,                # mismatch optima strength
        steps=90,
        lr=0.07,
        sigma=0.04,
        heavy_tail=True,
        p_spike=0.15,
        spike_scale=10.0,
        burst_prob=0.12,
        burst_scale=1.6,
        gate_tau=0.0,
        gate_ema_beta=0.95,
        save_dir="toy_rotquad_simple",
    )

    cfg.u_common = unit(cfg.u_common)

    Hs, mus = build_tasks(cfg, seed=2026)
    pre = presample_noise(cfg, seed=2027)

    methods = ["mean", "pcgrad", "common_gate"]
    results = {}
    for m in methods:
        path, logs = simulate(m, cfg, Hs, mus, pre)
        results[m] = (path, logs)

    plot_all(cfg, Hs, mus, results)
    print(f"Saved figures to: {cfg.save_dir}/ (traj.png, loss.png, conflict.png, gate.png)")

if __name__ == "__main__":
    main()