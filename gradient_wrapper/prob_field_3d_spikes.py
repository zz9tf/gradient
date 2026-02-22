import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def phi(z, beta=5.0, k=2.0):
    return (1.0 + beta * (1.0 - z)) ** (-k)


def _strength_palette():
    # weak -> strong  (low -> high)
    return [
        "#5e4fa2",  # purple (low)
        "#3288bd",
        "#66c2a5",
        "#abdda4",
        "#fee08b",
        "#f46d43",
        "#b40426",  # red (high)
    ]


def _field_cmap():
    return LinearSegmentedColormap.from_list("prob_field", _strength_palette(), N=256)


def _sample_uniform_sphere(rng):
    v = rng.standard_normal(3)
    return v / (np.linalg.norm(v) + 1e-12)


def plot_prob_field_3d_spikes(
    G3,
    strengths,
    beta=5.0,
    k=2.0,
    n_points=30000,
    reject_max=3000000,
    seed=0,
    point_size=4,
    alpha=0.25,
    spike_power=1.0,
    r_min=0.3,
    save_path="figures/prob_field_3d_spikes.png",
    stretch=4.0,          # 你原来 log1p(4*c) 的 “4”，我做成参数
):
    rng = np.random.default_rng(seed)
    cmap = _field_cmap()

    G3 = np.asarray(G3, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    T = G3.shape[0]

    G3 = G3 / (np.linalg.norm(G3, axis=1, keepdims=True) + 1e-12)

    Amax = strengths.sum() + 1e-12

    pts, avals = [], []
    proposals = 0

    while len(pts) < n_points and proposals < reject_max:
        proposals += 1
        v = _sample_uniform_sphere(rng)

        z = G3 @ v
        z = np.clip(z, -1.0, 1.0)
        A = np.sum(strengths * phi(z, beta=beta, k=k))
        acc = A / Amax
        acc = min(1.0, A / Amax)

        if rng.uniform(0.0, 1.0) < acc:
            pts.append(v)
            avals.append(A)

    pts = np.array(pts)
    avals = np.array(avals)

    if len(pts) == 0:
        raise RuntimeError("No samples accepted. Try smaller beta/k or larger reject_max.")

    # normalize A for [0,1]
    c = avals / (avals.max() + 1e-12)   # 或 quantile 0.99
    R = r_min + (1-r_min) * (c ** spike_power)   # 不做 log
    c = np.clip(c, 0.0, 1.0)

    # radius mapping using spike_power (as documented)
    R = r_min + (1.0 - r_min) * (c ** spike_power)

    X = R * pts[:, 0]
    Y = R * pts[:, 1]
    Z = R * pts[:, 2]

    fig = plt.figure(figsize=(7.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    norm = mpl.colors.Normalize(0, 1)
    colors = cmap(norm(c))

    # scatter (use explicit per-point RGBA)
    ax.scatter(
        X, Y, Z,
        color=colors,
        s=point_size,
        alpha=alpha,
        linewidths=0,
        zorder=2,
    )

    # ONE colorbar only (from ScalarMappable)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("normalized A(v) (red=high, purple=low)")

    # reference unit sphere wireframe
    uu = np.linspace(0, 2*np.pi, 70)
    vv = np.linspace(0, np.pi, 35)
    xs = np.outer(np.cos(uu), np.sin(vv))
    ys = np.outer(np.sin(uu), np.sin(vv))
    zs = np.outer(np.ones_like(uu), np.cos(vv))

    wf_color = cmap(0.0)
    ax.plot_wireframe(xs, ys, zs, rstride=6, cstride=6, linewidth=0.45, color="black", alpha=0.10, zorder=1)

    # task rays
    pal = _strength_palette()
    order = np.argsort(-strengths)
    task_color = [None] * T
    for rank, idx in enumerate(order):
        task_color[idx] = pal[min(rank, len(pal)-1)]

    smin, smax = strengths.min(), strengths.max()
    s_norm = (strengths - smin) / (smax - smin + 1e-12)
    arrow_len = 0.35 + 0.30 * s_norm

    for i in range(T):
        col = task_color[i]
        ax.plot([0, arrow_len[i] * G3[i, 0]],
                [0, arrow_len[i] * G3[i, 1]],
                [0, arrow_len[i] * G3[i, 2]],
                color=col, linewidth=2, alpha=0.95, zorder=6)
        ax.text(1.06 * arrow_len[i] * G3[i, 0],
                1.06 * arrow_len[i] * G3[i, 1],
                1.06 * arrow_len[i] * G3[i, 2],
                f"t{i}", color=col, fontsize=11, weight="bold", zorder=6)

    ax.set_title(f'3D "spikes" probability field (spike_power={spike_power})')
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    print(f"[saved] {save_path}")
    print(f"[stats] accepted={len(pts)} proposals={proposals} accept_rate={len(pts)/proposals:.4f}")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 6
    G3 = rng.standard_normal((T, 3))
    G3 = G3 / (np.linalg.norm(G3, axis=1, keepdims=True) + 1e-12)
    strengths = rng.uniform(0.5, 2.0, size=T)

    plot_prob_field_3d_spikes(
        G3, strengths,
        beta=5.0, k=2.0,
        n_points=30000,
        spike_power=1,          # try 0.8 / 1.0 / 1.5
        save_path="figures/prob_field_3d_spikes.png",
    )
