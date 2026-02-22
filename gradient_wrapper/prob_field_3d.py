import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def phi(z, beta=5.0, k=2.0):
    return (1.0 + beta * (1.0 - z)) ** (-k)


def _strength_palette():
    return [
        "#b40426",  # red (high)
        "#f46d43",
        "#fee08b",
        "#abdda4",
        "#66c2a5",
        "#3288bd",
        "#5e4fa2",  # purple (low)
    ]


def _field_cmap():
    colors = _strength_palette()
    return LinearSegmentedColormap.from_list("prob_field", colors, N=256).reversed()


def _sample_uniform_sphere(rng):
    """Uniform on S^2."""
    x = rng.standard_normal(3)
    return x / (np.linalg.norm(x) + 1e-12)


def plot_prob_field_3d(
    G3,                    # [T,3] unit vectors
    strengths,             # [T]
    beta=5.0,
    k=2.0,
    n_points=25000,
    reject_max=2000000,
    seed=0,
    point_size=4,
    alpha=0.9,
    radial_mode="sphere",  # "sphere" or "spikes"
    spike_power=1.0,       # only used when radial_mode="spikes"
    save_path="figures/prob_field_3d.png",
):
    """
    3D visualization of probability field on the sphere.

    radial_mode:
      - "sphere": plot points on unit sphere, color by A(v)
      - "spikes": plot points with radius r ∝ A(v)^spike_power (hedgehog)
    """
    rng = np.random.default_rng(seed)
    cmap = _field_cmap()

    T = G3.shape[0]
    Amax = strengths.sum() + 1e-12  # since phi<=1

    pts = []
    avals = []
    proposals = 0

    while len(pts) < n_points and proposals < reject_max:
        proposals += 1
        v = _sample_uniform_sphere(rng)       # candidate direction on S^2
        z = G3 @ v                            # [T]
        A = np.sum(strengths * phi(z, beta=beta, k=k))
        acc = A / Amax

        if rng.uniform(0.0, 1.0) < acc:
            pts.append(v)
            avals.append(A)

    pts = np.array(pts)
    avals = np.array(avals)

    if len(pts) == 0:
        raise RuntimeError("No samples accepted. Try smaller beta/k or larger reject_max.")

    # global normalize to [0,1]
    c = avals / (Amax + 1e-12)
    c = np.clip(c, 0.0, 1.0)

    # radius choice
    if radial_mode == "sphere":
        R = np.ones(len(pts))
    elif radial_mode == "spikes":
        R = (c ** spike_power)
        # keep a minimum radius so points don't collapse at origin
        R = 0.15 + 0.95 * R
    else:
        raise ValueError("radial_mode must be 'sphere' or 'spikes'")

    X = R * pts[:, 0]
    Y = R * pts[:, 1]
    Z = R * pts[:, 2]

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(X, Y, Z, c=c, cmap=cmap, s=point_size, alpha=alpha, linewidths=0)

    # draw a faint wireframe sphere for reference
    uu = np.linspace(0, 2*np.pi, 40)
    vv = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(uu), np.sin(vv))
    ys = np.outer(np.sin(uu), np.sin(vv))
    zs = np.outer(np.ones_like(uu), np.cos(vv))
    ax.plot_wireframe(xs, ys, zs, rstride=4, cstride=4, linewidth=0.4, alpha=0.15)

    # plot task directions as arrows from origin
    pal = _strength_palette()  # strongest red -> weakest purple
    order = np.argsort(-strengths)
    task_color = [None] * T
    for rank, idx in enumerate(order):
        task_color[idx] = pal[min(rank, len(pal)-1)]

    # arrow length in 3D
    smin, smax = strengths.min(), strengths.max()
    s_norm = (strengths - smin) / (smax - smin + 1e-12)
    arrow_len = 0.65 + 0.45 * s_norm

    for i in range(T):
        col = task_color[i]
        ax.plot([0, arrow_len[i]*G3[i,0]],
                [0, arrow_len[i]*G3[i,1]],
                [0, arrow_len[i]*G3[i,2]],
                color=col, linewidth=2.5)
        ax.text(1.05*arrow_len[i]*G3[i,0],
                1.05*arrow_len[i]*G3[i,1],
                1.05*arrow_len[i]*G3[i,2],
                f"t{i}", color=col, fontsize=11, weight="bold")

    ax.set_title(f"3D sampled directions on sphere (mode={radial_mode})")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)

    # cleaner axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("normalized A(v) (red=high, purple=low)")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    print(f"[saved] {save_path}")
    print(f"[stats] accepted={len(pts)} proposals={proposals} accept_rate={len(pts)/proposals:.4f}")


# 示例：如果你想在你原函数里拿到 G3/strengths 后调用
if __name__ == "__main__":
    # 这里用随机示例
    rng = np.random.default_rng(0)
    T = 6
    G3 = rng.standard_normal((T, 3))
    G3 = G3 / (np.linalg.norm(G3, axis=1, keepdims=True) + 1e-12)
    strengths = rng.uniform(0.5, 2.0, size=T)

    plot_prob_field_3d(G3, strengths, radial_mode="sphere")
    plot_prob_field_3d(G3, strengths, radial_mode="spikes", save_path="figures/prob_field_3d_spikes.png")
