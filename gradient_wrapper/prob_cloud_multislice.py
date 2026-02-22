import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

def phi(z, beta=5.0, k=2.0):
    return (1.0 + beta * (1.0 - z)) ** (-k)

def draw_nice_arrow(ax, x, y, color, label, zorder=6):
    """
    Draw a nicer, thicker arrow with outline + readable label.
    x,y are arrow end coords in axis units.
    """
    arrow = FancyArrowPatch(
        (0.0, 0.0), (x, y),
        arrowstyle="-|>",          # clean filled head
        mutation_scale=18,         # head size (bigger => more obvious)
        linewidth=2.6,             # shaft thickness
        color=color,
        zorder=zorder,
    )
    # white outline so it pops on any background
    arrow.set_path_effects([
        pe.Stroke(linewidth=4.2, foreground="white", alpha=0.9),
        pe.Normal()
    ])
    ax.add_patch(arrow)

    # label with white halo
    txt = ax.text(
        x*1.10, y*1.10, label,
        color=color, fontsize=11, weight="bold",
        ha="center", va="center",
        zorder=zorder+1
    )
    txt.set_path_effects([
        pe.Stroke(linewidth=3.5, foreground="white", alpha=0.95),
        pe.Normal()
    ])

def _orthonormal_3d_subspace(rng, d: int):
    """Random orthonormal basis B in R^{d x 3}."""
    M = rng.standard_normal((d, 3))
    Q, _ = np.linalg.qr(M)  # Q: d x 3
    return Q


def _strength_palette():
    """
    Strong -> weak colors:
    red (high) ... purple (low)
    """
    return [
        "#b40426",  # red (strongest)
        "#f46d43",  # orange
        "#fee08b",  # pale yellow
        "#abdda4",  # light green
        "#66c2a5",  # teal
        "#3288bd",  # blue
        "#5e4fa2",  # purple (weakest)
    ]


def _field_cmap():
    """
    For probability field coloring:
    high = red, low = purple
    """
    colors = _strength_palette()
    return LinearSegmentedColormap.from_list("prob_field", colors, N=256)


def plot_prob_cloud_multislice_3d(
    d=200,
    T=6,
    beta=5.0,
    k=2.0,
    n_slices=9,
    n_points_per_slice=5000,
    reject_max_per_slice=200000,
    seed=0,
    point_size=6,
    alpha=0.75,
    save_path="figures/prob_cloud_multislice.png",
):
    """
    Visualize 3D-sphere probability field by many 2D great-circle slices.

    - High-D task directions g_hat in R^d
    - Randomly embed into a 3D subspace (for visualization)
    - For each slice angle psi, define great-circle plane spanned by:
        e1(psi) = cos(psi)*b1 + sin(psi)*b2
        e2      = b3
      and directions on that circle:
        v(theta, psi) = cos(theta)*e1(psi) + sin(theta)*e2
    - Define field:
        A(v) = sum_i a_i * phi(<g_i, v>)
      where a_i = w_i||g_i||  (here we simulate)
    - Sample points on circle with density ∝ A(v) via rejection sampling
    - Plot as a disk-cloud by adding random radius r (visual jitter)
    - Task arrows on each slice show in-plane projection of g_i, colored by strength rank
    """
    rng = np.random.default_rng(seed)

    # ---- colormap for field (red=high, purple=low)
    cmap = _field_cmap().reversed()

    # ---- simulate high-d task gradients and strengths
    def random_unit(d_):
        x = rng.standard_normal(d_)
        return x / (np.linalg.norm(x) + 1e-12)

    G_hd = np.stack([random_unit(d) for _ in range(T)], axis=0)  # [T,d]
    strengths = rng.uniform(0.5, 2.0, size=T)                    # a_i ~ w_i||g_i||

    # ---- embed into a random 3D subspace for sphere visualization
    B = _orthonormal_3d_subspace(rng, d)   # d x 3
    G3 = G_hd @ B                          # [T,3]
    G3 = G3 / (np.linalg.norm(G3, axis=1, keepdims=True) + 1e-12)  # unit in 3D

    # ---- build a stable 3D orthonormal basis (b1,b2,b3)
    # use QR on some matrix seeded by task vectors to make it consistent
    M = np.stack([G3[0], G3[min(1, T-1)], rng.standard_normal(3)], axis=1)  # 3x3
    Q, _ = np.linalg.qr(M)
    b1, b2, b3 = Q[:, 0], Q[:, 1], Q[:, 2]  # each is 3D

    # ---- assign task colors by strength rank (strongest red -> weakest purple)
    pal = _strength_palette()
    order = np.argsort(-strengths)  # descending
    task_color = [None] * T
    for rank, idx in enumerate(order):
        # if T > len(pal), wrap (or you can interpolate; wrap is ok for now)
        task_color[idx] = pal[min(rank, len(pal)-1)]

    # arrow lengths scaled by strength
    smin, smax = strengths.min(), strengths.max()
    s_norm = (strengths - smin) / (smax - smin + 1e-12)  # [0,1]
    arrow_len = 0.55 + 0.45 * s_norm

    # ---- slice angles (left -> right)
    psis = np.linspace(0.0, np.pi, n_slices, endpoint=False)  # half-turn is enough for planes
    fig, axes = plt.subplots(1, n_slices, figsize=(3.2*n_slices, 3.6), sharex=True, sharey=True)
    if n_slices == 1:
        axes = [axes]

    # acceptance upper bound
    Amax = strengths.sum() + 1e-12

    for ax, psi in zip(axes, psis):
        # plane basis for this slice
        e1 = np.cos(psi) * b1 + np.sin(psi) * b2   # 3D unit
        e2 = b3                                     # 3D unit (orthonormal to b1,b2 so ok)

        xs, ys, Aval = [], [], []
        proposals = 0

        while len(xs) < n_points_per_slice and proposals < reject_max_per_slice:
            proposals += 1

            # propose theta uniformly on circle
            th = rng.uniform(0.0, 2.0*np.pi)
            v3 = np.cos(th) * e1 + np.sin(th) * e2  # 3D unit direction on this great circle

            # field value A(v3)
            z = G3 @ v3  # [T], z_i = <g_i, v>
            A = np.sum(strengths * phi(z, beta=beta, k=k))
            acc = A / Amax

            if rng.uniform(0.0, 1.0) < acc:
                # visualize as disk cloud (radius jitter)
                r = np.sqrt(rng.uniform(0.0, 1.0))

                # 2D coordinates within the slice plane are simply (cos th, sin th)
                # because v3 = cos th * e1 + sin th * e2
                xs.append(r * np.cos(th))
                ys.append(r * np.sin(th))
                Aval.append(A)

        xs = np.array(xs)
        ys = np.array(ys)
        Aval = np.array(Aval)

        if len(xs) == 0:
            raise RuntimeError("No samples accepted in a slice. Try smaller beta/k or larger reject_max_per_slice.")

        c = Aval / (Aval.max() + 1e-12)
        c = np.clip(c, 0.0, 1.0)

        sc = ax.scatter(xs, ys, c=c, cmap=cmap, s=point_size, alpha=alpha, linewidths=0, zorder=2)

        # circle boundary
        tt = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(tt), np.sin(tt), color="black", lw=1)

        # draw task arrows: project task directions onto this plane
        # projection coords: (x_i, y_i) = (<g_i,e1>, <g_i,e2>)
        for i in range(T):
            px = float(np.dot(G3[i], e1))
            py = float(np.dot(G3[i], e2))
            norm = np.sqrt(px*px + py*py) + 1e-12
            px, py = px / norm, py / norm  # in-plane unit direction
            px *= arrow_len[i]
            py *= arrow_len[i]

            col = task_color[i]
            draw_nice_arrow(ax, px, py, col, f"t{i}", zorder=6)


        ax.set_aspect("equal")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$\psi={psi:.2f}$", fontsize=11)

        # (optional) print accept-rate per slice
        # print(f"[slice psi={psi:.3f}] accepted={len(xs)} proposals={proposals} rate={len(xs)/proposals:.4f}")

    # single shared colorbar
    fig.suptitle("3D sphere probability field via many 2D great-circle slices", y=1.02)

    # 先tight_layout，再手动留右边空间
    fig.tight_layout()
    fig.subplots_adjust(right=0.93)  # 给右侧留出放colorbar的空间

    # 手动创建 colorbar 轴: [left, bottom, width, height] (figure coords)
    cax = fig.add_axes([0.945, 0.18, 0.012, 0.64])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("normalized A(v) (red=high, purple=low)")


    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    print(f"[saved] {save_path}")


if __name__ == "__main__":
    plot_prob_cloud_multislice_3d()
