import os, json, argparse, re, math
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# IO
# -------------------------
def read_jsonl_rows(path: str):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_rows_to_df(path: str, stats_key="stats") -> pd.DataFrame:
    rows = read_jsonl_rows(os.path.abspath(path))
    out = []
    for r in rows:
        base = {k: v for k, v in r.items() if k != stats_key}
        st = r.get(stats_key, {}) or {}
        for k, v in st.items():
            base[f"stats.{k}"] = v
        out.append(base)
    if not out:
        raise ValueError("Empty train.jsonl")
    df = pd.DataFrame(out)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df


class TrainStatsHub:
    def __init__(self, path: str, stats_prefix="stats."):
        self.path = os.path.abspath(path)
        self.run_dir = os.path.dirname(self.path)
        self.df = normalize_rows_to_df(self.path)

        self.stats_prefix = stats_prefix
        self.base_cols = [c for c in self.df.columns if not c.startswith(stats_prefix)]
        self.stats_cols = [c for c in self.df.columns if c.startswith(stats_prefix)]
        self.stats_keys = sorted([c[len(stats_prefix):] for c in self.stats_cols])
        self.x_default = "step" if "step" in self.df.columns else ("iter" if "iter" in self.df.columns else None)

        self.block_metrics = self._discover_block_metrics()

    def _discover_block_metrics(self):
        pat = re.compile(rf"^{re.escape(self.stats_prefix)}(pre|post)\.([^.]+(?:\.[^.]+)*)\.([^.]+)$")
        out = {}
        for c in self.stats_cols:
            m = pat.match(c)
            if not m:
                continue
            phase, block, metric = m.group(1), m.group(2), m.group(3)
            out.setdefault(block, {"pre": set(), "post": set()})
            out[block][phase].add(metric)
        return out

    def has(self, col: str) -> bool:
        return col in self.df.columns

    def resolve_col(self, y: str) -> str:
        if y in self.df.columns:
            return y
        if not y.startswith(self.stats_prefix) and (self.stats_prefix + y) in self.df.columns:
            return self.stats_prefix + y
        return y

    def get_series(self, y: str, x: Optional[str] = None, dropna=True):
        if x is None:
            if self.x_default is None:
                raise KeyError("Missing x-axis column")
            x = self.x_default
        y = self.resolve_col(y)
        if x not in self.df.columns or y not in self.df.columns:
            raise KeyError(f"Missing column: x='{x}' or y='{y}'")
        d = self.df[[x, y]].copy()
        if dropna:
            d = d.dropna()
        return d[x].to_numpy(), d[y].to_numpy(dtype=float)

    def block_list(self) -> List[str]:
        keep = []
        for b, d in self.block_metrics.items():
            if len(d.get("pre", set()) | d.get("post", set())) >= 4:
                keep.append(b)
        return sorted(keep)


def moving_avg(y, win=11):
    y = np.asarray(y, dtype=float)
    if win is None or win <= 1 or len(y) < 2:
        return y
    win = min(int(win), len(y))
    return pd.Series(y).rolling(window=win, center=True, min_periods=1).mean().to_numpy()


def save_fig(fig, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_blockwise] Saved: {outpath}")


def _set_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def plot_line(ax, hub: TrainStatsHub, y: str, *, x=None, label=None, smooth_win=0, linewidth=1.7, alpha=0.95):
    try:
        xs, ys = hub.get_series(y=y, x=x, dropna=True)
    except KeyError:
        return False
    if smooth_win and smooth_win > 1:
        ys = moving_avg(ys, smooth_win)
    ax.plot(xs, ys, label=(label or y), linewidth=linewidth, alpha=alpha)
    return True


def plot_group(ax, hub: TrainStatsHub, specs: List[Tuple[str, str]], *, title: str, ylabel: Optional[str] = None, smooth_win=0):
    ok = False
    for y, label in specs:
        ok |= plot_line(ax, hub, y, label=label, smooth_win=smooth_win)
    _set_ax(ax, title=title, xlabel=hub.x_default or "x", ylabel=ylabel)
    if ok:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
    return ok


def bcol(phase: str, block: str, metric: str) -> str:
    return f"stats.{phase}.{block}.{metric}"


def block_metric_pairs(block: str):
    return [
        ("norm_mean", f"{block}: norm mean", "value"),
        ("norm_cv", f"{block}: norm cv", "value"),
        ("norm_max_frac", f"{block}: norm max_frac", "ratio"),
        ("trace", f"{block}: trace", "value"),
        ("erank", f"{block}: erank", "value"),
        ("condish", f"{block}: condish", "value"),
        ("lambda1_ratio", f"{block}: lambda1 ratio", "ratio"),
        ("lambda2_ratio", f"{block}: lambda2 ratio", "ratio"),
        ("gmean_drift", f"{block}: gmean drift", "value"),
        ("gpop_drift", f"{block}: gpop drift", "value"),
        ("gpop_norm_ratio", f"{block}: gpop norm ratio", "ratio"),
        ("viol_frac", f"{block}: viol frac", "ratio"),
        ("gpop_neg_frac", f"{block}: gpop neg frac", "ratio"),
        ("gpop_rho_mean", f"{block}: gpop rho mean", "value"),
        ("eff_sum", f"{block}: eff sum", "value"),
        ("sum_norm", f"{block}: sum norm", "value"),
        ("sum_vec_norm", f"{block}: sum vec norm", "value"),
    ]


def available_block_panels(hub: TrainStatsHub, block: str):
    panels = []
    for metric, title, ylabel in block_metric_pairs(block):
        pre = bcol("pre", block, metric)
        post = bcol("post", block, metric)
        if hub.has(pre) or hub.has(post):
            panels.append((metric, title, ylabel))
    return panels


# -------------------------
# Figure 1: auto-split, one metric pair per panel
# -------------------------
def fig_block_summary(hub: TrainStatsHub, block: str, outpath: str, smooth_win=11, ncols=3):
    panels = available_block_panels(hub, block)
    n = len(panels)
    if n == 0:
        return
    ncols = max(1, int(ncols))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.4 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, (metric, title, ylabel) in zip(axes, panels):
        plot_group(
            ax,
            hub,
            [
                (bcol("pre", block, metric), f"pre {metric}"),
                (bcol("post", block, metric), f"post {metric}"),
            ],
            title=title,
            ylabel=ylabel,
            smooth_win=smooth_win,
        )

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Block Summary (auto-split): {block}", fontsize=16, y=1.01)
    save_fig(fig, outpath)


# -------------------------
# Figure 2: model-level performance / losses
# -------------------------
def fig_model_losses(hub: TrainStatsHub, outpath: str, smooth_win=11):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    plot_group(
        axes[0, 0], hub,
        [("loss_cls", "cls"), ("loss_rot", "rot"), ("loss_rec", "rec")],
        title="Task losses (raw scales)",
        ylabel="loss",
        smooth_win=smooth_win,
    )

    for col, label in [("loss_cls", "cls"), ("loss_rot", "rot"), ("loss_rec", "rec")]:
        if col in hub.df.columns:
            s = pd.to_numeric(hub.df[col], errors="coerce").astype(float)
            valid = s.dropna()
            if len(valid) == 0:
                continue
            s = s / (valid.iloc[0] + 1e-12)
            xs = hub.df[hub.x_default].to_numpy()
            ys = moving_avg(s.to_numpy(), smooth_win)
            axes[0, 1].plot(xs, ys, label=label)
    _set_ax(axes[0, 1], title="Task losses (normalized to first point)", xlabel=hub.x_default or "x", ylabel="relative loss")
    axes[0, 1].legend(fontsize=8)

    plot_group(
        axes[1, 0], hub,
        [("lr", "lr")],
        title="Learning rate",
        ylabel="lr",
        smooth_win=0,
    )

    plot_group(
        axes[1, 1], hub,
        [
            ("stats.pre.global.eff_sum", "pre global eff_sum"),
            ("stats.pre.global.norm_cv", "pre global norm_cv"),
            ("stats.pre.global.norm_max_frac", "pre global norm_max_frac"),
            ("stats.pre.global.sum_vec_norm", "pre global sum_vec_norm"),
        ],
        title="Model-level geometry snapshot",
        ylabel="value",
        smooth_win=smooth_win,
    )

    fig.suptitle("Model View: optimization / performance", fontsize=16, y=1.02)
    save_fig(fig, outpath)


# -------------------------
# Figure 3: gpop-policy-only figure
# -------------------------
def fig_gpop_policy(hub: TrainStatsHub, outpath: str, smooth_win=11, policy="cov_inv"):
    base = f"stats.common_gpop_surgery.{policy}"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    plot_group(
        axes[0, 0], hub,
        [
            (f"{base}.dot.mean", "dot.mean"),
            (f"{base}.dot.min", "dot.min"),
            (f"{base}.dot.neg_frac", "dot.neg_frac"),
        ],
        title="Projection statistics onto v_ref",
        ylabel="value",
        smooth_win=smooth_win,
    )

    plot_group(
        axes[0, 1], hub,
        [
            (f"{base}.v_ref_norm", "v_ref_norm"),
            (f"{base}.g_pop_norm", "g_pop_norm"),
        ],
        title="Reference direction vs gpop norm",
        ylabel="norm",
        smooth_win=smooth_win,
    )

    plot_group(
        axes[1, 0], hub,
        [
            (f"{base}.applied", "applied"),
            (f"{base}.denom_used", "denom_used"),
            (f"{base}.centered_used", "centered_used"),
            (f"{base}.damping", "damping"),
        ],
        title="Policy runtime/config signals",
        ylabel="value",
        smooth_win=0,
    )

    plot_group(
        axes[1, 1], hub,
        [
            (f"{base}.dot.neg_frac", "policy dot.neg_frac"),
            ("stats.post.global.viol_frac", "post global viol_frac"),
            ("stats.post.global.eff_sum", "post global eff_sum"),
            ("stats.post.global.sum_vec_norm", "post global sum_vec_norm"),
        ],
        title="Policy stats vs post-global geometry",
        ylabel="value",
        smooth_win=smooth_win,
    )

    fig.suptitle(f"Gpop Policy Diagnostics: {policy}", fontsize=16, y=1.02)
    save_fig(fig, outpath)


# -------------------------
# Summary dump
# -------------------------
def dump_summary(hub: TrainStatsHub, outpath: str):
    lines = []
    lines.append(f"Input: {hub.path}")
    lines.append(f"Rows: {len(hub.df)}")
    lines.append(f"x_default: {hub.x_default}")
    lines.append("")
    lines.append("Blocks discovered:")
    for b in hub.block_list():
        pre = sorted(hub.block_metrics[b]["pre"])
        post = sorted(hub.block_metrics[b]["post"])
        lines.append(f"- {b}")
        lines.append(f"    pre : {', '.join(pre)}")
        lines.append(f"    post: {', '.join(post)}")
    lines.append("")
    lines.append("Policy cols:")
    for c in sorted([c for c in hub.stats_cols if "common_gpop_surgery" in c]):
        lines.append(f"- {c}")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"[plot_blockwise] Saved: {outpath}")


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="train.jsonl")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--smooth", type=int, default=100)
    ap.add_argument("--policy", type=str, default="cov_inv")
    ap.add_argument("--block-cols", type=int, default=3)
    args = ap.parse_args()

    hub = TrainStatsHub(args.input)
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.input)), "plots_blockwise_auto_split")
    os.makedirs(outdir, exist_ok=True)

    print(f"[plot_blockwise] Input: {hub.path}")
    print(f"[plot_blockwise] Rows: {len(hub.df)} | stats cols: {len(hub.stats_cols)} | x={hub.x_default}")
    print(f"[plot_blockwise] Blocks: {hub.block_list()}")

    for block in hub.block_list():
        safe = block.replace('.', '_')
        fig_block_summary(
            hub,
            block,
            os.path.join(outdir, f"block_{safe}_summary.png"),
            smooth_win=args.smooth,
            ncols=args.block_cols,
        )

    fig_model_losses(hub, os.path.join(outdir, "model_losses.png"), smooth_win=args.smooth)
    fig_gpop_policy(hub, os.path.join(outdir, f"gpop_policy_{args.policy}.png"), smooth_win=args.smooth, policy=args.policy)
    dump_summary(hub, os.path.join(outdir, "summary.txt"))
    print("[plot_blockwise] Done.")


if __name__ == "__main__":
    main()
