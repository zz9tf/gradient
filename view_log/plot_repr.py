
import os, json, argparse, re, math
from typing import Optional, List, Tuple, Dict

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
        self.x_default = "step" if "step" in self.df.columns else ("iter" if "iter" in self.df.columns else None)
        self.repr_metrics = self._discover_repr_metrics()

    def _discover_repr_metrics(self) -> Dict[str, set]:
        pat = re.compile(rf"^{re.escape(self.stats_prefix)}repr\.([^.]+(?:\.[^.]+)*)\.([^.]+)$")
        out = {}
        for c in self.stats_cols:
            m = pat.match(c)
            if not m:
                continue
            block, metric = m.group(1), m.group(2)
            out.setdefault(block, set()).add(metric)
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

    def repr_blocks(self) -> List[str]:
        keep = []
        for b, metrics in self.repr_metrics.items():
            if b != "monitor" and len(metrics) >= 4:
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
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_repr_refined] Saved: {outpath}")


def _set_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)


def plot_line(ax, hub: TrainStatsHub, y: str, *, x=None, label=None, smooth_win=0, linewidth=1.8, alpha=0.95):
    try:
        xs, ys = hub.get_series(y=y, x=x, dropna=True)
    except KeyError:
        return False
    if smooth_win and smooth_win > 1:
        ys = moving_avg(ys, smooth_win)
    ax.plot(xs, ys, label=(label or y), linewidth=linewidth, alpha=alpha)
    return True


def repr_col(block: str, metric: str) -> str:
    return f"stats.repr.{block}.{metric}"


# -------------------------
# repr plot helpers
# -------------------------
def add_repr_derived(hub: TrainStatsHub):
    df = hub.df
    for block in hub.repr_blocks():
        Bc = repr_col(block, "B")
        er = repr_col(block, "erank")
        pr = repr_col(block, "prank")
        top3 = repr_col(block, "top3_ratio")
        tail = repr_col(block, "tail_ratio")
        l1 = repr_col(block, "lambda1_ratio")

        if Bc in df.columns and er in df.columns:
            df[repr_col(block, "erank_frac")] = pd.to_numeric(df[er], errors="coerce") / (
                pd.to_numeric(df[Bc], errors="coerce") + 1e-12
            )
        if Bc in df.columns and pr in df.columns:
            df[repr_col(block, "prank_frac")] = pd.to_numeric(df[pr], errors="coerce") / (
                pd.to_numeric(df[Bc], errors="coerce") + 1e-12
            )
        if top3 in df.columns and tail in df.columns:
            df[repr_col(block, "shape_balance")] = pd.to_numeric(df[tail], errors="coerce") - pd.to_numeric(df[top3], errors="coerce")
        if l1 in df.columns:
            df[repr_col(block, "collapse_index")] = pd.to_numeric(df[l1], errors="coerce")


def fig_repr_dashboard(hub: TrainStatsHub, outpath: str, smooth_win=31):
    add_repr_derived(hub)
    blocks = hub.repr_blocks()
    if not blocks:
        return

    backbone_blocks = [b for b in blocks if b.startswith("backbone.")]
    head_blocks = [b for b in blocks if not b.startswith("backbone.")]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.ravel()

    panels = [
        ("erank_frac", "Effective rank / B", "fraction", (0, 1.05)),
        ("prank_frac", "Participation rank / B", "fraction", (0, 1.05)),
        ("lambda1_ratio", "Lambda1 ratio", "ratio", (0, 1.05)),
        ("top3_ratio", "Top-3 energy ratio", "ratio", (0, 1.05)),
        ("tail_ratio", "Tail energy ratio", "ratio", (0, 1.05)),
        ("subspace_stab", "Subspace stability", "value", None),
    ]

    for ax, (metric, title, ylabel, ylim) in zip(axes, panels):
        ok = False
        for block in backbone_blocks:
            ok |= plot_line(ax, hub, repr_col(block, metric), label=block.replace("backbone.", ""), smooth_win=smooth_win, linewidth=1.9)
        for block in head_blocks:
            ok |= plot_line(ax, hub, repr_col(block, metric), label=block, smooth_win=smooth_win, linewidth=1.2, alpha=0.75)
        _set_ax(ax, title=title, xlabel=hub.x_default or "x", ylabel=ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if ok:
            ax.legend(fontsize=8, ncol=2)

    fig.suptitle("Representation Dashboard", fontsize=17, y=1.02)
    save_fig(fig, outpath)


def fig_repr_block_focus(hub: TrainStatsHub, block: str, outpath: str, smooth_win=31):
    add_repr_derived(hub)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()

    specs = [
        ("trace", "Trace (energy)", "trace", False),
        ("erank", "Effective rank", "rank", False),
        ("prank", "Participation rank", "rank", False),
        ("lambda1_ratio", "Lambda1 ratio", "ratio", False),
        ("top3_ratio", "Top-3 ratio", "ratio", False),
        ("subspace_stab", "Subspace stability", "value", False),
    ]

    for ax, (metric, title, ylabel, use_log) in zip(axes, specs):
        ok = plot_line(ax, hub, repr_col(block, metric), label=metric, smooth_win=smooth_win)
        _set_ax(ax, title=title, xlabel=hub.x_default or "x", ylabel=ylabel)
        if use_log:
            ax.set_yscale("log")
        if ok:
            # add faint raw curve underneath smoothed
            try:
                xs, ys = hub.get_series(repr_col(block, metric))
                ax.plot(xs, ys, linewidth=0.8, alpha=0.25)
            except Exception:
                pass

    fig.suptitle(f"Representation Focus: {block}", fontsize=16, y=1.02)
    save_fig(fig, outpath)


def fig_repr_heatmap(hub: TrainStatsHub, metric: str, outpath: str, smooth_win=31):
    add_repr_derived(hub)
    blocks = hub.repr_blocks()
    series = []
    keep_blocks = []
    xs_ref = None

    for block in blocks:
        col = repr_col(block, metric)
        if not hub.has(col):
            continue
        xs, ys = hub.get_series(col)
        ys = moving_avg(ys, smooth_win)
        if xs_ref is None:
            xs_ref = xs
        n = min(len(xs_ref), len(ys))
        series.append(ys[:n])
        keep_blocks.append(block)
        xs_ref = xs_ref[:n]

    if not series:
        return

    arr = np.vstack(series)
    fig, ax = plt.subplots(figsize=(13, max(4, 0.55 * len(keep_blocks))))
    im = ax.imshow(arr, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(keep_blocks)))
    ax.set_yticklabels(keep_blocks)
    ax.set_xticks(np.linspace(0, len(xs_ref) - 1, min(8, len(xs_ref))).astype(int))
    ax.set_xticklabels([f"{int(xs_ref[i])}" for i in np.linspace(0, len(xs_ref) - 1, min(8, len(xs_ref))).astype(int)])
    ax.set_xlabel(hub.x_default or "x")
    ax.set_title(f"Representation heatmap: {metric}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    save_fig(fig, outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="train.jsonl")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--smooth", type=int, default=31)
    args = ap.parse_args()

    hub = TrainStatsHub(args.input)
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.input)), "plots_repr_refined")
    os.makedirs(outdir, exist_ok=True)

    print(f"[plot_repr_refined] Input: {hub.path}")
    print(f"[plot_repr_refined] Rows: {len(hub.df)} | x={hub.x_default}")
    print(f"[plot_repr_refined] Repr blocks: {hub.repr_blocks()}")

    fig_repr_dashboard(hub, os.path.join(outdir, "repr_dashboard.png"), smooth_win=args.smooth)
    
    # plot focus figure for every repr block
    for block in hub.repr_blocks():
        safe_name = block.replace(".", "_")
        fig_repr_block_focus(
            hub,
            block,
            os.path.join(outdir, f"repr_focus_{safe_name}.png"),
            smooth_win=args.smooth,
        )

    fig_repr_heatmap(hub, "erank_frac", os.path.join(outdir, "repr_heatmap_erank_frac.png"), smooth_win=args.smooth)
    fig_repr_heatmap(hub, "subspace_stab", os.path.join(outdir, "repr_heatmap_subspace_stab.png"), smooth_win=args.smooth)

    print("[plot_repr] Done.")


if __name__ == "__main__":
    main()
