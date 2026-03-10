"""
plot_compare_runs.py

Compare multiple CIFAR-MTL runs under a folder (e.g. mtl_cifar/runs_cifar/*).

This script is intentionally aligned with plot_all.py:
- Same JSONL flattening of `stats` into `stats.<key>` columns.
- Same moving-average smoothing.

Outputs:
- compare_losses.png
- compare_losses_normalized.png
- compare_valid_metrics.png
- compare_global_geometry.png
- compare_gpop_policy.png (if policy columns exist)
- runs_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_jsonl_rows(path: str) -> List[Dict]:
    """Read a JSONL file into a list of dict rows."""
    rows: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_rows_to_df(path: str, stats_key: str = "stats") -> pd.DataFrame:
    """
    Flatten JSONL rows into a DataFrame.

    Keys under `stats` are flattened as columns: stats.<key>.
    Numeric columns are coerced where possible.
    """
    rows = read_jsonl_rows(os.path.abspath(path))
    out: List[Dict] = []
    for r in rows:
        base = {k: v for k, v in r.items() if k != stats_key}
        st = r.get(stats_key, {}) or {}
        for k, v in st.items():
            base[f"stats.{k}"] = v
        out.append(base)
    if not out:
        raise ValueError(f"Empty jsonl: {path}")
    df = pd.DataFrame(out)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df


def moving_avg(y: Sequence[float], win: int = 11) -> np.ndarray:
    """Centered moving average with min_periods=1 (same behavior as plot_all.py)."""
    y = np.asarray(y, dtype=float)
    if win is None or win <= 1 or len(y) < 2:
        return y
    win = min(int(win), len(y))
    return pd.Series(y).rolling(window=win, center=True, min_periods=1).mean().to_numpy()


@dataclass(frozen=True)
class RunInfo:
    """In-memory container for one run's data."""

    name: str
    run_dir: str
    train_path: str
    eval_path: Optional[str]
    config_path: Optional[str]
    cfg: Dict
    train_df: pd.DataFrame
    eval_df: Optional[pd.DataFrame]


def _safe_get(d: Dict, keys: Sequence[str], default=None):
    """Get a nested dict value by a path of keys."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_config(path: str) -> Dict:
    """Load a JSON config file. Returns empty dict if missing."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_run(run_dir: str) -> RunInfo:
    """Load one run folder: train.jsonl (required), eval.jsonl/config.json (optional)."""
    run_dir = os.path.abspath(run_dir)
    name = os.path.basename(run_dir)

    train_path = os.path.join(run_dir, "train.jsonl")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train.jsonl in {run_dir}")

    eval_path = os.path.join(run_dir, "eval.jsonl")
    if not os.path.exists(eval_path):
        eval_path = None

    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        config_path = None

    cfg = load_config(config_path) if config_path else {}

    train_df = normalize_rows_to_df(train_path)
    eval_df = None
    if eval_path:
        eval_df = pd.DataFrame(read_jsonl_rows(eval_path))
        for c in eval_df.columns:
            try:
                eval_df[c] = pd.to_numeric(eval_df[c])
            except Exception:
                pass

    return RunInfo(
        name=name,
        run_dir=run_dir,
        train_path=train_path,
        eval_path=eval_path,
        config_path=config_path,
        cfg=cfg,
        train_df=train_df,
        eval_df=eval_df,
    )


def discover_runs(runs_dir: str, include: Optional[str] = None, exclude: Optional[str] = None) -> List[str]:
    """
    Discover run directories under runs_dir.

    A directory is considered a run if it contains train.jsonl.
    include/exclude are regex patterns matched against run directory name.
    """
    runs_dir = os.path.abspath(runs_dir)
    if not os.path.isdir(runs_dir):
        raise NotADirectoryError(runs_dir)

    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None

    out: List[str] = []
    for p in sorted(Path(runs_dir).iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if inc and not inc.search(name):
            continue
        if exc and exc.search(name):
            continue
        if (p / "train.jsonl").exists():
            out.append(str(p.resolve()))
    return out


def run_label(ri: RunInfo) -> str:
    """Build a compact label: run_name | mode | gpop/policy."""
    grad_mode = str(ri.cfg.get("grad_mode", ri.train_df.get("mode", ["?"])[0] if "mode" in ri.train_df else "?"))
    gpop = bool(_safe_get(ri.cfg, ["cli", "gpop"], False))
    policy = str(_safe_get(ri.cfg, ["cli", "gpop_policy"], "none"))
    if gpop and policy and policy != "none":
        return f"{ri.name} ({grad_mode}, gpop={policy})"
    return f"{ri.name} ({grad_mode})"


def _get_xy(df: pd.DataFrame, x_col: str, y_col: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Get numeric x/y arrays; returns None if columns missing or empty."""
    if x_col not in df.columns or y_col not in df.columns:
        return None
    x = pd.to_numeric(df[x_col], errors="coerce").astype(float).to_numpy()
    y = pd.to_numeric(df[y_col], errors="coerce").astype(float).to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return None
    return x[m], y[m]


def split_eval_df(df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Split eval DataFrame into per-epoch validation rows and final test rows.

    Validation rows are numeric epochs with `acc_cls_valid`.
    Final rows are rows containing at least one `*_test` metric.
    """
    if df is None or df.empty:
        return None, None

    valid_df = None
    final_df = None

    if "epoch" in df.columns:
        epoch_num = pd.to_numeric(df["epoch"], errors="coerce")
        valid_mask = epoch_num.notna() & df.columns.isin(["acc_cls_valid"]).any()
        if valid_mask.any() and "acc_cls_valid" in df.columns:
            valid_df = df.loc[valid_mask].copy()
            valid_df["epoch"] = pd.to_numeric(valid_df["epoch"], errors="coerce").astype(int)

    test_cols = [c for c in df.columns if c.endswith("_test")]
    if test_cols:
        test_mask = pd.Series(False, index=df.index)
        for col in test_cols:
            test_mask = test_mask | df[col].notna()
        if test_mask.any():
            final_df = df.loc[test_mask].copy()

    return valid_df, final_df


def _best_valid(ri: RunInfo) -> Tuple[Optional[float], Optional[int]]:
    """Return (best_acc_cls_valid, epoch_of_best)."""
    valid_df, _ = split_eval_df(ri.eval_df)
    if valid_df is None or "acc_cls_valid" not in valid_df.columns:
        return None, None
    s = pd.to_numeric(valid_df["acc_cls_valid"], errors="coerce").astype(float)
    if s.dropna().empty:
        return None, None
    idx = int(s.idxmax())
    best = float(s.loc[idx])
    ep = None
    if "epoch" in valid_df.columns:
        try:
            ep = int(valid_df.loc[idx, "epoch"])
        except Exception:
            ep = None
    return best, ep


def _final_metric(ri: RunInfo, key: str) -> Optional[float]:
    """Return the last available final test metric for a run."""
    _, final_df = split_eval_df(ri.eval_df)
    if final_df is None or key not in final_df.columns:
        return None
    s = pd.to_numeric(final_df[key], errors="coerce").astype(float).dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _color_cycle(n: int) -> List:
    """Return a list of distinct colors."""
    cmap = plt.get_cmap("tab20")
    if n <= 20:
        return [cmap(i) for i in range(n)]
    # Repeat with offset for >20 runs.
    return [cmap(i % 20) for i in range(n)]


def save_fig(fig: plt.Figure, outpath: str) -> None:
    """Save a matplotlib figure to outpath, creating parent directory."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_compare_runs] Saved: {outpath}")


def fig_compare_losses(runs: Sequence[RunInfo], outpath: str, smooth: int, normalize: bool) -> None:
    """Plot loss curves (cls/rot/rec) overlayed across runs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    tasks = [("loss_cls", "Classification loss"), ("loss_rot", "Rotation loss"), ("loss_rec", "Reconstruction loss")]

    colors = _color_cycle(len(runs))
    for i, ri in enumerate(runs):
        df = ri.train_df
        x_col = "step" if "step" in df.columns else ("iter" if "iter" in df.columns else None)
        if x_col is None:
            continue
        label = run_label(ri)
        for ax, (col, title) in zip(axes, tasks):
            xy = _get_xy(df, x_col, col)
            if xy is None:
                continue
            x, y = xy
            if normalize:
                y0 = float(y[0]) if np.isfinite(y[0]) else 1.0
                y = y / (y0 + 1e-12)
            if smooth and smooth > 1:
                y = moving_avg(y, smooth)
            ax.plot(x, y, color=colors[i], linewidth=1.7, alpha=0.95, label=label)
            ax.set_title(title + (" (normalized)" if normalize else ""))
            ax.set_xlabel(x_col)
            ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("loss" if not normalize else "relative loss")
    # One shared legend to reduce clutter.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle("Loss comparison across runs", y=1.02)
    save_fig(fig, outpath)


def fig_compare_valid(runs: Sequence[RunInfo], outpath: str) -> None:
    """Plot validation curves and final test markers across runs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)
    metrics = [
        ("acc_cls_valid", "acc_cls_test", "Classification accuracy", "accuracy"),
        ("acc_rot_valid", "acc_rot_test", "Rotation accuracy", "accuracy"),
        ("loss_rec_valid", "loss_rec_test", "Reconstruction loss", "loss"),
    ]

    colors = _color_cycle(len(runs))
    for i, ri in enumerate(runs):
        if ri.eval_df is None:
            continue
        valid_df, final_df = split_eval_df(ri.eval_df)
        if valid_df is None and final_df is None:
            continue
        label = run_label(ri)
        for ax, (valid_col, test_col, title, ylabel) in zip(axes, metrics):
            last_epoch = None
            if valid_df is not None:
                xy = _get_xy(valid_df, "epoch", valid_col)
                if xy is not None:
                    x, y = xy
                    last_epoch = int(x[-1])
                    ax.plot(x, y, color=colors[i], linewidth=1.9, alpha=0.95, label=f"{label} | valid")
            if final_df is not None and test_col in final_df.columns:
                test_s = pd.to_numeric(final_df[test_col], errors="coerce").astype(float).dropna()
                if not test_s.empty:
                    test_y = float(test_s.iloc[-1])
                    test_x = last_epoch if last_epoch is not None else 0
                    ax.scatter([test_x], [test_y], color=colors[i], marker="x", s=42, label=f"{label} | test")
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle("Validation comparison across runs", y=1.02)
    save_fig(fig, outpath)


def fig_compare_global_geometry(runs: Sequence[RunInfo], outpath: str, smooth: int) -> None:
    """Plot selected global geometry stats (pre/post) across runs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    specs = [
        ("stats.pre.global.eff_sum", "pre global eff_sum"),
        ("stats.post.global.eff_sum", "post global eff_sum"),
        ("stats.pre.global.viol_frac", "pre global viol_frac"),
        ("stats.post.global.viol_frac", "post global viol_frac"),
        ("stats.pre.global.sum_vec_norm", "pre global sum_vec_norm"),
        ("stats.post.global.sum_vec_norm", "post global sum_vec_norm"),
        ("stats.pre.global.norm_cv", "pre global norm_cv"),
        ("stats.post.global.norm_cv", "post global norm_cv"),
    ]

    # 4 panels: eff_sum, viol_frac, sum_vec_norm, norm_cv
    panel_defs = [
        (0, [specs[0], specs[1]], "eff_sum"),
        (1, [specs[2], specs[3]], "viol_frac"),
        (2, [specs[4], specs[5]], "sum_vec_norm"),
        (3, [specs[6], specs[7]], "norm_cv"),
    ]

    colors = _color_cycle(len(runs))
    for i, ri in enumerate(runs):
        df = ri.train_df
        x_col = "step" if "step" in df.columns else ("iter" if "iter" in df.columns else None)
        if x_col is None:
            continue
        label = run_label(ri)
        for ax_idx, spec_list, ylabel in panel_defs:
            ax = axes[ax_idx]
            any_ok = False
            for y_col, y_label in spec_list:
                xy = _get_xy(df, x_col, y_col)
                if xy is None:
                    continue
                x, y = xy
                if smooth and smooth > 1:
                    y = moving_avg(y, smooth)
                ax.plot(x, y, color=colors[i], linewidth=1.5, alpha=0.90, label=f"{label} | {y_label}")
                any_ok = True
            ax.set_title(f"Global {ylabel}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            if not any_ok:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    fig.suptitle("Global geometry comparison across runs", y=1.01)
    save_fig(fig, outpath)


def fig_compare_gpop_policy(runs: Sequence[RunInfo], outpath: str, smooth: int, policy: str) -> bool:
    """
    Plot gpop-policy columns across runs.

    Returns True if at least one run had the expected columns.
    """
    base = f"stats.common_gpop_surgery.{policy}"
    cols = [
        (f"{base}.dot.mean", "dot.mean"),
        (f"{base}.dot.min", "dot.min"),
        (f"{base}.dot.neg_frac", "dot.neg_frac"),
        (f"{base}.v_ref_norm", "v_ref_norm"),
        (f"{base}.g_pop_norm", "g_pop_norm"),
        (f"{base}.damping", "damping"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    panels = [
        (0, cols[:3], "Projection stats"),
        (1, cols[3:5], "Norms"),
        (2, [cols[5]], "Damping"),
        (3, [(f"{base}.dot.neg_frac", "dot.neg_frac"), ("stats.post.global.viol_frac", "post global viol_frac")], "Policy vs post-global"),
    ]

    colors = _color_cycle(len(runs))
    any_found = False
    for i, ri in enumerate(runs):
        df = ri.train_df
        x_col = "step" if "step" in df.columns else ("iter" if "iter" in df.columns else None)
        if x_col is None:
            continue
        label = run_label(ri)
        for ax_idx, spec_list, title in panels:
            ax = axes[ax_idx]
            for y_col, y_label in spec_list:
                xy = _get_xy(df, x_col, y_col)
                if xy is None:
                    continue
                any_found = True
                x, y = xy
                if smooth and smooth > 1 and ("damping" not in y_col):
                    y = moving_avg(y, smooth)
                ax.plot(x, y, color=colors[i], linewidth=1.6, alpha=0.92, label=f"{label} | {y_label}")
            ax.set_title(title)
            ax.set_xlabel(x_col)
            ax.grid(True, alpha=0.25)

    if not any_found:
        plt.close(fig)
        return False

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    fig.suptitle(f"Gpop policy comparison across runs ({policy})", y=1.01)
    save_fig(fig, outpath)
    return True


def write_summary_csv(runs: Sequence[RunInfo], outpath: str) -> None:
    """Write a run summary table as CSV."""
    rows: List[Dict] = []
    for ri in runs:
        best_acc, best_ep = _best_valid(ri)
        rows.append(
            {
                "run": ri.name,
                "run_dir": ri.run_dir,
                "grad_mode": ri.cfg.get("grad_mode", ""),
                "gpop": bool(_safe_get(ri.cfg, ["cli", "gpop"], False)),
                "gpop_policy": _safe_get(ri.cfg, ["cli", "gpop_policy"], ""),
                "epochs": ri.cfg.get("epochs", ""),
                "batch_size": ri.cfg.get("batch_size", ""),
                "lr": ri.cfg.get("lr", ""),
                "seed": ri.cfg.get("seed", ""),
                "n_train_rows": len(ri.train_df),
                "best_acc_cls_valid": best_acc if best_acc is not None else "",
                "best_epoch": best_ep if best_ep is not None else "",
                "acc_cls_test": _final_metric(ri, "acc_cls_test") or "",
                "acc_rot_test": _final_metric(ri, "acc_rot_test") or "",
                "loss_rec_test": _final_metric(ri, "loss_rec_test") or "",
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False)
    print(f"[plot_compare_runs] Saved: {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple mtl_cifar runs under a folder")
    parser.add_argument(
        "--runs_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_cifar"),
        help="Folder containing multiple run subfolders (each must contain train.jsonl).",
    )
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: <runs_dir>/plots_compare)")
    parser.add_argument("--include", type=str, default=None, help="Regex include filter for run folder name")
    parser.add_argument("--exclude", type=str, default=None, help="Regex exclude filter for run folder name")
    parser.add_argument("--max_runs", type=int, default=50, help="Limit number of runs to plot")
    parser.add_argument("--smooth", type=int, default=101, help="Moving average window on step-series plots")
    parser.add_argument("--policy", type=str, default="cov_inv", help="Gpop policy kind to plot (e.g. cov_inv)")
    args = parser.parse_args()

    run_dirs = discover_runs(args.runs_dir, include=args.include, exclude=args.exclude)
    if not run_dirs:
        raise ValueError(f"No runs found under: {args.runs_dir}")

    run_dirs = run_dirs[: int(args.max_runs)]
    runs = [load_run(d) for d in run_dirs]

    outdir = args.outdir or os.path.join(os.path.abspath(args.runs_dir), "plots_compare")
    os.makedirs(outdir, exist_ok=True)

    # Core figures
    fig_compare_losses(runs, os.path.join(outdir, "compare_losses.png"), smooth=args.smooth, normalize=False)
    fig_compare_losses(runs, os.path.join(outdir, "compare_losses_normalized.png"), smooth=args.smooth, normalize=True)
    fig_compare_valid(runs, os.path.join(outdir, "compare_valid_metrics.png"))
    fig_compare_global_geometry(runs, os.path.join(outdir, "compare_global_geometry.png"), smooth=args.smooth)

    # Optional: gpop policy plot if any run has those columns
    _ = fig_compare_gpop_policy(
        runs,
        os.path.join(outdir, "compare_gpop_policy.png"),
        smooth=args.smooth,
        policy=str(args.policy),
    )

    write_summary_csv(runs, os.path.join(outdir, "runs_summary.csv"))
    print("[plot_compare_runs] Done.")


if __name__ == "__main__":
    main()

