"""Run and summarize HoVer-Net test results across multiple runs.

Usage:
  run_test_summary.py <root> [--gpu=<id>] [--max-epoch=<n>] [--force=<0|1>] [--skip-errors=<0|1>]
    [--metric=<key>] [--csv=<path>] [--sort=<order>]
  run_test_summary.py (-h | --help)

Options:
  -h --help        Show this help message.
  --gpu=<id>       Comma-separated GPU list for CUDA_VISIBLE_DEVICES. [default: 0]
  --max-epoch=<n>  If set, select best checkpoint only among epochs <= n.
  --force=<0|1>    If 1, re-run test even if <log_dir>_test/stats.json exists. [default: 0]
  --skip-errors=<0|1>  If 1, skip runs that error instead of failing fast. [default: 0]
  --metric=<key>   Metric key to sort by (e.g. test-np_dice). If not provided,
                   the script will try common defaults (test-np_dice, test_np_dice,
                   valid-np_dice, valid_np_dice) or fall back to the first
                   numeric metric it finds.
  --csv=<path>     Optional path to write a CSV summary.
  --sort=<order>   Sort order: asc or desc. [default: desc]

Description:
  This script searches recursively under <root> for training run directories
  that contain a "stats.json" file and at least one "*_epoch=*.tar" checkpoint.

  For each training "<log_dir>", it optionally runs test evaluation using
  TrainManager.run_test(log_dir, max_epoch=...), producing:
    "<log_dir>_test/stats.json"

  By default it reuses existing test results if present, unless --force=1.
  After ensuring test stats exist, it loads the last epoch's numeric metrics
  from each "<log_dir>_test/stats.json", prints a summary table (one row per
  run, relative to <root>), and optionally writes the same information as CSV.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

from docopt import docopt


def _ensure_conda_lib_in_ld_path() -> None:
    """Re-exec this script with CONDA_PREFIX/lib prepended to LD_LIBRARY_PATH.

    This helps ensure that libraries such as cv2/torch load the correct
    libstdc++ from the active conda environment when running the script
    directly in a shell that does not set LD_LIBRARY_PATH.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    conda_lib = os.path.join(conda_prefix, "lib")
    if not os.path.isdir(conda_lib):
        return
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if conda_lib in ld.split(":"):
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = conda_lib + (":" + ld if ld else "")
    os.execve(sys.executable, [sys.executable, __file__] + sys.argv[1:], env)


_ensure_conda_lib_in_ld_path()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from run_train import TrainManager  # noqa: E402

def _has_any_checkpoint(log_dir: str) -> bool:
    """Check whether a directory contains at least one epoch checkpoint.

    Args:
        log_dir: Training log directory.

    Returns:
        True if any file in log_dir matches "*_epoch=*.tar".
    """
    try:
        for name in os.listdir(log_dir):
            if name.endswith(".tar") and "_epoch=" in name:
                return True
    except FileNotFoundError:
        return False
    return False


def _find_train_log_dirs(root: str) -> List[str]:
    """Find all training log directories under the given root.

    Args:
        root: Root directory to search from.

    Returns:
        Sorted list of training log directories containing:
            - stats.json
            - at least one "*_epoch=*.tar" checkpoint file
        Directories ending with "_test" are ignored.
    """
    root = os.path.abspath(root)
    results: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if dirpath.endswith("_test"):
            continue
        if "stats.json" not in filenames:
            continue
        if not _has_any_checkpoint(dirpath):
            continue
        results.append(dirpath)
    results.sort()
    return results


def _load_last_epoch_metrics(stats_path: str) -> Tuple[int, Dict[str, float]]:
    """Load the last epoch's numeric metrics from a stats.json file.

    Args:
        stats_path: Path to a stats.json file as written by LoggingEpochOutput.

    Returns:
        A tuple (epoch, metrics_dict) where:
            - epoch is the selected epoch number.
            - metrics_dict maps metric keys to numeric values.

    Raises:
        ValueError: If the JSON structure is empty or does not match the
            expected {epoch: {metric_key: value}} mapping.
    """
    with open(stats_path) as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError(f"Empty or invalid stats.json: {stats_path}")

    try:
        epochs = sorted((int(k) for k in data.keys()))
    except ValueError as exc:
        raise ValueError(f"Non-integer epoch keys in {stats_path}") from exc

    last_epoch = epochs[-1]
    epoch_key = str(last_epoch)
    epoch_data = data.get(epoch_key)
    if not isinstance(epoch_data, dict):
        raise ValueError(
            f"Epoch entry {epoch_key} in {stats_path} is not a dict: {type(epoch_data)}"
        )

    metrics: Dict[str, float] = {}
    for key, value in epoch_data.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)

    if not metrics:
        raise ValueError(f"No numeric metrics found for epoch {epoch_key} in {stats_path}")

    return last_epoch, metrics


def _pick_sort_metric(
    rows: List[Dict[str, object]],
    user_metric: Optional[str],
) -> Optional[str]:
    """Pick a metric key to sort on.

    Args:
        rows: List of per-run metric dicts (each must include "run" and "epoch").
        user_metric: Optional metric key requested by the user.

    Returns:
        The metric key to sort on, or None if no numeric metrics are available.
    """
    if not rows:
        return None

    # Collect all metric keys except "run" and "epoch".
    metric_keys = set()
    for row in rows:
        for key in row.keys():
            if key in ("run", "epoch"):
                continue
            metric_keys.add(key)

    if not metric_keys:
        return None

    if user_metric:
        if user_metric in metric_keys:
            return user_metric
        print(
            f"Warning: requested metric '{user_metric}' not found; "
            "falling back to automatic choice."
        )

    # Try common defaults first.
    preferred = [
        "test-np_dice",
        "test_np_dice",
        "valid-np_dice",
        "valid_np_dice",
    ]
    for key in preferred:
        if key in metric_keys:
            return key

    # Fall back to a deterministic but arbitrary choice.
    return sorted(metric_keys)[0]


def _print_table(
    rows: List[Dict[str, object]],
    metric_keys: List[str],
) -> None:
    """Print a plain-text table of summary statistics to stdout.

    Args:
        rows: Per-run metric rows including "run" and "epoch".
        metric_keys: Ordered list of metric keys to display as columns.
    """
    if not rows:
        print("No runs to display.")
        return

    header = ["run", "epoch"] + metric_keys
    # Compute column widths.
    col_widths = {name: len(name) for name in header}
    for row in rows:
        col_widths["run"] = max(col_widths["run"], len(str(row["run"])))
        col_widths["epoch"] = max(col_widths["epoch"], len(str(row["epoch"])))
        for key in metric_keys:
            value = row.get(key, "")
            text = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
            col_widths[key] = max(col_widths[key], len(text))

    def _format_row(row_dict: Dict[str, object]) -> str:
        cells: List[str] = []
        run_text = str(row_dict["run"]).ljust(col_widths["run"])
        epoch_text = str(row_dict["epoch"]).rjust(col_widths["epoch"])
        cells.extend([run_text, epoch_text])
        for key in metric_keys:
            value = row_dict.get(key, "")
            if isinstance(value, (int, float)):
                text = f"{value:.4f}"
            else:
                text = str(value)
            cells.append(text.rjust(col_widths[key]))
        return " | ".join(cells)

    header_line = " | ".join(name.ljust(col_widths[name]) for name in header)
    sep_line = "-+-".join("-" * col_widths[name] for name in header)

    print(header_line)
    print(sep_line)
    for row in rows:
        print(_format_row(row))


def _write_csv(
    path: str,
    rows: List[Dict[str, object]],
    metric_keys: List[str],
) -> None:
    """Write summary statistics to a CSV file.

    Args:
        path: Output CSV path.
        rows: Per-run metric rows including "run" and "epoch".
        metric_keys: Ordered list of metric keys to include as columns.
    """
    import csv

    if not rows:
        raise ValueError("No rows available to write CSV.")

    fieldnames = ["run", "epoch"] + metric_keys
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row: Dict[str, object] = {"run": row["run"], "epoch": row["epoch"]}
            for key in metric_keys:
                out_row[key] = row.get(key, "")
            writer.writerow(out_row)


def _ensure_test_stats(
    trainer: TrainManager,
    log_dir: str,
    max_epoch: Optional[int],
    force: bool,
) -> str:
    """Ensure "<log_dir>_test/stats.json" exists, optionally running test.

    Args:
        trainer: TrainManager instance.
        log_dir: Training log directory.
        max_epoch: Optional max epoch constraint for best checkpoint selection.
        force: If True, re-run test even if stats file already exists.

    Returns:
        Path to "<log_dir>_test/stats.json".
    """
    test_log_dir = log_dir.rstrip("/") + "_test"
    stats_path = os.path.join(test_log_dir, "stats.json")
    if not force and os.path.isfile(stats_path):
        return stats_path

    trainer.run_test(log_dir, max_epoch=max_epoch)
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(
            f"Test finished but stats.json not found: {stats_path}. "
            "Check TrainManager.run_test() output."
        )
    return stats_path


def main() -> None:
    """Entry point for the CLI script."""
    args = docopt(__doc__)
    root = args["<root>"]
    metric = args.get("--metric")
    csv_path = args.get("--csv")
    sort_order = (args.get("--sort") or "desc").lower()
    gpu = args.get("--gpu") or "0"
    force = bool(int(args.get("--force") or 0))
    skip_errors = bool(int(args.get("--skip-errors") or 0))
    max_epoch_raw = args.get("--max-epoch")
    max_epoch = int(max_epoch_raw) if max_epoch_raw is not None else None
    if sort_order not in ("asc", "desc"):
        raise ValueError(f"Invalid --sort value: {sort_order} (expected 'asc' or 'desc').")

    train_log_dirs = _find_train_log_dirs(root)
    if not train_log_dirs:
        print(f"No training log dirs found under: {root}")
        sys.exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    trainer = TrainManager(grad_overrides={})
    trainer.nr_gpus = torch.cuda.device_count()
    print(
        "Using %d GPU(s). Root: %s. Runs: %d. max_epoch=%s. force=%s"
        % (trainer.nr_gpus, os.path.abspath(root), len(train_log_dirs), str(max_epoch), str(force))
    )

    rows: List[Dict[str, object]] = []
    for log_dir in train_log_dirs:
        try:
            test_stats_path = _ensure_test_stats(
                trainer=trainer, log_dir=log_dir, max_epoch=max_epoch, force=force
            )
            epoch, metrics = _load_last_epoch_metrics(test_stats_path)
            rel_run = os.path.relpath(log_dir, root)
            row: Dict[str, object] = {"run": rel_run, "epoch": epoch}
            row.update(metrics)
            rows.append(row)
        except Exception as exc:
            if skip_errors:
                print(f"Error processing run {log_dir}: {exc}")
                continue
            raise

    if not rows:
        print("No valid stats.json files could be read.")
        sys.exit(1)

    sort_metric = _pick_sort_metric(rows, metric)
    if sort_metric is None:
        print("No numeric metrics found to summarize.")
        sys.exit(1)

    reverse = sort_order == "desc"
    rows.sort(key=lambda r: r.get(sort_metric, float("-inf")), reverse=reverse)

    metric_keys: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in ("run", "epoch"):
                continue
            if key not in seen:
                seen.add(key)
                metric_keys.append(key)

    print(
        f"Found {len(rows)} test runs under '{root}'. "
        f"Sorting by '{sort_metric}' ({'descending' if reverse else 'ascending'})."
    )
    _print_table(rows, metric_keys)

    if csv_path:
        _write_csv(csv_path, rows, metric_keys)
        print(f"\nCSV summary written to: {csv_path}")


if __name__ == "__main__":
    main()

