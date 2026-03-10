# plot_eval.py
# Plot classification accuracy curves from runs_cifar/*/eval.jsonl.
# Supports both per-epoch validation records and optional final test record.

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib


def load_eval_jsonl(path: str) -> List[dict]:
    """
    Load a JSONL file; each line is one eval record.

    Args:
        path: Path to eval.jsonl.

    Returns:
        List of dicts with keys epoch, acc_cls, etc.
    """
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def split_eval_records(records: List[dict]) -> tuple[List[dict], List[dict]]:
    """
    Split eval records into per-epoch validation rows and final test rows.

    Args:
        records: Parsed rows from eval.jsonl.

    Returns:
        A tuple of:
        - validation rows with numeric epochs and `acc_cls_valid`
        - final rows containing `acc_cls_test`
    """
    valid_rows: List[dict] = []
    final_rows: List[dict] = []
    for r in records:
        epoch = r.get("epoch")
        if isinstance(epoch, (int, float)) and "acc_cls_valid" in r:
            valid_rows.append(r)
        elif "acc_cls_test" in r:
            final_rows.append(r)
    return valid_rows, final_rows


def main():
    """
    Load all eval.jsonl files under runs_dir and plot validation/test accuracy.
    """
    parser = argparse.ArgumentParser(description="Plot validation/test classification curves from eval.jsonl")
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs_cifar",
        help="Directory containing run subdirs, each with eval.jsonl",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="eval_cls_curves.png",
        help="Output figure path",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    eval_files = sorted(runs_dir.glob("*/eval.jsonl"))
    if not eval_files:
        raise FileNotFoundError(f"No eval.jsonl found under {runs_dir}")

    matplotlib.use("Agg")
    fig, ax = plt.subplots()

    for ef in eval_files:
        run_name = ef.parent.name
        records = load_eval_jsonl(str(ef))
        valid_rows, final_rows = split_eval_records(records)

        if valid_rows:
            epochs = [int(r["epoch"]) for r in valid_rows]
            accs = [float(r["acc_cls_valid"]) for r in valid_rows]
            ax.plot(epochs, accs, label=f"{run_name} (valid)", alpha=0.9)

        if final_rows:
            final_acc = float(final_rows[-1]["acc_cls_test"])
            if valid_rows:
                final_x = int(valid_rows[-1]["epoch"])
            else:
                final_x = 0
            ax.scatter([final_x], [final_acc], marker="x", s=36, label=f"{run_name} (test)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Validation/Test classification curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    plt.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
