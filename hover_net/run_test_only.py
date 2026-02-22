"""Standalone script to run test evaluation using saved checkpoints.

Results are written under the same logs tree, e.g.:
  log_dir=logs/grad_sum/01  ->  logs/grad_sum/01_test/stats.json

Note: Random seed (from config, default 10) is set in run_test() for reproducibility.
It does NOT affect test metrics (np_dice, hv_mse, etc.): test uses shuffle=False,
deterministic center-crop only, and metrics are computed on the full test set.
Seed only affects which 8 samples are picked for TensorBoard visualization.

Usage:
  run_test_only.py <log_dir> [--gpu=<id>]
  run_test_only.py (-h | --help)

Options:
  -h --help     Show this string.
  --gpu=<id>    Comma-separated GPU list. [default: 0]

Examples:
  python run_test_only.py logs/grad_sum/01
  python run_test_only.py logs/grad_sum/01 --gpu 0
  python run_test_only.py /home/zz/zheng/gradient/hover_net/logs/grad_sum/01
"""

import os
import sys


def _ensure_conda_lib_in_ld_path():
    """
    Re-exec this script with CONDA_PREFIX/lib prepended to LD_LIBRARY_PATH
    so that cv2/torch etc. load conda's libstdc++ (avoids CXXABI_1.3.15 errors
    when run directly in a shell that does not set LD_LIBRARY_PATH).
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

# Project root on path (same as run_train.py) before importing run_train
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from docopt import docopt

from run_train import TrainManager


def main():
    args = docopt(__doc__)
    log_dir = args["<log_dir>"].rstrip("/")
    if not log_dir:
        print("Error: <log_dir> is required.")
        sys.exit(1)

    # Resolve to absolute path so it works regardless of cwd
    if not os.path.isabs(log_dir):
        log_dir = os.path.abspath(log_dir)

    if not os.path.isdir(log_dir):
        print("Error: log_dir is not a directory: %s" % log_dir)
        sys.exit(1)

    stats_file = os.path.join(log_dir, "stats.json")
    if not os.path.isfile(stats_file):
        print("Error: stats.json not found in %s (is this a training log dir?)" % log_dir)
        sys.exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args["--gpu"] or "0"
    trainer = TrainManager(grad_overrides={})
    trainer.nr_gpus = torch.cuda.device_count()
    print("Using %d GPU(s). Log dir: %s" % (trainer.nr_gpus, log_dir))

    trainer.run_test(log_dir)
    print("Done. Test results: %s_test" % log_dir)


if __name__ == "__main__":
    main()
