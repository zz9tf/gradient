"""
Small, generic stats.json writer for plot-friendly experiment logging.

This module writes a dict-of-dicts JSON file:
  {
    "1": {"train-grad_cos_tt_mean": 0.1, "train-overall_loss": 3.2, ...},
    "2": {...},
    ...
  }

The intent is to keep a simple key:value format that is easy to extend with new
components (e.g., extra diagnostics) while remaining compatible with the
existing plotting script under `CellViT/logs/plot_all.py`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping


def _is_number(x: object) -> bool:
    """Return True if `x` is an int/float (but not bool)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _require_finite_float(value: object, *, key: str) -> float:
    """Convert value to finite float or raise with a clear error."""
    if not _is_number(value):
        raise TypeError(f"Stats value for '{key}' must be numeric, got: {type(value)}")
    y = float(value)
    if not math.isfinite(y):
        raise ValueError(f"Stats value for '{key}' must be finite, got: {y}")
    return y


def canonicalize_scalar_metrics(metrics: Mapping[str, object]) -> Dict[str, float]:
    """Convert trainer scalar metrics to a stable, plot-friendly flat dict.

    This keeps the format simple (key:value), while providing:
    - "train-grad_<metric>" aliases for GradAgg stats (required by plot_all.py).
    - A small set of "train-*" / "valid-*" aliases for common metrics.

    Args:
        metrics: Scalar metrics dict produced by trainers (typically W&B logging dicts).

    Returns:
        Dict of canonical key -> finite float.
    """
    out: Dict[str, float] = {}

    # Keep a raw numeric namespace for forward-compatibility (easy key:value extension).
    for k, v in metrics.items():
        if not isinstance(k, str):
            continue
        if not _is_number(v):
            continue
        raw_key = "raw-" + k.replace("/", "_").replace(" ", "_")
        out[raw_key] = _require_finite_float(v, key=k)

    # Special-cases for existing plot panels.
    if "Loss/Train" in metrics:
        out["train-overall_loss"] = _require_finite_float(metrics["Loss/Train"], key="Loss/Train")
    if "Loss/Validation" in metrics:
        out["valid-overall_loss"] = _require_finite_float(metrics["Loss/Validation"], key="Loss/Validation")
    if "hv_map_mse/Validation" in metrics:
        out["valid-hv_mse"] = _require_finite_float(metrics["hv_map_mse/Validation"], key="hv_map_mse/Validation")
    if "Learning-Rate/Learning-Rate" in metrics:
        out["lr"] = _require_finite_float(metrics["Learning-Rate/Learning-Rate"], key="Learning-Rate/Learning-Rate")

    for k, v in metrics.items():
        if not isinstance(k, str):
            continue
        if k.startswith("GradAgg/") and k.endswith("/Train"):
            name = k[len("GradAgg/") : -len("/Train")]
            out[f"train-grad_{name}"] = _require_finite_float(v, key=k)

    return out


@dataclass
class StatsJsonWriter:
    """Append-or-merge writer for stats.json (dict-of-dicts).

    The writer loads an existing stats.json if present, merges new metrics into
    the row for a given step, and writes back the full JSON.
    """

    path: Path

    def _load(self) -> MutableMapping[str, MutableMapping[str, float]]:
        """Load existing stats.json or return an empty dict."""
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"stats.json root must be a dict, got: {type(obj)}")
        # Ensure nested dict shape.
        for step, row in obj.items():
            if not isinstance(step, str) or not isinstance(row, dict):
                raise ValueError("stats.json must be a dict-of-dicts keyed by string step.")
        return obj  # type: ignore[return-value]

    def update_step(self, step: int, metrics: Mapping[str, object]) -> None:
        """Merge metrics into the row for `step` and write to disk.

        Args:
            step: Step/epoch index (1-based to match trainer logging).
            metrics: Scalar metrics mapping (will be canonicalized and validated).
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got: {step}")

        data = self._load()
        row = data.setdefault(str(step), {})

        canon = canonicalize_scalar_metrics(metrics)
        row.update(canon)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

