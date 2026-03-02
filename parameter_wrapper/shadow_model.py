"""
parameter_wrapper/shadow_model.py

Shadow model (EMA) in PARAMETER space.

What you get:
1) A clean EMA "shadow" copy of parameters.
2) Utilities to snapshot live params, measure step size, and decide when to stabilize.
3) Two stabilization actions (parameter-space, no gradients needed):
   - pull_to_shadow: theta <- (1-lam)*theta + lam*shadow
   - correct_toward_shadow: theta <- theta + lr*lam * normalize(shadow - theta)

Typical loop (recommended):
    shadow = ShadowEMA(model, beta=0.999)

    for step, batch in enumerate(loader):
        snap_prev = shadow.snapshot_live(model)      # snapshot BEFORE backward/step

        loss = ...
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        shadow.update(model)                         # update EMA AFTER step

        # Optional: stabilize only when step becomes tiny (late-stage)
        info = shadow.step_ratio(snap_prev, model)   # uses prev snapshot + current live
        if info["rel_step"] < 1e-4:                  # you pick threshold
            shadow.pull_to_shadow(model, lam=0.05)

Notes:
- This file is self-contained and safe to drop into your repo.
- EMA is computed over floating-point parameters (and optionally buffers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple, Callable

import math
import torch
from torch import nn


# -------------------------
# helpers
# -------------------------

def _iter_tensors(module: nn.Module, include_buffers: bool) -> Iterator[Tuple[str, torch.Tensor]]:
    for n, p in module.named_parameters(recurse=True):
        if p is None:
            continue
        yield n, p
    if include_buffers:
        for n, b in module.named_buffers(recurse=True):
            if b is None:
                continue
            yield f"__buffer__.{n}", b


def _is_float_tensor(_: str, t: torch.Tensor) -> bool:
    return torch.is_floating_point(t)


def _l2_sq(t: torch.Tensor) -> torch.Tensor:
    return torch.sum(t * t)


# -------------------------
# config
# -------------------------

@dataclass
class ShadowConfig:
    beta: float = 0.999
    eps: float = 1e-8
    include_buffers: bool = False


# -------------------------
# main
# -------------------------

class ShadowEMA:
    """
    Maintains an EMA ("shadow") of parameters.

    Key methods:
    - update(model): shadow <- beta*shadow + (1-beta)*theta
    - snapshot_live(model): returns a detached copy of current live params
    - step_ratio(prev_snap, model): measures ||theta - prev|| / ||theta||
    - pull_to_shadow(model, lam): convex blend toward shadow
    - correct_toward_shadow(model, lr, lam): small normalized correction toward shadow
    """

    def __init__(
        self,
        model: nn.Module,
        beta: float = 0.999,
        *,
        eps: float = 1e-8,
        include_buffers: bool = False,
        param_filter: Callable[[str, torch.Tensor], bool] = _is_float_tensor,
        shadow_device: Optional[torch.device] = None,
    ):
        if not (0.0 < float(beta) < 1.0):
            raise ValueError(f"beta must be in (0,1), got {beta}")

        self.cfg = ShadowConfig(beta=float(beta), eps=float(eps), include_buffers=bool(include_buffers))
        self.param_filter = param_filter
        self.shadow_device = shadow_device

        self.shadow: Dict[str, torch.Tensor] = {}
        self._init_from(model)

    def _init_from(self, model: nn.Module) -> None:
        self.shadow.clear()
        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            s = t.detach().clone()
            if self.shadow_device is not None:
                s = s.to(self.shadow_device)
            self.shadow[name] = s

    @torch.no_grad()
    def update(self, model: nn.Module, *, beta: Optional[float] = None) -> None:
        b = float(self.cfg.beta if beta is None else beta)
        if not (0.0 < b < 1.0):
            raise ValueError(f"beta must be in (0,1), got {b}")

        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue

            if name not in self.shadow:
                s = t.detach().clone()
                if self.shadow_device is not None:
                    s = s.to(self.shadow_device)
                self.shadow[name] = s
                continue

            live = t.detach()
            if self.shadow_device is not None:
                live = live.to(self.shadow_device)

            self.shadow[name].mul_(b).add_(live, alpha=(1.0 - b))

    # -------------------------
    # snapshot + metrics
    # -------------------------

    @torch.no_grad()
    def snapshot_live(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Detached copy of current live parameters (and optional buffers) on their native devices.
        Use this BEFORE optimizer.step() if you want accurate step norms.
        """
        snap: Dict[str, torch.Tensor] = {}
        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            snap[name] = t.detach().clone()
        return snap

    @torch.no_grad()
    def step_ratio(self, prev_snap: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, float]:
        """
        Compute:
          step_l2 = ||theta_now - theta_prev||
          theta_l2 = ||theta_now||
          rel_step = step_l2 / (theta_l2 + eps)
        """
        step_sq = 0.0
        theta_sq = 0.0

        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            if name not in prev_snap:
                continue
            prev = prev_snap[name].to(t.device)
            diff = (t.detach() - prev).float()
            step_sq += float(_l2_sq(diff).item())
            theta_sq += float(_l2_sq(t.detach().float()).item())

        step_l2 = math.sqrt(step_sq + self.cfg.eps)
        theta_l2 = math.sqrt(theta_sq + self.cfg.eps)
        rel_step = step_l2 / (theta_l2 + self.cfg.eps)

        return {"step_l2": step_l2, "theta_l2": theta_l2, "rel_step": rel_step}

    @torch.no_grad()
    def shadow_gap(self, model: nn.Module) -> Dict[str, float]:
        """
        gap_l2 = ||theta - shadow||
        rel_gap = gap_l2 / (||theta|| + eps)
        """
        gap_sq = 0.0
        theta_sq = 0.0

        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            if name not in self.shadow:
                continue

            s = self.shadow[name]
            if s.device != t.device:
                s = s.to(t.device)
            diff = (t.detach() - s).float()
            gap_sq += float(_l2_sq(diff).item())
            theta_sq += float(_l2_sq(t.detach().float()).item())

        gap_l2 = math.sqrt(gap_sq + self.cfg.eps)
        theta_l2 = math.sqrt(theta_sq + self.cfg.eps)
        rel_gap = gap_l2 / (theta_l2 + self.cfg.eps)

        return {"gap_l2": gap_l2, "theta_l2": theta_l2, "rel_gap": rel_gap}

    # -------------------------
    # actions (parameter space)
    # -------------------------

    @torch.no_grad()
    def pull_to_shadow(self, model: nn.Module, *, lam: float = 0.05) -> None:
        """
        Convex blend:
            theta <- (1-lam)*theta + lam*shadow
        """
        lam = float(lam)
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"lam must be in [0,1], got {lam}")

        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            if name not in self.shadow:
                continue
            s = self.shadow[name]
            if s.device != t.device:
                s = s.to(t.device)
            t.mul_(1.0 - lam).add_(s, alpha=lam)

    @torch.no_grad()
    def correct_toward_shadow(self, model: nn.Module, *, lr: float, lam: float = 1.0) -> Dict[str, float]:
        """
        Small normalized correction step toward shadow:
            d = shadow - theta
            theta <- theta + lr*lam * d / (||d|| + eps)

        Useful when you believe EMA direction is the stable attractor late-stage.
        Returns {"d_l2":..., "applied": 0/1}
        """
        lr = float(lr)
        lam = float(lam)
        if lr <= 0 or lam <= 0:
            return {"d_l2": 0.0, "applied": 0.0}

        # compute global d norm
        d_sq = 0.0
        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            if name not in self.shadow:
                continue
            s = self.shadow[name]
            if s.device != t.device:
                s = s.to(t.device)
            d = (s - t.detach()).float()
            d_sq += float(_l2_sq(d).item())

        d_l2 = math.sqrt(d_sq + self.cfg.eps)
        if d_l2 <= 0:
            return {"d_l2": 0.0, "applied": 0.0}

        scale = (lr * lam) / (d_l2 + self.cfg.eps)

        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            if name not in self.shadow:
                continue
            s = self.shadow[name]
            if s.device != t.device:
                s = s.to(t.device)
            d = (s - t).float()
            t.add_(d, alpha=scale)

        return {"d_l2": d_l2, "applied": 1.0}

    # -------------------------
    # evaluation convenience
    # -------------------------

    @torch.no_grad()
    def copy_shadow_to(self, model: nn.Module) -> None:
        """Overwrite live params with shadow (useful for eval-time EMA weights)."""
        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            if name not in self.shadow:
                continue
            s = self.shadow[name]
            if s.device != t.device:
                s = s.to(t.device)
            t.copy_(s)
