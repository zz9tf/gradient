# grad_methods_B.py
# B派：权重/标量化（不“手术”梯度，主要动态调 w_t，再做 sum）
from __future__ import annotations

import torch
from typing import Dict, Tuple, Optional


@torch.no_grad()
def _norm_to_simplex(w: torch.Tensor, eps: float) -> torch.Tensor:
    w = w.clamp_min(0.0)
    return w / w.sum().clamp_min(eps)


@torch.no_grad()
def _entropy(w: torch.Tensor, eps: float) -> torch.Tensor:
    w = w.clamp_min(eps)
    return -(w * w.log()).sum()


# ============================================================
# DWA
# ============================================================

@torch.no_grad()
def dwa_weights(
    losses_vec: torch.Tensor,                # [T]
    prev_losses_vec: Optional[torch.Tensor], # [T] (your wrapper passes prev_losses_2)
    Ttemp: float = 2.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device, dtype = losses_vec.device, losses_vec.dtype
    T = int(losses_vec.numel())
    eps_t = float(eps)

    if prev_losses_vec is None:
        w = torch.full((T,), 1.0 / max(T, 1), device=device, dtype=dtype)
    else:
        prev = prev_losses_vec.to(device=device, dtype=dtype)
        r = losses_vec / (prev + eps_t)
        w = torch.softmax(r / float(Ttemp), dim=0)  # already on simplex

    st = {"w_entropy": _entropy(w, eps_t)}
    return w, st


# ============================================================
# GradNorm (heuristic)
# ============================================================

@torch.no_grad()
def gradnorm_weights(
    G: torch.Tensor,                         # [T,P]
    losses_vec: torch.Tensor,                # [T]
    init_losses_vec: Optional[torch.Tensor], # [T]
    alpha: float = 1.5,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device, dtype = losses_vec.device, losses_vec.dtype
    eps_t = float(eps)

    gnorm = (G.norm(dim=1) + eps_t).to(device=device, dtype=dtype)  # [T]

    if init_losses_vec is None:
        init_losses_vec = losses_vec.detach().clone().clamp_min(eps_t)

    L0 = init_losses_vec.to(device=device, dtype=dtype).clamp_min(eps_t)
    r = (losses_vec / L0).clamp_min(eps_t)
    r_bar = r.mean().clamp_min(eps_t)

    target = (r / r_bar).pow(float(alpha))
    target_g = (gnorm.mean() * target).detach()

    w = (target_g / gnorm).clamp_min(eps_t)
    w = _norm_to_simplex(w, eps_t)

    st = {"w_entropy": _entropy(w, eps_t)}
    return w, st


# ============================================================
# UW heuristic (EMA-loss)
# ============================================================

@torch.no_grad()
def uw_heuristic_weights(
    losses_vec: torch.Tensor,                      # [T]
    beta: float = 0.9,
    state: Optional[Dict[str, torch.Tensor]] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    device, dtype = losses_vec.device, losses_vec.dtype
    eps_t = float(eps)

    if state is None:
        state = {}

    ema = state.get("ema_loss", None)
    if ema is None:
        ema = losses_vec.detach().clone()
    else:
        ema = ema.to(device=device, dtype=dtype)

    b = float(beta)
    ema = b * ema + (1.0 - b) * losses_vec.detach()
    state["ema_loss"] = ema.detach()

    w = (1.0 / (ema + eps_t)).clamp_min(eps_t)
    w = _norm_to_simplex(w, eps_t)

    st = {"w_entropy": _entropy(w, eps_t)}
    return w, st, state


# ============================================================
# Apply
# ============================================================

@torch.no_grad()
def apply_weighting_then_sum(G: torch.Tensor, w_task: torch.Tensor) -> torch.Tensor:
    return (w_task[:, None] * G).sum(dim=0)