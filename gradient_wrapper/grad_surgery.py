# grad_methods_A.py
# A派：梯度手术 / 投影 / 冲突处理（直接改梯度方向）
from __future__ import annotations

import torch
from typing import Dict, Tuple


# -------------------------
# simplex projection
# -------------------------

@torch.no_grad()
def _proj_simplex(v: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    # Euclidean projection onto simplex {x>=0, sum x = z}
    if v.numel() == 1:
        return v.new_tensor([z]).clamp_min(0.0)

    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - float(z)
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0

    if not bool(cond.any().item()):
        # fallback: uniform
        return v.new_full((v.numel(),), 1.0 / v.numel())

    rho = torch.nonzero(cond, as_tuple=False).max()
    theta = cssv[rho] / (rho + 1.0)
    return torch.clamp(v - theta, min=0.0)


# ============================================================
# 1) SUM
# ============================================================

@torch.no_grad()
def sum_grad(G: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    g = (w * G).sum(dim=0)
    return g, {}


# ============================================================
# 2) PCGrad
# ============================================================

@torch.no_grad()
def pcgrad(
    G: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    PCGrad: pairwise projection when dot(gi, gj) < 0.
    Returns:
      - g_final
      - stats: {"conflict_frac": ...}  # method-unique, minimal
    """
    T = int(G.shape[0])
    Gm = G.clone()
    Gref = G.detach()

    conflict = 0
    total = 0

    for i in range(T):
        gi = Gm[i]
        perm = torch.randperm(T, device=G.device)
        for j in perm:
            j = int(j)
            if i == j:
                continue
            total += 1
            gj = Gref[j]
            dot = torch.dot(gi, gj)
            if dot < 0:
                conflict += 1
                gi = gi - (dot / (torch.dot(gj, gj) + eps)) * gj
        Gm[i] = gi

    g = (w * Gm).sum(dim=0)
    st = {"conflict_frac": g.new_tensor(float(conflict) / max(1, total))}
    return g, st


# ============================================================
# 3) GradDrop
# ============================================================

@torch.no_grad()
def graddrop(
    G: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    GradDrop: per-dimension sign dropout.
    Minimal stats: conflict_frac + P_mean (optional but useful).
    """
    Gw = w * G
    S = Gw.sum(dim=0)
    A = Gw.abs().sum(dim=0)
    P = 0.5 * (1.0 + S / (A + eps))

    choose_pos = (P > torch.rand_like(P))

    signs = torch.sign(Gw)
    mask = torch.ones_like(Gw)

    drop_pos = (~choose_pos)[None, :] & (signs > 0)
    drop_neg = ( choose_pos)[None, :] & (signs < 0)
    mask[drop_pos | drop_neg] = 0.0

    g_final = (Gw * mask).sum(dim=0)

    conflict_dim = ((signs > 0).any(dim=0) & (signs < 0).any(dim=0))
    st = {
        "conflict_frac": conflict_dim.float().mean(),
        "P_mean": P.mean(),
    }
    return g_final, st


# ============================================================
# 4) MGDA / MinNorm
# ============================================================

@torch.no_grad()
def mgda_min_norm(
    G: torch.Tensor,
    iters: int = 80,
    lr: float = 0.2,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    MinNorm: min_{alpha in simplex} || sum alpha_t g_t ||^2
    Minimal stats: alpha_entropy, alpha_max.
    """
    T = int(G.shape[0])
    alpha = torch.full((T,), 1.0 / max(T, 1), device=G.device, dtype=G.dtype)

    for _ in range(int(iters)):
        g = (alpha[:, None] * G).sum(dim=0)
        grad = 2.0 * (G @ g)  # [T]
        alpha = _proj_simplex(alpha - lr * grad, z=1.0)

    g_final = (alpha[:, None] * G).sum(dim=0)
    a = alpha.clamp_min(float(eps))
    st = {
        "alpha_max": alpha.max(),
        "alpha_entropy": -(a * a.log()).sum(),
    }
    return g_final, st


# ============================================================
# 5) CAGrad
# ============================================================

@torch.no_grad()
def cagrad(
    G: torch.Tensor,
    w: torch.Tensor,
    alpha: float = 0.5,
    iters: int = 60,
    lr: float = 0.2,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    CAGrad: PGD on simplex (engineering approximation).
    Minimal stats: a_entropy, a_max.
    """
    g0 = (w * G).sum(dim=0)
    g0n = g0.norm() + eps

    T = int(G.shape[0])
    a = torch.full((T,), 1.0 / max(T, 1), device=G.device, dtype=G.dtype)

    for _ in range(int(iters)):
        g = (a[:, None] * G).sum(dim=0)
        g_norm = g.norm() + eps

        cos = (G @ g) / ((G.norm(dim=1) + eps) * g_norm)  # [T]
        tau = 10.0
        pi = torch.softmax(-tau * cos, dim=0)

        Gi_unit = G / (G.norm(dim=1, keepdim=True) + eps)
        g_unit = g / g_norm
        dcos_dg = (Gi_unit - cos[:, None] * g_unit[None, :]) / g_norm
        dsmin_dg = -(pi[:, None] * dcos_dg).sum(dim=0)

        d_obj_dg = 2.0 * (g - g0) - 2.0 * float(alpha) * g0n * dsmin_dg
        grad_a = (G @ d_obj_dg)
        a = _proj_simplex(a - lr * grad_a, z=1.0)

    g = (a[:, None] * G).sum(dim=0)
    g_final = (1.0 - float(alpha)) * g0 + float(alpha) * g

    aa = a.clamp_min(float(eps))
    st = {
        "a_max": a.max(),
        "a_entropy": -(aa * aa.log()).sum(),
    }
    return g_final, st