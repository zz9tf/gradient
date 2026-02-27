# grad_methods_A.py
# A派：梯度手术/投影/冲突处理（直接改梯度方向）
import torch
from typing import Dict, Tuple, Optional


@torch.no_grad()
def _proj_simplex(v: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    """
    Euclidean projection onto simplex {x>=0, sum x = z}.
    v: [T]
    """
    if v.numel() == 1:
        return v.new_tensor([z]).clamp_min(0.0)
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - z
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    rho = torch.nonzero(cond, as_tuple=False).max()
    theta = cssv[rho] / (rho + 1.0)
    w = torch.clamp(v - theta, min=0.0)
    return w


@torch.no_grad()
def sum_grad(G: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    g = (w * G).sum(dim=0)
    return g, {}


@torch.no_grad()
def pcgrad(G: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    PCGrad: project conflicting components pairwise.
    G: [T,P], w: [T,1]
    """
    T = G.shape[0]
    Gm = G.clone()
    Gref = G.detach()

    for i in range(T):
        gi = Gm[i]
        perm = torch.randperm(T, device=G.device)
        for j in perm:
            j = int(j)
            if i == j:
                continue
            gj = Gref[j]
            denom = torch.dot(gj, gj) + eps
            dot = torch.dot(gi, gj)
            if dot < 0:
                gi = gi - (dot / denom) * gj
        Gm[i] = gi

    g = (w * Gm).sum(dim=0)
    return g, {}


@torch.no_grad()
def graddrop(G: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    GradDrop (sign dropout per-dimension).
    """
    Gw = w * G
    S = Gw.sum(dim=0)
    A = Gw.abs().sum(dim=0)
    P = 0.5 * (1.0 + S / (A + eps))

    U = torch.rand_like(P)
    choose_pos = (P > U)

    signs = torch.sign(Gw)
    mask = torch.ones_like(Gw)

    drop_pos = (~choose_pos)[None, :] & (signs > 0)
    drop_neg = ( choose_pos)[None, :] & (signs < 0)
    mask[drop_pos | drop_neg] = 0.0

    g_final = (Gw * mask).sum(dim=0)
    conflict = ((signs > 0).any(dim=0) & (signs < 0).any(dim=0))
    stats = {"conflict_frac": conflict.float().mean(), "P_mean": P.mean()}
    return g_final, stats


@torch.no_grad()
def mgda_min_norm(
    G: torch.Tensor,
    iters: int = 80,
    lr: float = 0.2,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    MGDA / MinNorm:  min_{alpha in simplex} || sum_t alpha_t g_t ||^2
    torch-only projected gradient on simplex (approx).
    """
    T = G.shape[0]
    alpha = torch.full((T,), 1.0 / T, device=G.device, dtype=G.dtype)

    for _ in range(int(iters)):
        g = (alpha[:, None] * G).sum(dim=0)          # [P]
        grad = 2.0 * (G @ g)                         # [T]
        alpha = _proj_simplex(alpha - lr * grad, z=1.0)

    g_final = (alpha[:, None] * G).sum(dim=0)
    stats = {
        "alpha_min": alpha.min(),
        "alpha_max": alpha.max(),
        "alpha_entropy": -(alpha.clamp_min(eps) * alpha.clamp_min(eps).log()).sum(),
    }
    return g_final, stats


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
    CAGrad (NeurIPS'21) 常用 torch-only PGD 近似版本（工程里很常见）。
    直觉：在 simplex 上找一个组合方向，兼顾接近 raw + 减少 worst conflict。
    """
    # raw grad uses provided w (external weights)
    g0 = (w * G).sum(dim=0)                          # [P]
    g0n = g0.norm() + eps

    T = G.shape[0]
    a = torch.full((T,), 1.0 / T, device=G.device, dtype=G.dtype)

    # optimize a on simplex
    for _ in range(int(iters)):
        g = (a[:, None] * G).sum(dim=0)              # [P]
        # objective: ||g - g0||^2  - 2*alpha*g0n*(min_t cos(g, g_t))  (smooth via logsumexp)
        g_norm = g.norm() + eps
        cos = (G @ g) / ((G.norm(dim=1) + eps) * g_norm)            # [T]
        # smooth-min
        tau = 10.0
        smooth_min = -(torch.logsumexp(-tau * cos, dim=0) / tau)

        obj = (g - g0).pow(2).sum() - 2.0 * float(alpha) * g0n * smooth_min
        # gradient wrt a: d/d a_i = <d obj/dg, g_i>
        # compute d obj/dg approximately by autograd on a small graph? we are in no_grad.
        # Use a finite-diff-free surrogate: use direction combining terms:
        # d/dg ||g-g0||^2 = 2(g-g0), and for smooth_min term use grad approx:
        # smooth_min = sum_i pi_i cos_i, pi = softmax(-tau*cos)
        pi = torch.softmax(-tau * cos, dim=0)                         # [T]
        # grad of cos_i wrt g is roughly: g_i_unit / ||g|| - cos_i * g_unit / ||g||
        Gi_unit = G / (G.norm(dim=1, keepdim=True) + eps)             # [T,P]
        g_unit = g / g_norm
        dcos_dg = (Gi_unit - cos[:, None] * g_unit[None, :]) / g_norm # [T,P]
        dsmin_dg = -(pi[:, None] * dcos_dg).sum(dim=0)                # [P], because smooth_min = -logsumexp(-tau cos)/tau

        d_obj_dg = 2.0 * (g - g0) - 2.0 * float(alpha) * g0n * dsmin_dg
        grad_a = (G @ d_obj_dg)                                       # [T]

        a = _proj_simplex(a - lr * grad_a, z=1.0)

    g = (a[:, None] * G).sum(dim=0)

    # final mix: (1 - alpha)*g0 + alpha*g   (common engineering form)
    g_final = (1.0 - float(alpha)) * g0 + float(alpha) * g
    stats = {
        "a_min": a.min(),
        "a_max": a.max(),
        "a_entropy": -(a.clamp_min(eps) * a.clamp_min(eps).log()).sum(),
    }
    return g_final, stats