# grad_methods_C.py
# C派：博弈 / Pareto / bargaining（Nash-MTL 等）
import torch
from typing import Dict, Tuple


@torch.no_grad()
def _proj_simplex(v: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    if v.numel() == 1:
        return v.new_tensor([z]).clamp_min(0.0)
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - z
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    rho = torch.nonzero(cond, as_tuple=False).max()
    theta = cssv[rho] / (rho + 1.0)
    return torch.clamp(v - theta, min=0.0)


@torch.no_grad()
def nash_mtl(
    G: torch.Tensor,             # [T,P]
    iters: int = 80,
    lr: float = 0.2,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Nash-MTL (ICML'22) 的 torch-only 近似：
    maximize  sum_t log( <g, g_t>_+ )  over alpha in simplex, where g = sum alpha_t g_t.
    这是一个常见的 bargaining 近似（比“原始KKT形式”更工程友好）。
    """
    T = G.shape[0]
    a = torch.full((T,), 1.0 / T, device=G.device, dtype=G.dtype)

    Gn = G / (G.norm(dim=1, keepdim=True) + eps)   # normalize helps stability

    for _ in range(int(iters)):
        g = (a[:, None] * Gn).sum(dim=0)                 # [P]
        # payoffs: <g, g_t> in normalized space
        p = (Gn @ g).clamp_min(eps)                      # [T]
        obj = p.log().sum()

        # gradient wrt a_i: d/d a_i sum_t log( <g, g_t> ) = sum_t (1/p_t) * <g_i, g_t>
        # where g = sum_j a_j g_j, so d<g, g_t>/d a_i = <g_i, g_t>
        # => grad_a_i = sum_t ( <g_i, g_t> / p_t )
        # compute M_ij = <g_i, g_j>
        M = Gn @ Gn.T                                    # [T,T]
        grad_a = (M / p[None, :]).sum(dim=1)             # [T]

        # ascent on obj
        a = _proj_simplex(a + lr * grad_a, z=1.0)

    g_final = (a[:, None] * G).sum(dim=0)                # use original (unnormalized) grads to step
    stats = {
        "a_min": a.min(),
        "a_max": a.max(),
        "a_entropy": -(a.clamp_min(eps) * a.clamp_min(eps).log()).sum(),
        "payoff_mean": ((G @ (g_final / (g_final.norm() + eps))).clamp_min(eps)).mean(),
    }
    return g_final, stats