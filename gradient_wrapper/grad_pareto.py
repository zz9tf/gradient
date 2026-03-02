# grad_methods_C.py
# C派：博弈 / Pareto / bargaining（Nash-MTL 等）
import torch
from typing import Dict, Tuple


def _to_tdict(stats: Dict, device, dtype=None) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in stats.items():
        if v is None:
            out[k] = torch.tensor(0.0, device=device, dtype=dtype)
        elif torch.is_tensor(v):
            out[k] = v.detach()
        else:
            out[k] = torch.tensor(v, device=device, dtype=dtype)
    return out


def _prefix(stats: Dict[str, torch.Tensor], p: str) -> Dict[str, torch.Tensor]:
    return {f"{p}.{k}": v for k, v in stats.items()}


@torch.no_grad()
def _proj_simplex(v: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    if v.numel() == 1:
        return v.new_tensor([z]).clamp_min(0.0)

    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - float(z)
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0

    if not bool(cond.any().item()):
        return v.new_full((v.numel(),), 1.0 / v.numel()) * float(z)

    rho = torch.nonzero(cond, as_tuple=False).max()
    theta = cssv[rho] / (rho + 1.0)
    return torch.clamp(v - theta, min=0.0)


@torch.no_grad()
def nash_mtl(
    G: torch.Tensor,             # [T,P]
    iters: int = 80,
    lr: float = 0.2,
    eps: float = 1e-8,
    prefix: str = "c.nash",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Nash-MTL (ICML'22) 常用工程近似：
      maximize sum_t log(<g, g_t>_+) over a in simplex, g = sum a_t g_t.
    用 Gn 做几何，最终 step 用原始 G。
    """
    T = int(G.shape[0])
    device, dtype = G.device, G.dtype
    eps_t = float(eps)

    a = G.new_full((T,), 1.0 / max(1, T))

    Gn = G / (G.norm(dim=1, keepdim=True) + eps_t)
    M = Gn @ Gn.T  # [T,T] 固定

    for _ in range(int(iters)):
        g = (a[:, None] * Gn).sum(dim=0)          # [P]
        p = (Gn @ g).clamp_min(eps_t)             # [T]
        grad_a = (M / p[None, :]).sum(dim=1)      # [T]
        a = _proj_simplex(a + float(lr) * grad_a, z=1.0)

    g_final = (a[:, None] * G).sum(dim=0)

    a_cl = a.clamp_min(eps_t)
    stats = _to_tdict({
        "a_entropy": -(a_cl * a_cl.log()).sum(),
        "g_norm": g_final.norm(),
    }, device=device, dtype=dtype)

    return g_final, _prefix(stats, prefix)