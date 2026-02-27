# grad_methods_B.py
# B派：权重/标量化（不“手术”梯度，主要动态调 w_t，再做 sum）
import torch
from typing import Dict, Tuple, Optional


@torch.no_grad()
def dwa_weights(
    losses_vec: torch.Tensor,          # [T] current losses
    prev_losses_vec: Optional[torch.Tensor],
    Ttemp: float = 2.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    DWA (CVPR'19): weights based on loss change ratios.
    w_t ∝ exp( r_t / T ), r_t = L_t(k-1) / (L_t(k-2)+eps)
    Needs prev loss history; wrapper should store prev2.
    """
    T = losses_vec.numel()
    if prev_losses_vec is None:
        w = torch.full((T,), 1.0 / T, device=losses_vec.device, dtype=losses_vec.dtype)
        return w, {"dwa_init": torch.tensor(1.0, device=losses_vec.device)}
    r = losses_vec / (prev_losses_vec + eps)
    w = torch.softmax(r / float(Ttemp), dim=0)
    return w, {"dwa_r_mean": r.mean(), "dwa_r_max": r.max()}


@torch.no_grad()
def gradnorm_weights(
    G: torch.Tensor,                   # [T,P]
    losses_vec: torch.Tensor,          # [T]
    init_losses_vec: Optional[torch.Tensor],
    alpha: float = 1.5,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    GradNorm (ICML'18) 的“无可学习参数”近似：
    原论文是学习 w_t 使得各任务梯度范数匹配目标。
    这里给一个实用近似：w_t ∝ (target_gnorm / (||g_t||+eps)).
    target_gnorm 由相对训练速率 r_t = L_t / L_t0 的幂决定。
    """
    T = G.shape[0]
    gnorm = G.norm(dim=1) + eps  # [T]

    if init_losses_vec is None:
        init_losses_vec = losses_vec.detach().clone().clamp_min(eps)

    r = (losses_vec / init_losses_vec.clamp_min(eps)).clamp_min(eps)  # [T]
    r_bar = r.mean()
    target = (r / (r_bar + eps)).pow(float(alpha))                    # [T]
    target_g = (gnorm.mean() * target).detach()

    w = (target_g / gnorm).clamp_min(eps)
    w = w / w.sum().clamp_min(eps)

    stats = {
        "gradnorm_g_mean": gnorm.mean(),
        "gradnorm_r_mean": r.mean(),
        "gradnorm_w_max": w.max(),
    }
    return w, stats


@torch.no_grad()
def uw_heuristic_weights(
    losses_vec: torch.Tensor,          # [T]
    beta: float = 0.9,
    state: Optional[Dict[str, torch.Tensor]] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Uncertainty Weighting (Kendall'18) 原版需要学习 log(sigma^2)。
    你现在的 aggregator 只写 .grad，没法自然把 logvars 交给 optimizer。
    所以给一个“启发式UW”：用 EMA(loss) 当作 sigma proxy：w_t ∝ 1 / EMA(L_t)
    """
    T = losses_vec.numel()
    if state is None:
        state = {}
    ema = state.get("ema_loss", None)
    if ema is None:
        ema = losses_vec.detach().clone()
    ema = float(beta) * ema + (1.0 - float(beta)) * losses_vec.detach()
    state["ema_loss"] = ema

    w = (1.0 / (ema + eps)).clamp_min(eps)
    w = w / w.sum().clamp_min(eps)
    stats = {"uw_ema_mean": ema.mean(), "uw_w_max": w.max()}
    return w, stats, state


@torch.no_grad()
def apply_weighting_then_sum(
    G: torch.Tensor,                   # [T,P]
    w_task: torch.Tensor,              # [T]
) -> torch.Tensor:
    return (w_task[:, None] * G).sum(dim=0)