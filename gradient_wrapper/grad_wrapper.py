# grad_wrapper.py
# Wrapper：统一计算 per-task grads，然后按 mode 调 A/B/C 策略，最后写回 .grad
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple

from gradient_wrapper.grad_surgery import sum_grad, pcgrad, graddrop, mgda_min_norm, cagrad
from gradient_wrapper.grad_weight import (
    dwa_weights, gradnorm_weights, uw_heuristic_weights, apply_weighting_then_sum
)
from gradient_wrapper.grad_pareto import nash_mtl
from gradient_wrapper.grad_gpop import GpopCommonGate, GpopConfig


def named_params(
    model: torch.nn.Module,
    only: Optional[Callable[[str, torch.nn.Parameter], bool]] = None
) -> List[Tuple[str, torch.nn.Parameter]]:
    out = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if only is None or bool(only(n, p)):
            out.append((n, p))
    return out


def safe_grads(grads, params):
    return [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]


def flatten(grads: List[torch.Tensor]) -> torch.Tensor:
    if len(grads) == 0:
        return torch.zeros(0)
    return torch.cat([g.reshape(-1) for g in grads], dim=0)


def unflatten(vec: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    out, off = [], 0
    for p in params:
        n = p.numel()
        out.append(vec[off: off + n].view_as(p))
        off += n
    return out


def gradvec(loss: torch.Tensor, params: List[torch.nn.Parameter], retain_graph: bool) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss, params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    grads = safe_grads(grads, params)
    return flatten(grads)


@dataclass
class GradAggConfig:
    mode: str = "sum"
    eps: float = 1e-8

    # A: CAGrad
    cagrad_alpha: float = 0.5
    cagrad_iters: int = 60
    cagrad_lr: float = 0.2

    # A: MGDA
    mgda_iters: int = 80
    mgda_lr: float = 0.2

    # C: Nash-MTL
    nash_iters: int = 80
    nash_lr: float = 0.2

    # B: DWA
    dwa_T: float = 2.0

    # B: GradNorm
    gradnorm_alpha: float = 1.5

    # B: UW heuristic
    uw_beta: float = 0.9
    
    # ---- Gpop common gate (optional) ----
    gpop_enabled: bool = False
    gpop_beta: float = 0.99
    gpop_rho_thr: float = 0.0
    gpop_freeze_common_on_fail: bool = True
    gpop_sup_lambda: float = 0.0
    gpop_sup_mode: str = "pull_to_gpop"  # pull_to_gpop | proj_to_gpop

    def validate(self):
        m = self.mode.lower()
        allowed = {
            # A
            "sum", "pcgrad", "graddrop", "mgda", "cagrad",
            # B
            "dwa", "gradnorm", "uw_heuristic",
            # C
            "nash_mtl",
        }
        if self.gpop_enabled:
            GpopConfig(
                enabled=True,
                beta=self.gpop_beta,
                rho_thr=self.gpop_rho_thr,
                freeze_common_on_fail=self.gpop_freeze_common_on_fail,
                sup_lambda=self.gpop_sup_lambda,
                sup_mode=self.gpop_sup_mode,
                eps=self.eps,
            ).validate()
        if m not in allowed:
            raise ValueError(f"Unknown mode: {self.mode}. Allowed: {sorted(list(allowed))}")


class GradAggregator:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: GradAggConfig,
        param_filter: Optional[Callable[[str, torch.nn.Parameter], bool]] = None,
        common_param_filter=None, 
        verbose=True
    ):
        self.model = model
        self.cfg = cfg
        self.cfg.validate()

        named_all = named_params(model, only=param_filter)
        
        self._named_all = [(n, p) for n, p in named_all]  # [(name, param)]
        self.gpop = None
        if bool(getattr(self.cfg, "gpop_enabled", False)):
            if common_param_filter is None:
                raise ValueError("gpop_enabled=True but common_param_filter is None")
            gpcfg = GpopConfig(
                enabled=True,
                beta=self.cfg.gpop_beta,
                rho_thr=self.cfg.gpop_rho_thr,
                freeze_common_on_fail=self.cfg.gpop_freeze_common_on_fail,
                sup_lambda=self.cfg.gpop_sup_lambda,
                sup_mode=self.cfg.gpop_sup_mode,
                eps=self.cfg.eps,
            )
            self.gpop = GpopCommonGate(self._named_all, common_param_filter=common_param_filter, cfg=gpcfg)
            if verbose:
                print("[gpop] common tensors:", len(self.gpop.common_names), "examples:", self.gpop.common_names[:5])
        
        if len(named_all) == 0:
            raise ValueError("No trainable params selected.")

        self.all_names = [n for n, _ in named_all]
        self.params = [p for _, p in named_all]

        # ---- B-method states ----
        self._prev_losses_1: Optional[torch.Tensor] = None  # L(k-1)
        self._prev_losses_2: Optional[torch.Tensor] = None  # L(k-2)
        self._init_losses: Optional[torch.Tensor] = None
        self._uw_state: Optional[Dict[str, torch.Tensor]] = None

        # diagnostics
        self.last_stats: Dict[str, torch.Tensor] = {}
        self.overall_loss_raw: Optional[torch.Tensor] = None
        self.shrink_ratio: Optional[torch.Tensor] = None
        self.cos_raw_final: Optional[torch.Tensor] = None

        if verbose:
            print("[grad] tensors:", len(self.params), "examples:", self.all_names[:5])

    def backward(self, losses: Dict[str, torch.Tensor], weights: Optional[Dict[str, float]] = None):
        if weights is None:
            weights = {k: 1.0 for k in losses.keys()}

        for p in self.params:
            p.grad = None

        names = list(losses.keys())
        T = len(names)
        device = next(iter(losses.values())).device
        eps = float(self.cfg.eps)

        # per-task gradients (global space)
        w_external = torch.tensor(
            [float(weights.get(k, 1.0)) for k in names],
            device=device,
            dtype=next(iter(losses.values())).dtype,
        ).view(T, 1)

        g_list = []
        for i, k in enumerate(names):
            rg = (i != T - 1)
            g_list.append(gradvec(losses[k], self.params, retain_graph=rg))
        G = torch.stack(g_list, dim=0)  # [T,P]

        # raw baseline
        g_raw = (w_external * G).sum(dim=0)

        mode = self.cfg.mode.lower()
        MODE2ID = {
            "sum": 0, "pcgrad": 1, "graddrop": 2, "mgda": 3, "cagrad": 4,
            "dwa": 10, "gradnorm": 11, "uw_heuristic": 12,
            "nash_mtl": 20,
        }
        stats: Dict[str, torch.Tensor] = {"mode_id": torch.tensor(MODE2ID.get(mode, -1), device=device)}
        # ===== conflict geometry logs (raw, before aggregation) =====
        with torch.no_grad():
            # per-task norms
            g_t_norm = G.norm(dim=1) + eps  # [T]

            # pairwise cosine among task grads
            G_unit = G / g_t_norm[:, None]  # [T,P]
            cos_tt = G_unit @ G_unit.T      # [T,T]
            triu = torch.triu(torch.ones_like(cos_tt, dtype=torch.bool), diagonal=1)
            cos_vals = cos_tt[triu]
            stats_geom = {
                "cos_tt_mean": cos_vals.mean() if cos_vals.numel() else cos_tt.new_tensor(0.0),
                "cos_tt_min":  cos_vals.min()  if cos_vals.numel() else cos_tt.new_tensor(0.0),
                "cos_tt_neg_frac": (cos_vals < 0).float().mean() if cos_vals.numel() else cos_tt.new_tensor(0.0),
            }

            # raw violation: whether raw summed gradient hurts a task (1st-order)
            dot_t_raw = (G @ g_raw)  # [T]
            raw_viol_frac = (dot_t_raw < 0).float().mean()

            stats_geom.update({
                "raw_viol_frac": raw_viol_frac,
                "dot_raw_min": dot_t_raw.min(),
                "dot_raw_mean": dot_t_raw.mean(),
            })
            
        stats.update(stats_geom)
        with torch.no_grad():
            self.overall_loss_raw = sum(losses[k].detach() * float(weights.get(k, 1.0)) for k in names)

        # vector of losses for B
        losses_vec = torch.stack([losses[k].detach() for k in names], dim=0).to(dtype=G.dtype)
        # -------------------- A: surgery / projection --------------------
        if mode == "sum":
            g_final, st = sum_grad(G, w_external)
            stats.update(st)

        elif mode == "pcgrad":
            g_final, st = pcgrad(G, w_external, eps=eps)
            stats.update(st)

        elif mode == "graddrop":
            g_final, st = graddrop(G, w_external, eps=eps)
            stats.update(st)

        elif mode == "mgda":
            # NOTE: MGDA learns alpha on simplex; external weights are not used inside the solver
            g_final, st = mgda_min_norm(G, iters=self.cfg.mgda_iters, lr=self.cfg.mgda_lr, eps=eps)
            stats.update(st)

        elif mode == "cagrad":
            g_final, st = cagrad(
                G, w_external,
                alpha=self.cfg.cagrad_alpha,
                iters=self.cfg.cagrad_iters,
                lr=self.cfg.cagrad_lr,
                eps=eps,
            )
            stats.update(st)

        # -------------------- B: weighting / scalarization --------------------
        elif mode == "dwa":
            # DWA needs L(k-1)/L(k-2). We keep 2-step history.
            prev = self._prev_losses_2
            w_task, st = dwa_weights(losses_vec, prev_losses_vec=prev, Ttemp=self.cfg.dwa_T, eps=eps)
            g_final = apply_weighting_then_sum(G, w_task)  # ignores external weights by design
            stats.update(st)
            stats.update({"w_task_max": w_task.max(), "w_task_min": w_task.min()})

        elif mode == "gradnorm":
            if self._init_losses is None:
                self._init_losses = losses_vec.detach().clone().clamp_min(eps)
            w_task, st = gradnorm_weights(
                G, losses_vec,
                init_losses_vec=self._init_losses,
                alpha=self.cfg.gradnorm_alpha,
                eps=eps,
            )
            g_final = apply_weighting_then_sum(G, w_task)
            stats.update(st)
            stats.update({"w_task_max": w_task.max(), "w_task_min": w_task.min()})

        elif mode == "uw_heuristic":
            w_task, st, self._uw_state = uw_heuristic_weights(
                losses_vec, beta=self.cfg.uw_beta, state=self._uw_state, eps=eps
            )
            g_final = apply_weighting_then_sum(G, w_task)
            stats.update(st)
            stats.update({"w_task_max": w_task.max(), "w_task_min": w_task.min()})

        # -------------------- C: game / Pareto --------------------
        elif mode == "nash_mtl":
            g_final, st = nash_mtl(G, iters=self.cfg.nash_iters, lr=self.cfg.nash_lr, eps=eps)
            stats.update(st)
        
        # ---- optional gpop common gate / supervision ----
        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")
        
        if self.gpop is not None:
            stats.update(self.gpop.monitor(G=G, g_raw=g_raw, g_final=g_final, names=names))

        # ===== conflict logs (final, after aggregation) =====
        with torch.no_grad():
            dot_t_final = (G @ g_final)  # [T]
            viol_frac = (dot_t_final < 0).float().mean()

            # normalized projection strength (helps see which task is sacrificed)
            g_t_norm = G.norm(dim=1) + eps
            proj_strength = dot_t_final / g_t_norm  # [T]

            stats.update({
                "viol_frac": viol_frac,
                "dot_final_min": dot_t_final.min(),
                "dot_final_mean": dot_t_final.mean(),
            })

            # per-task logs with stable keys
            for i, k in enumerate(names):
                stats[f"g_t_norm/{k}"] = g_t_norm[i]
                stats[f"dot_final/{k}"] = dot_t_final[i]
                stats[f"proj_final_over_norm/{k}"] = proj_strength[i]
                
        # -------------------- write back grads --------------------
        grads_final = unflatten(g_final.detach(), self.params)
        for p, g in zip(self.params, grads_final):
            p.grad = g

        # -------------------- update B states --------------------
        with torch.no_grad():
            # shift history for DWA
            self._prev_losses_2 = self._prev_losses_1
            self._prev_losses_1 = losses_vec.detach().clone()

            raw_n = g_raw.norm() + eps
            fin_n = g_final.norm() + eps
            self.shrink_ratio = fin_n / raw_n
            self.cos_raw_final = torch.dot(g_raw, g_final) / (raw_n * fin_n)

        stats.update({
            "overall_loss_raw": self.overall_loss_raw.detach(),
            "shrink_ratio": self.shrink_ratio.detach(),
            "cos_raw_final": self.cos_raw_final.detach(),
            "g_raw_norm": g_raw.detach().norm(),
            "g_final_norm": g_final.detach().norm(),
        })

        self.last_stats = {k: (v.detach() if torch.is_tensor(v) else torch.tensor(v, device=device)) for k, v in stats.items()}
        return self.last_stats