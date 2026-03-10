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
from gradient_wrapper.grad_gpop import CommonGpopSurgery, CommonGpopConfig

# ✅ monitor (prefix already supported in your GradientMonitor via MonitorConfig.prefix)
# NOTE: adjust the import path if your file name differs (e.g. grad_monitor.py / monitor.py)
from gradient_wrapper.grad_block_monitor import GradientMonitor, MonitorConfig


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
    gpop_policy_kind: str = "cov_inv"
    gpop_ema_beta: float = 0.999
    gpop_merge_kind: str = "sum"
    gpop_task_grad_norm: bool = False
    gpop_task_grad_norm_common_only: bool = False
    gpop_cov_center: bool = True
    gpop_unbiased: bool = True
    gpop_cov_inv_damping: float = 1e-3
    gpop_cov_inv_max_iter: int = 30
    gpop_cov_inv_tol: float = 1e-6

    # ---- Monitor (optional): log block stats before/after gpop.apply ----
    # This uses your GradientMonitor with cfg.prefix="pre"/"post"
    gpop_monitor: bool = False
    monitor_detach: bool = True
    monitor_eps: float = 1e-8
    monitor_cov_unbiased: bool = True
    monitor_gpop_beta: float = 0.999      # monitor's own EMA for block_gpop alignment
    monitor_gpop_update: bool = True
    monitor_gpop_warmup_steps: int = 0
    monitor_cov_mode_k: int = 3

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

        if m not in allowed:
            raise ValueError(f"Unknown mode: {self.mode}. Allowed: {sorted(list(allowed))}")

        if float(self.eps) <= 0:
            raise ValueError(f"eps must be >0, got {self.eps}")

        if self.gpop_enabled:
            CommonGpopConfig(
                policy_kind=self.gpop_policy_kind,
                ema_beta=self.gpop_ema_beta,
                merge_kind=self.gpop_merge_kind,
                task_grad_norm=self.gpop_task_grad_norm,
                task_grad_norm_common_only=self.gpop_task_grad_norm_common_only,
                cov_center=self.gpop_cov_center,
                unbiased=self.gpop_unbiased,
                cov_inv_damping=self.gpop_cov_inv_damping,
                cov_inv_max_iter=self.gpop_cov_inv_max_iter,
                cov_inv_tol=self.gpop_cov_inv_tol,
                eps=self.eps,
            ).validate()

def _default_monitor_block_fn(name: str) -> str:
    """
    Extract block ID from param name for per-block recording.
    E.g. 'backbone.stem.0.weight' -> 'backbone.stem', 'cls_fc.weight' -> 'cls_fc'.
    """
    n = name[7:] if name.startswith("module.") else name
    base = n.replace(".weight", "").replace(".bias", "")
    parts = base.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "unknown")


class GradAggregator:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: GradAggConfig,
        param_filter: Optional[Callable[[str, torch.nn.Parameter], bool]] = None,
        common_param_filter: Optional[Callable[[str, torch.nn.Parameter], bool]] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.cfg = cfg
        self.cfg.validate()

        named_all = named_params(model, only=param_filter)
        if len(named_all) == 0:
            raise ValueError("No trainable params selected.")

        self._named_all = [(n, p) for n, p in named_all]  # [(name, param)]
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

        # step counter for monitors etc.
        self._step: int = 0

        # ---- gpop common gate ----
        self.gpop: Optional[CommonGpopSurgery] = None
        if bool(getattr(self.cfg, "gpop_enabled", False)):
            if common_param_filter is None:
                raise ValueError("gpop_enabled=True but common_param_filter is None")
            gpcfg = CommonGpopConfig(
                policy_kind=self.cfg.gpop_policy_kind,
                ema_beta=self.cfg.gpop_ema_beta,
                eps=self.cfg.eps,

                merge_kind=str(getattr(self.cfg, "gpop_merge_kind", "sum")),
                task_grad_norm=bool(getattr(self.cfg, "gpop_task_grad_norm", False)),
                task_grad_norm_common_only=bool(getattr(self.cfg, "gpop_task_grad_norm_common_only", False)),

                cov_center=bool(getattr(self.cfg, "gpop_cov_center", True)),
                unbiased=bool(getattr(self.cfg, "gpop_unbiased", True)),

                cov_inv_damping=float(getattr(self.cfg, "gpop_cov_inv_damping", 1e-3)),
                cov_inv_max_iter=int(getattr(self.cfg, "gpop_cov_inv_max_iter", 30)),
                cov_inv_tol=float(getattr(self.cfg, "gpop_cov_inv_tol", 1e-6)),
            )
            self.gpop = CommonGpopSurgery(
                self._named_all,
                common_param_filter=common_param_filter,
                cfg=gpcfg,
            )
            if verbose:
                print("[gpop] common tensors:", len(self.gpop.common_names), "examples:", self.gpop.common_names[:5])

        # ---- optional monitor pre/post for gpop impact ----
        self._monitor_pre: Optional[GradientMonitor] = None
        self._monitor_post: Optional[GradientMonitor] = None
        if bool(getattr(self.cfg, "gpop_monitor", False)):
            # Block recording: per-module blocks (backbone.stem, backbone.stage1, cls_fc, etc.),
            # independent of common/private (which is for gpop policy only).
            _monitor_block_fn = getattr(
                self.cfg, "monitor_block_split_fn", None
            ) or _default_monitor_block_fn

            mon_base = MonitorConfig(
                prefix="",  # set per-instance below
                eps=float(getattr(self.cfg, "monitor_eps", self.cfg.eps)),
                detach=bool(getattr(self.cfg, "monitor_detach", True)),
                cov_unbiased=bool(getattr(self.cfg, "monitor_cov_unbiased", True)),
                gpop_beta=float(getattr(self.cfg, "monitor_gpop_beta", 0.99)),
                gpop_update=bool(getattr(self.cfg, "monitor_gpop_update", True)),
                gpop_warmup_steps=int(getattr(self.cfg, "monitor_gpop_warmup_steps", 0)),
                # optional: if you exposed these in cfg
                cov_mode_k=int(getattr(self.cfg, "monitor_cov_mode_k", 3)),
                enable_global=bool(getattr(self.cfg, "monitor_enable_global", True)),
                enable_block_energy=bool(getattr(self.cfg, "monitor_enable_block_energy", True)),
                enable_block_cov=bool(getattr(self.cfg, "monitor_enable_block_cov", True)),
                enable_drift=bool(getattr(self.cfg, "monitor_enable_drift", True)),
                enable_block_gpop=bool(getattr(self.cfg, "monitor_enable_block_gpop", True)),
            )

            mon_pre = MonitorConfig(**{**mon_base.__dict__, "prefix": "pre"})
            mon_post = MonitorConfig(**{**mon_base.__dict__, "prefix": "post"})

            self._monitor_pre = GradientMonitor(self._named_all, block_split_fn=_monitor_block_fn, cfg=mon_pre)
            self._monitor_post = GradientMonitor(self._named_all, block_split_fn=_monitor_block_fn, cfg=mon_post)

            if verbose:
                print("[monitor] enabled: pre/post (blocks=per-module, e.g. backbone.stem, cls_fc)")

        if verbose:
            print("[grad] tensors:", len(self.params), "examples:", self.all_names[:5])

    def backward(self, losses: Dict[str, torch.Tensor], weights: Optional[Dict[str, float]] = None):
        if weights is None:
            weights = {k: 1.0 for k in losses.keys()}

        # bump step
        self._step += 1

        # clear grads
        for p in self.params:
            p.grad = None

        names = list(losses.keys())
        T = len(names)
        device = next(iter(losses.values())).device
        eps = float(self.cfg.eps)

        # per-task gradients (global space)
        g_list = []
        for i, k in enumerate(names):
            rg = (i != T - 1)
            g_list.append(gradvec(losses[k], self.params, retain_graph=rg))
        G = torch.stack(g_list, dim=0)  # [T,P]
        
        # external weights (for A-type methods that accept weights)
        w_external = torch.tensor(
            [float(weights.get(k, 1.0)) for k in names],
            device=device,
            dtype=G.dtype,
        ).view(T, 1)

        mode = self.cfg.mode.lower()
        MODE2ID = {
            "sum": 0, "pcgrad": 1, "graddrop": 2, "mgda": 3, "cagrad": 4,
            "dwa": 10, "gradnorm": 11, "uw_heuristic": 12,
            "nash_mtl": 20,
        }
        stats: Dict[str, torch.Tensor] = {
            "mode_id": torch.tensor(MODE2ID.get(mode, -1), device=device),
            "step": torch.tensor(float(self._step), device=device),
            "T": torch.tensor(float(T), device=device),
        }

        with torch.no_grad():
            self.overall_loss_raw = sum(losses[k].detach() * float(weights.get(k, 1.0)) for k in names)

        # vector of losses for B
        losses_vec = torch.stack([losses[k].detach() for k in names], dim=0).to(dtype=G.dtype)

        # -------------------- (0) monitor pre: BEFORE gpop surgery --------------------
        if self._monitor_pre is not None:
            # 这里给一个简单的参考 merged grad（等价于 external-weighted sum）
            g_ref_pre = (G * w_external).sum(dim=0)
            stats.update(self._monitor_pre.monitor(G=G.detach(), g_ref=g_ref_pre.detach(), step=self._step))

        # -------------------- (1) gpop surgery on per-task grads (BEFORE aggregation) --------------------
        G_used = G
        if self.gpop is not None:
            G_used, st_gpop = self.gpop.apply(G_used)   # ✅ now returns [T,P]
            stats.update(st_gpop)

        # -------------------- (2) monitor post: AFTER gpop surgery --------------------
        if self._monitor_post is not None:
            g_ref_post = (G_used * w_external).sum(dim=0)
            stats.update(self._monitor_post.monitor(G=G_used.detach(), g_ref=g_ref_post.detach(), step=self._step))

        # -------------------- (3) Now aggregate on G_used with your chosen mode --------------------
        mode = self.cfg.mode.lower()

        # A: surgery / projection
        if mode == "sum":
            g_final, st = sum_grad(G_used, w_external)
            stats.update(st)

        elif mode == "pcgrad":
            g_final, st = pcgrad(G_used, w_external, eps=eps)
            stats.update(st)

        elif mode == "graddrop":
            g_final, st = graddrop(G_used, w_external, eps=eps)
            stats.update(st)

        elif mode == "mgda":
            g_final, st = mgda_min_norm(G_used, iters=self.cfg.mgda_iters, lr=self.cfg.mgda_lr, eps=eps)
            stats.update(st)

        elif mode == "cagrad":
            g_final, st = cagrad(
                G_used, w_external,
                alpha=self.cfg.cagrad_alpha,
                iters=self.cfg.cagrad_iters,
                lr=self.cfg.cagrad_lr,
                eps=eps,
            )
            stats.update(st)

        # B: weighting / scalarization
        elif mode == "dwa":
            prev = self._prev_losses_2
            w_task, st = dwa_weights(losses_vec, prev_losses_vec=prev, Ttemp=self.cfg.dwa_T, eps=eps)
            g_final = apply_weighting_then_sum(G_used, w_task)
            stats.update(st)
            stats.update({"b.w.max": w_task.max(), "b.w.min": w_task.min()})

        elif mode == "gradnorm":
            if self._init_losses is None:
                self._init_losses = losses_vec.detach().clone().clamp_min(eps)
            w_task, st = gradnorm_weights(
                G_used, losses_vec,
                init_losses_vec=self._init_losses,
                alpha=self.cfg.gradnorm_alpha,
                eps=eps,
            )
            g_final = apply_weighting_then_sum(G_used, w_task)
            stats.update(st)
            stats.update({"w_task_max": w_task.max(), "w_task_min": w_task.min()})

        elif mode == "uw_heuristic":
            w_task, st, self._uw_state = uw_heuristic_weights(
                losses_vec, beta=self.cfg.uw_beta, state=self._uw_state, eps=eps
            )
            g_final = apply_weighting_then_sum(G_used, w_task)
            stats.update(st)
            stats.update({"w_task_max": w_task.max(), "w_task_min": w_task.min()})

        # C: game / Pareto
        elif mode == "nash_mtl":
            g_final, st = nash_mtl(G_used, iters=self.cfg.nash_iters, lr=self.cfg.nash_lr, eps=eps)
            stats.update(st)

        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")

        # -------------------- (4) update DWA history (2-step) --------------------
        self._prev_losses_2 = self._prev_losses_1
        self._prev_losses_1 = losses_vec.detach().clone()

        # ---- save diagnostics ----
        self.last_stats = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in stats.items()}

        # -------------------- write back grads --------------------
        grads_final = unflatten(g_final.detach(), self.params)
        for p, g in zip(self.params, grads_final):
            p.grad = g

        return self.last_stats

    def state_dict(self) -> dict:
        """State dict for checkpointing (e.g. Gpop common gate)."""
        out: Dict = {}
        if self.gpop is not None:
            out["gpop"] = self.gpop.state_dict()
        # monitor has temporal states too; optionally save if you want reproducible drift/gpop_ema stats
        if self._monitor_pre is not None:
            out["monitor_pre"] = {
                "prev_block_gmean": {k: v.detach().cpu() for k, v in self._monitor_pre.prev_block_gmean.items()},
                "prev_block_u": {k: v.detach().cpu() for k, v in self._monitor_pre.prev_block_u.items()},
                "block_gpop": {k: v.detach().cpu() for k, v in self._monitor_pre.block_gpop.items()},
                "prev_block_gpop": {k: v.detach().cpu() for k, v in self._monitor_pre.prev_block_gpop.items()},
                "step": int(self._monitor_pre._step),
            }
        if self._monitor_post is not None:
            out["monitor_post"] = {
                "prev_block_gmean": {k: v.detach().cpu() for k, v in self._monitor_post.prev_block_gmean.items()},
                "prev_block_u": {k: v.detach().cpu() for k, v in self._monitor_post.prev_block_u.items()},
                "block_gpop": {k: v.detach().cpu() for k, v in self._monitor_post.block_gpop.items()},
                "prev_block_gpop": {k: v.detach().cpu() for k, v in self._monitor_post.prev_block_gpop.items()},
                "step": int(self._monitor_post._step),
            }
        out["step"] = int(self._step)
        return out

    def load_state_dict(self, st: dict, strict: bool = False) -> None:
        """Restore state from checkpoint."""
        if st is None:
            return

        if "step" in st:
            self._step = int(st["step"])

        if self.gpop is not None and "gpop" in st:
            self.gpop.load_state_dict(st["gpop"])

        # restore monitor states (optional)
        dev = next(self.model.parameters()).device

        if self._monitor_pre is not None and "monitor_pre" in st:
            ms = st["monitor_pre"]
            self._monitor_pre.prev_block_gmean = {k: v.to(dev) for k, v in ms.get("prev_block_gmean", {}).items()}
            self._monitor_pre.prev_block_u = {k: v.to(dev) for k, v in ms.get("prev_block_u", {}).items()}
            self._monitor_pre.block_gpop = {k: v.to(dev) for k, v in ms.get("block_gpop", {}).items()}
            self._monitor_pre.prev_block_gpop = {k: v.to(dev) for k, v in ms.get("prev_block_gpop", {}).items()}
            self._monitor_pre._step = int(ms.get("step", 0))

        if self._monitor_post is not None and "monitor_post" in st:
            ms = st["monitor_post"]
            self._monitor_post.prev_block_gmean = {k: v.to(dev) for k, v in ms.get("prev_block_gmean", {}).items()}
            self._monitor_post.prev_block_u = {k: v.to(dev) for k, v in ms.get("prev_block_u", {}).items()}
            self._monitor_post.block_gpop = {k: v.to(dev) for k, v in ms.get("block_gpop", {}).items()}
            self._monitor_post.prev_block_gpop = {k: v.to(dev) for k, v in ms.get("prev_block_gpop", {}).items()}
            self._monitor_post._step = int(ms.get("step", 0))