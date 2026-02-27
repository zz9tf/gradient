# gradient_wrapper/grad_gpop.py
import torch
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class GpopConfig:
    # 永远监视（建议默认 True）
    monitor: bool = True

    # 默认不做策略（建议默认 False）
    policy: bool = False

    beta: float = 0.999
    rho_thr: float = 0.0
    freeze_common_on_fail: bool = True

    sup_lambda: float = 0.0
    sup_mode: str = "pull_to_gpop"  # pull_to_gpop | proj_to_gpop

    eps: float = 1e-8

    def validate(self):
        if self.monitor or self.policy:
            if not (0.0 < float(self.beta) < 1.0):
                raise ValueError(f"[gpop] beta must be in (0,1), got {self.beta}")
            if not (-1.0 <= float(self.rho_thr) <= 1.0):
                raise ValueError(f"[gpop] rho_thr must be in [-1,1), got {self.rho_thr}")
            if float(self.sup_lambda) < 0:
                raise ValueError(f"[gpop] sup_lambda must be >= 0, got {self.sup_lambda}")
            if self.sup_mode not in ("pull_to_gpop", "proj_to_gpop"):
                raise ValueError(f"[gpop] sup_mode must be pull_to_gpop|proj_to_gpop, got {self.sup_mode}")


class GpopCommonGate:
    """
    - common_mask: define common subspace
    - monitor(): ALWAYS safe, never changes gradients
    - apply_policy(): optional, changes gradients and updates EMA (only if you call it)
    """

    def __init__(
        self,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        common_param_filter: Callable[[str, torch.nn.Parameter], bool],
        cfg: GpopConfig,
    ):
        self.cfg = cfg
        self.cfg.validate()

        if common_param_filter is None:
            raise ValueError("[gpop] common_param_filter must be provided")

        mask = []
        common_names = []
        for n, p in named_params:
            is_common = bool(common_param_filter(n, p))
            if is_common:
                common_names.append(n)
            mask.append(torch.full((p.numel(),), is_common, dtype=torch.bool))
        self._common_mask_cpu = torch.cat(mask, dim=0)  # [P] bool on CPU
        self.common_names = common_names

        self.Gpop_common: Optional[torch.Tensor] = None  # [Pc]

    def common_mask(self, device):
        return self._common_mask_cpu.to(device=device)

    @torch.no_grad()
    def _rho(self, Gc: torch.Tensor, gpop_c: torch.Tensor) -> torch.Tensor:
        eps = float(self.cfg.eps)
        gpop_norm = gpop_c.norm() + eps
        g_norm = Gc.norm(dim=1) + eps
        return (Gc @ gpop_c) / (g_norm * gpop_norm)

    @torch.no_grad()
    def _ema_update(self, g_raw_c: torch.Tensor):
        beta = float(self.cfg.beta)
        if self.Gpop_common is None:
            self.Gpop_common = g_raw_c.detach().clone()
        else:
            self.Gpop_common = beta * self.Gpop_common + (1.0 - beta) * g_raw_c.detach()

    @torch.no_grad()
    def _ensure_gpop_init(self, g_raw_c: torch.Tensor):
        if self.Gpop_common is None:
            self.Gpop_common = g_raw_c.detach().clone()

    @torch.no_grad()
    def monitor(
        self,
        G: torch.Tensor,        # [T,P]
        g_raw: torch.Tensor,    # [P]
        g_final: Optional[torch.Tensor] = None,  # [P] optional; if given compute final viol too
        names: Optional[List[str]] = None,       # optional task names for per-task keys
    ) -> Dict[str, torch.Tensor]:
        """
        Pure logging: NEVER modifies gradients, NEVER EMA update (unless你想的话也可改).
        """
        if not bool(self.cfg.monitor):
            return {}

        eps = float(self.cfg.eps)
        device = g_raw.device
        cmask = self.common_mask(device=device)
        Gc = G[:, cmask]       # [T,Pc]
        g_raw_c = g_raw[cmask] # [Pc]

        # init reference so rho is defined
        self._ensure_gpop_init(g_raw_c)

        rho = self._rho(Gc, self.Gpop_common)  # [T]

        # pairwise cos within common
        gnorm = (Gc.norm(dim=1) + eps)  # [T]
        Gu = Gc / gnorm[:, None]
        cos_tt = Gu @ Gu.T
        triu = torch.triu(torch.ones_like(cos_tt, dtype=torch.bool), diagonal=1)
        cos_vals = cos_tt[triu]

        # violation in common: does update hurt a task first-order?
        dot_raw = (Gc @ g_raw_c)  # [T]
        raw_viol = (dot_raw < 0).float().mean()

        out: Dict[str, torch.Tensor] = {
            "gpop_monitor": g_raw.new_tensor(1.0),
            "common_frac_params": cmask.float().mean(),

            "gpop_rho_mean": rho.mean(),
            "gpop_rho_min": rho.min(),

            "cos_tt_common_mean": cos_vals.mean() if cos_vals.numel() else cos_tt.new_tensor(0.0),
            "cos_tt_common_min":  cos_vals.min()  if cos_vals.numel() else cos_tt.new_tensor(0.0),
            "cos_tt_common_neg_frac": (cos_vals < 0).float().mean() if cos_vals.numel() else cos_tt.new_tensor(0.0),

            "raw_viol_frac_common": raw_viol,
            "dot_raw_common_min": dot_raw.min(),
            "dot_raw_common_mean": dot_raw.mean(),

            "g_raw_common_norm": g_raw_c.norm(),
            "gpop_norm": (self.Gpop_common.norm() + eps),
        }

        if g_final is not None:
            g_final_c = g_final[cmask]
            dot_fin = (Gc @ g_final_c)
            out.update({
                "final_viol_frac_common": (dot_fin < 0).float().mean(),
                "dot_final_common_min": dot_fin.min(),
                "dot_final_common_mean": dot_fin.mean(),
                "g_final_common_norm": g_final_c.norm(),
            })

        # optional per-task stats (建议只开少量任务，否则 log 很大)
        if names is not None and len(names) == rho.numel():
            for i, k in enumerate(names):
                out[f"gpop_rho/{k}"] = rho[i]
                out[f"dot_raw_common/{k}"] = dot_raw[i]

        return {k: v.detach() for k, v in out.items()}

    @torch.no_grad()
    def apply_policy(
        self,
        G: torch.Tensor,
        g_raw: torch.Tensor,
        g_final: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Real behavior: freeze/EMA update/supervision.
        ONLY called when you want policy to happen.
        """
        eps = float(self.cfg.eps)
        device = g_final.device
        cmask = self.common_mask(device=device)
        Gc = G[:, cmask]
        g_raw_c = g_raw[cmask]

        self._ensure_gpop_init(g_raw_c)

        rho = self._rho(Gc, self.Gpop_common)
        rho_thr = g_raw_c.new_tensor(float(self.cfg.rho_thr))
        can_update = (rho.min() >= rho_thr)

        g_new = g_final.clone()

        if (not bool(can_update.item())) and bool(self.cfg.freeze_common_on_fail):
            g_new[cmask] = 0.0

        if bool(can_update.item()):
            self._ema_update(g_raw_c)

        if float(self.cfg.sup_lambda) > 0.0:
            lam = g_raw_c.new_tensor(float(self.cfg.sup_lambda))
            if self.cfg.sup_mode == "pull_to_gpop":
                g_new[cmask] = g_new[cmask] + lam * (self.Gpop_common / (self.Gpop_common.norm() + eps))
            else:  # proj_to_gpop
                gc = g_new[cmask]
                dot = torch.dot(gc, self.Gpop_common)
                if dot < 0:
                    g_new[cmask] = gc - (dot / (self.Gpop_common.dot(self.Gpop_common) + eps)) * self.Gpop_common

        st = {
            "gpop_policy": g_raw.new_tensor(1.0),
            "gpop_rho_thr": rho_thr.detach(),
            "gpop_updated": can_update.float().detach(),
        }
        return g_new, st

    @torch.no_grad()
    def apply(
        self,
        G: torch.Tensor,
        g_raw: torch.Tensor,
        g_final: torch.Tensor,
        policy: Optional[bool] = None,
        names: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convenience:
        - always monitor (if cfg.monitor)
        - optionally apply policy (if policy==True or cfg.policy==True)
        """
        stats = self.monitor(G=G, g_raw=g_raw, g_final=g_final, names=names)

        do_policy = bool(self.cfg.policy) if policy is None else bool(policy)
        if do_policy:
            g_final, st = self.apply_policy(G=G, g_raw=g_raw, g_final=g_final)
            stats.update(st)

        return g_final, stats

    def state_dict(self):
        return {
            "Gpop_common": None if self.Gpop_common is None else self.Gpop_common.detach().cpu(),
        }

    def load_state_dict(self, st: dict, device=None, dtype=None):
        if st is None:
            self.Gpop_common = None
            return
        gc = st.get("Gpop_common", None)
        if gc is None:
            self.Gpop_common = None
            return
        if device is None:
            device = gc.device
        if dtype is None:
            dtype = gc.dtype
        self.Gpop_common = gc.to(device=device, dtype=dtype)