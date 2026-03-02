# gradient_wrapper/grad_gpop.py
import torch
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


# -------------------------
# tiny utils (unified style)
# -------------------------

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


def _merge(*parts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for d in parts:
        out.update(d)
    return out


def _safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.dot(a, b) / ((a.norm() + eps) * (b.norm() + eps))


def _quantile_kth(x: torch.Tensor, q: float) -> torch.Tensor:
    """
    x: 1D tensor
    returns kthvalue approximating q-quantile (q in (0,1)).
    """
    if x.numel() == 0:
        return x.new_tensor(0.0)
    n = x.numel()
    k = int(q * n)
    # kthvalue uses 1-based k; clamp to [1, n]
    k = max(1, min(n, k))
    return x.kthvalue(k).values


# -------------------------
# config
# -------------------------

@dataclass
class GpopConfig:
    monitor: bool = True
    policy: bool = False

    beta: float = 0.99
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


# -------------------------
# main
# -------------------------

class GpopCommonGate:
    """
    Unified outputs:
      - monitor() returns a FLAT dict of tensors with stable prefixes:
          gpop.base.*
          gpop.common.geom.*
          gpop.common.lag.*
          gpop.common.task.* /<task>
          gpop.common.final.*   (if g_final provided)
      - apply_policy() returns (g_new, stats) where stats keys are:
          gpop.policy.*
      - apply() convenience merges monitor + optional policy.
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

        # monitor-only states
        self._prev_g_raw_c: Optional[torch.Tensor] = None  # [Pc]
        self._prev_gt_c: Dict[str, torch.Tensor] = {}      # task -> [Pc]
        self._gpop_task_c: Dict[str, torch.Tensor] = {}    # task -> EMA_t [Pc]

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
    def monitor(
        self,
        G: torch.Tensor,                     # [T,P]
        g_raw: torch.Tensor,                 # [P]
        g_final: Optional[torch.Tensor] = None,  # [P]
        names: Optional[List[str]] = None,       # len==T
    ) -> Dict[str, torch.Tensor]:
        if not bool(self.cfg.monitor):
            return {}

        eps = float(self.cfg.eps)
        device = g_raw.device
        dtype = g_raw.dtype

        cmask = self.common_mask(device=device)
        Gc = G[:, cmask]          # [T,Pc]
        g_raw_c = g_raw[cmask]    # [Pc]

        # base bucket (always returned if monitor is on)
        base = _to_tdict({
            "monitor": 1.0,
            "common_frac_params": cmask.float().mean(),
            "common_dim": float(Gc.shape[1]),
            "T": float(Gc.shape[0]),
        }, device=device, dtype=dtype)

        # first call: init EMA then return base only (but stable keys are nice)
        if self.Gpop_common is None:
            self._ema_update(g_raw_c)
            self._prev_g_raw_c = g_raw_c.detach().clone()
            # return with base only (you can also add zeros for other buckets if你强迫全key稳定)
            return _prefix(base, "gpop.base")

        gpop = self.Gpop_common  # OLD reference for metrics
        rho = self._rho(Gc, gpop)

        # ---- lag/switch bucket ----
        if self._prev_g_raw_c is None:
            graw_cos_prev = g_raw_c.new_tensor(0.0)
            graw_prev_norm = g_raw_c.new_tensor(0.0)
        else:
            prev = self._prev_g_raw_c
            graw_cos_prev = _safe_cos(g_raw_c, prev, eps)
            graw_prev_norm = prev.norm()

        gpop_cos_raw = _safe_cos(g_raw_c, gpop, eps)

        lag = _to_tdict({
            "graw_cos_prev": graw_cos_prev,
            "graw_prev_norm": graw_prev_norm,
            "gpop_cos_raw": gpop_cos_raw,
            "g_raw_norm": (g_raw_c.norm() + eps),
            "gpop_norm": (gpop.norm() + eps),
            "rho.mean": rho.mean(),
            "rho.min": rho.min(),
            "rho.std": rho.std(unbiased=False) if rho.numel() > 1 else rho.new_tensor(0.0),
        }, device=device, dtype=dtype)

        # ---- common geometry bucket ----
        g_t_common_norm = Gc.norm(dim=1) + eps
        Gu = Gc / g_t_common_norm[:, None]
        cos_tt = Gu @ Gu.T
        triu = torch.triu(torch.ones_like(cos_tt, dtype=torch.bool), diagonal=1)
        cos_vals = cos_tt[triu]  # [T*(T-1)/2]

        neg_mask = (cos_vals < 0) if cos_vals.numel() else None
        if cos_vals.numel() == 0:
            cos_mean = cos_tt.new_tensor(0.0)
            cos_min = cos_tt.new_tensor(0.0)
            cos_neg_frac = cos_tt.new_tensor(0.0)
            cos_neg_mean = cos_tt.new_tensor(0.0)
            p05 = cos_tt.new_tensor(0.0)
            p10 = cos_tt.new_tensor(0.0)
        else:
            cos_mean = cos_vals.mean()
            cos_min = cos_vals.min()
            cos_neg_frac = (cos_vals < 0).float().mean()
            if neg_mask is not None and neg_mask.any():
                cos_neg_mean = cos_vals[neg_mask].mean()
            else:
                cos_neg_mean = cos_vals.new_tensor(0.0)
            p05 = _quantile_kth(cos_vals, 0.05)
            p10 = _quantile_kth(cos_vals, 0.10)

        dot_raw = (Gc @ g_raw_c)  # [T]
        raw_viol = (dot_raw < 0).float().mean()

        geom = _to_tdict({
            "cos_tt.mean": cos_mean,
            "cos_tt.min": cos_min,
            "cos_tt.neg_frac": cos_neg_frac,
            "cos_tt.neg_mean": cos_neg_mean,
            "cos_tt.p05": p05,
            "cos_tt.p10": p10,
            "viol_frac": raw_viol,
            "dot.min": dot_raw.min(),
            "dot.mean": dot_raw.mean(),
        }, device=device, dtype=dtype)

        # ---- per-task bucket (optional) ----
        task: Dict[str, torch.Tensor] = {}
        if names is not None and len(names) == Gc.shape[0]:
            beta = float(self.cfg.beta)

            share = g_t_common_norm / (g_t_common_norm.sum() + eps)
            cos_raw_each = (Gc @ g_raw_c) / (g_t_common_norm * (g_raw_c.norm() + eps))

            for i, k in enumerate(names):
                gt_c = Gc[i]

                task[f"rho/{k}"] = rho[i]
                task[f"dot_raw/{k}"] = dot_raw[i]
                task[f"share/{k}"] = share[i]
                task[f"cos_raw/{k}"] = cos_raw_each[i]
                task[f"g_norm/{k}"] = g_t_common_norm[i]

                # time correlation of gt
                prev_gt = self._prev_gt_c.get(k, None)
                if prev_gt is None:
                    task[f"gt_cos_prev/{k}"] = gt_c.new_tensor(0.0)
                else:
                    task[f"gt_cos_prev/{k}"] = _safe_cos(gt_c, prev_gt, eps)
                self._prev_gt_c[k] = gt_c.detach().clone()

                # per-task EMA reference
                gpop_t = self._gpop_task_c.get(k, None)
                if gpop_t is None:
                    self._gpop_task_c[k] = gt_c.detach().clone()
                    task[f"gt_cos_gpop_t/{k}"] = gt_c.new_tensor(0.0)
                    task[f"gpop_t_norm/{k}"] = (gt_c.norm() + eps)
                    task[f"gpop_t_delta_over_norm/{k}"] = gt_c.new_tensor(0.0)
                else:
                    task[f"gt_cos_gpop_t/{k}"] = _safe_cos(gt_c, gpop_t, eps)
                    task[f"gpop_t_norm/{k}"] = (gpop_t.norm() + eps)
                    gpop_t_new = beta * gpop_t + (1.0 - beta) * gt_c.detach()
                    delta = (gpop_t_new - gpop_t).norm()
                    task[f"gpop_t_delta_over_norm/{k}"] = delta / (gpop_t.norm() + eps)
                    self._gpop_task_c[k] = gpop_t_new

        task = _to_tdict(task, device=device, dtype=dtype)

        # ---- final bucket (optional) ----
        final = {}
        if g_final is not None:
            g_final_c = g_final[cmask]
            dot_fin = (Gc @ g_final_c)
            final = _to_tdict({
                "viol_frac": (dot_fin < 0).float().mean(),
                "dot.min": dot_fin.min(),
                "dot.mean": dot_fin.mean(),
                "g_norm": (g_final_c.norm() + eps),
            }, device=device, dtype=dtype)

        # update EMA reference for next step (logging)
        self._ema_update(g_raw_c)
        self._prev_g_raw_c = g_raw_c.detach().clone()

        # merge + prefix flat
        out = _merge(
            _prefix(base, "gpop.base"),
            _prefix(geom, "gpop.common.geom"),
            _prefix(lag, "gpop.common.lag"),
            _prefix(task, "gpop.common.task"),
            _prefix(final, "gpop.common.final") if len(final) else {},
        )
        return {k: v.detach() for k, v in out.items()}

    @torch.no_grad()
    def apply_policy(
        self,
        G: torch.Tensor,
        g_raw: torch.Tensor,
        g_final: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        eps = float(self.cfg.eps)
        device = g_final.device
        dtype = g_final.dtype

        cmask = self.common_mask(device=device)
        Gc = G[:, cmask]
        g_raw_c = g_raw[cmask]

        Gpop_common = self.Gpop_common if self.Gpop_common is not None else g_raw_c.detach().clone()
        rho = self._rho(Gc, Gpop_common)
        rho_thr = g_raw_c.new_tensor(float(self.cfg.rho_thr))
        can_update = (rho.min() >= rho_thr)

        g_new = g_final.clone()

        froze = 0.0
        if (not bool(can_update.item())) and bool(self.cfg.freeze_common_on_fail):
            g_new[cmask] = 0.0
            froze = 1.0

        updated = 0.0
        if bool(can_update.item()):
            self._ema_update(g_raw_c)
            updated = 1.0

        sup_applied = 0.0
        if float(self.cfg.sup_lambda) > 0.0:
            lam = g_raw_c.new_tensor(float(self.cfg.sup_lambda))
            if self.cfg.sup_mode == "pull_to_gpop":
                g_new[cmask] = g_new[cmask] + lam * (Gpop_common / (Gpop_common.norm() + eps))
                sup_applied = 1.0
            elif self.cfg.sup_mode == "proj_to_gpop":
                gc = g_new[cmask]
                dot = torch.dot(gc, Gpop_common)
                if dot < 0.0:
                    g_new[cmask] = gc - (dot / (Gpop_common.dot(Gpop_common) + eps)) * Gpop_common
                    sup_applied = 1.0
            else:
                raise ValueError(f"[gpop] unknown sup_mode: {self.cfg.sup_mode}")

        st = _to_tdict({
            "policy": 1.0,
            "rho_thr": rho_thr,
            "rho_min": rho.min(),
            "updated": updated,
            "froze_common": froze,
            "sup_applied": sup_applied,
        }, device=device, dtype=dtype)

        return g_new, _prefix(st, "gpop.policy")

    @torch.no_grad()
    def apply(
        self,
        G: torch.Tensor,
        g_raw: torch.Tensor,
        g_final: torch.Tensor,
        policy: Optional[bool] = None,
        names: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        stats = {}
        if bool(self.cfg.monitor):
            stats.update(self.monitor(G=G, g_raw=g_raw, g_final=g_final, names=names))

        do_policy = bool(self.cfg.policy) if policy is None else bool(policy)
        if do_policy:
            g_final, st = self.apply_policy(G=G, g_raw=g_raw, g_final=g_final)
            stats.update(st)

        return g_final, stats

    def state_dict(self):
        return {
            "Gpop_common": None if self.Gpop_common is None else self.Gpop_common.detach().cpu(),
            "prev_g_raw_c": None if self._prev_g_raw_c is None else self._prev_g_raw_c.detach().cpu(),
        }

    def load_state_dict(self, st: dict, device=None, dtype=None):
        if st is None:
            self.Gpop_common = None
            self._prev_g_raw_c = None
            return

        gc = st.get("Gpop_common", None)
        prev = st.get("prev_g_raw_c", None)

        if gc is None:
            self.Gpop_common = None
        else:
            if device is None:
                device = gc.device
            if dtype is None:
                dtype = gc.dtype
            self.Gpop_common = gc.to(device=device, dtype=dtype)

        if prev is None:
            self._prev_g_raw_c = None
        else:
            if device is None:
                device = prev.device
            if dtype is None:
                dtype = prev.dtype
            self._prev_g_raw_c = prev.to(device=device, dtype=dtype)