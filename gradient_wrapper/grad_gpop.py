# gradient_wrapper/grad_gpop.py
# Minimal "Common Gpop + Surgery" gate (paste-ready).
#
# What it does:
#   - (optional) per-task normalize G (to remove magnitude dominance; focus on conflict geometry)
#   - Maintain EMA gpop_base on COMMON params (Pc) using merged grad computed from G
#   - Build a reference direction v_ref on COMMON params using ONE of 3 policies:
#       (1) "gg"      : v_ref = (G^T G) gpop_base
#       (2) "cov_mul" : v_ref = Cov * gpop_base,   Cov = (X^T X)/denom
#       (3) "cov_inv" : v_ref = (Cov + λI)^{-1} gpop_base   (solved by CG; no explicit inverse)
#
#     Here:
#       G_common = G[:, cmask]   shape [T, Pc]
#       X = center(G_common) if cov_center else G_common
#       denom = (T-1) if unbiased else T
#
#   - Perform ONE surgery step on per-task grads along v_ref:
#       for tasks with dot(g_i, v_ref) < 0:
#           g_i <- g_i - (dot / ||v_ref||^2) * v_ref
#
#   - Output G_new where COMMON columns have been surgery-adjusted per-task.
#   - No freeze, no pull, no rho_thr, no other strategies.

import torch
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class CommonGpopConfig:
    # Choose reference-direction policy:
    #   - "gg":      (G^T G) gpop
    #   - "cov_mul": Cov gpop
    #   - "cov_inv": (Cov + λI)^{-1} gpop  (CG)
    policy_kind: str = "cov_mul"  # gg | cov_mul | cov_inv

    # Covariance construction flags (used by cov_mul / cov_inv)
    cov_center: bool = True
    unbiased: bool = True

    # EMA on gpop base
    ema_beta: float = 0.999

    # Optional: how to form merged grad for EMA from per-task G
    #   - "sum":  sum over tasks
    #   - "mean": mean over tasks
    merge_kind: str = "sum"  # sum | mean

    # Optional: normalize each task gradient before doing anything
    # This is recommended if you want to study "conflict only" (remove magnitude dominance).
    task_grad_norm: bool = False
    task_grad_norm_common_only: bool = False  # if True, compute norm on common dims but scale full vector

    # Numerical
    eps: float = 1e-8

    # Only for cov_inv
    cov_inv_damping: float = 1e-3
    cov_inv_max_iter: int = 30
    cov_inv_tol: float = 1e-6

    def validate(self):
        if self.policy_kind not in ["gg", "cov_mul", "cov_inv"]:
            raise ValueError(
                f"[gpop] policy_kind must be in ['gg','cov_mul','cov_inv'], got {self.policy_kind}"
            )
        # cov_mul/cov_inv: always center (true covariance). gg: no centering (Gram matrix).
        if self.policy_kind in ("cov_mul", "cov_inv"):
            self.cov_center = True
        if self.merge_kind not in ["sum", "mean"]:
            raise ValueError(f"[gpop] merge_kind must be in ['sum','mean'], got {self.merge_kind}")
        if not (0.0 < float(self.ema_beta) < 1.0):
            raise ValueError(f"[gpop] ema_beta must be in (0,1), got {self.ema_beta}")
        if float(self.eps) <= 0:
            raise ValueError(f"[gpop] eps must be > 0, got {self.eps}")
        if float(self.cov_inv_damping) < 0:
            raise ValueError(f"[gpop] cov_inv_damping must be >= 0, got {self.cov_inv_damping}")
        if int(self.cov_inv_max_iter) <= 0:
            raise ValueError(f"[gpop] cov_inv_max_iter must be > 0, got {self.cov_inv_max_iter}")
        if float(self.cov_inv_tol) <= 0:
            raise ValueError(f"[gpop] cov_inv_tol must be > 0, got {self.cov_inv_tol}")


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


class CommonGpopSurgery:
    """
    Maintain EMA gpop on common params, build v_ref by policy_kind, do one-shot projection surgery.
    Input/Output: per-task gradient matrix G [T,P] -> G_new [T,P].
    """

    def __init__(
        self,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        common_param_filter: Callable[[str, torch.nn.Parameter], bool],
        cfg: CommonGpopConfig,
    ):
        self.cfg = cfg
        self.cfg.validate()

        if common_param_filter is None:
            raise ValueError("[gpop] common_param_filter must be provided")

        # Build a flat mask over all params (same flatten order as your grad vector)
        mask_parts: List[torch.Tensor] = []
        common_names: List[str] = []
        for n, p in named_params:
            if not p.requires_grad:
                continue
            is_common = bool(common_param_filter(n, p))
            if is_common:
                common_names.append(n)
            mask_parts.append(torch.full((p.numel(),), is_common, dtype=torch.bool))

        self._common_mask_cpu = (
            torch.cat(mask_parts, dim=0) if len(mask_parts) else torch.empty((0,), dtype=torch.bool)
        )
        self.common_names = common_names

        self.g_common_pop: Optional[torch.Tensor] = None  # EMA base on common params [Pc]

    def common_mask(self, device):
        return self._common_mask_cpu.to(device=device)

    # -------------------------
    # EMA update
    # -------------------------
    @torch.no_grad()
    def _ema_update(self, g_raw_c: torch.Tensor):
        beta = float(self.cfg.ema_beta)
        if self.g_common_pop is None:
            self.g_common_pop = g_raw_c.detach().clone()
        else:
            self.g_common_pop = beta * self.g_common_pop + (1.0 - beta) * g_raw_c.detach()

    # -------------------------
    # Merge + normalize helpers
    # -------------------------
    @torch.no_grad()
    def _merge_from_G(self, G: torch.Tensor) -> torch.Tensor:
        """
        Merge per-task grads into one vector for EMA reference.
        Uses cfg.merge_kind (sum/mean).
        """
        if int(G.shape[0]) == 0:
            return torch.zeros((G.shape[1],), device=G.device, dtype=G.dtype)
        if self.cfg.merge_kind == "mean":
            return G.mean(dim=0)
        return G.sum(dim=0)

    @torch.no_grad()
    def _maybe_task_normalize(self, G: torch.Tensor, cmask: torch.Tensor, eps: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Optional per-task normalization. If task_grad_norm_common_only, compute norms on common dims but scale full vector.
        """
        if not bool(self.cfg.task_grad_norm):
            return G, {}

        if bool(self.cfg.task_grad_norm_common_only):
            Gc = G[:, cmask]
            row_norm = torch.linalg.norm(Gc, dim=1, keepdim=True).clamp_min(eps)
        else:
            row_norm = torch.linalg.norm(G, dim=1, keepdim=True).clamp_min(eps)

        G2 = G / row_norm
        st = {
            "task_norm.enabled": torch.tensor(1.0, device=G.device, dtype=G.dtype),
            "task_norm.common_only": torch.tensor(float(bool(self.cfg.task_grad_norm_common_only)), device=G.device, dtype=G.dtype),
            "task_norm.row_norm.mean": row_norm.mean(),
            "task_norm.row_norm.min": row_norm.min(),
            "task_norm.row_norm.max": row_norm.max(),
        }
        return G2, st

    # -------------------------
    # Cov helpers
    # -------------------------
    @torch.no_grad()
    def _center_and_denom(self, G_common: torch.Tensor) -> Tuple[torch.Tensor, float]:
        X = G_common
        if bool(self.cfg.cov_center):
            X = X - X.mean(dim=0, keepdim=True)
        T = int(X.shape[0])
        denom = float(max(T - 1, 1) if bool(self.cfg.unbiased) else max(T, 1))
        return X, denom

    @torch.no_grad()
    def _cov_mul(self, X: torch.Tensor, v: torch.Tensor, denom: float) -> torch.Tensor:
        # Cov = (X^T X)/denom; Cov v = X^T (X v)/denom
        u = X @ v
        return (X.T @ u) / denom

    @torch.no_grad()
    def _gg_mul(self, G_common: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # (G^T G) v = G^T (G v)
        u = G_common @ v
        return (G_common.T @ u)

    @torch.no_grad()
    def _cov_inv_cg(
        self,
        X: torch.Tensor,
        v: torch.Tensor,
        denom: float,
        damping: float,
        max_iter: int,
        tol: float,
        eps: float,
    ) -> torch.Tensor:
        """
        Solve (Cov + damping * I) x = v with Conjugate Gradient.
        Cov = (X^T X)/denom.
        """
        def A(p: torch.Tensor) -> torch.Tensor:
            return self._cov_mul(X, p, denom) + damping * p

        x = torch.zeros_like(v)
        r = v - A(x)
        p = r.clone()
        rs_old = torch.dot(r, r)

        if float(rs_old) < (tol * tol):
            return x

        for _ in range(int(max_iter)):
            Ap = A(p)
            denom_alpha = torch.dot(p, Ap) + eps
            alpha = rs_old / denom_alpha

            x = x + alpha * p
            r = r - alpha * Ap

            rs_new = torch.dot(r, r)
            if float(rs_new) < (tol * tol):
                break

            beta = rs_new / (rs_old + eps)
            p = r + beta * p
            rs_old = rs_new

        return x

    # -------------------------
    # Main policy
    # -------------------------
    @torch.no_grad()
    def apply(
        self,
        G: torch.Tensor,  # [T,P]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
          - G_new: [T,P] per-task grads after surgery on COMMON dims
          - stats: dict
        """
        if G.dim() != 2:
            raise ValueError(f"[gpop] expected G to be 2D [T,P], got shape {tuple(G.shape)}")

        device = G.device
        eps = float(self.cfg.eps)

        cmask = self.common_mask(device=device)
        if cmask.numel() == 0:
            return G, {}

        # (optional) per-task normalize first
        G_work, st_norm = self._maybe_task_normalize(G, cmask=cmask, eps=eps)

        # Split common
        G_common = G_work[:, cmask]  # [T,Pc]

        # EMA reference from merged grad (computed internally)
        g_merged = self._merge_from_G(G_work)         # [P]
        g_merged_common = g_merged[cmask]             # [Pc]
        g_pop = self.g_common_pop if self.g_common_pop is not None else g_merged_common.detach().clone()

        # ---- Build reference direction v_ref ----
        kind = str(self.cfg.policy_kind).lower()

        if kind == "gg":
            v_ref = self._gg_mul(G_common, g_pop)  # [Pc]
            denom_used = torch.tensor(float(G_common.shape[0]), device=device, dtype=G.dtype)
            centered_used = torch.tensor(0.0, device=device, dtype=G.dtype)
        else:
            X, denom = self._center_and_denom(G_common)
            denom_used = torch.tensor(float(denom), device=device, dtype=G.dtype)
            centered_used = torch.tensor(float(bool(self.cfg.cov_center)), device=device, dtype=G.dtype)

            if kind == "cov_mul":
                v_ref = self._cov_mul(X, g_pop, denom)  # [Pc]
            elif kind == "cov_inv":
                v_ref = self._cov_inv_cg(
                    X,
                    g_pop,
                    denom,
                    damping=float(self.cfg.cov_inv_damping),
                    max_iter=int(self.cfg.cov_inv_max_iter),
                    tol=float(self.cfg.cov_inv_tol),
                    eps=eps,
                )
            else:
                raise ValueError(f"[gpop] Unknown policy_kind: {self.cfg.policy_kind}")

        # ---- Surgery on per-task grads along v_ref ----
        dot = G_common @ v_ref  # [T]
        denom_ref = torch.dot(v_ref, v_ref) + eps

        surgery_mask = dot < 0.0
        applied = surgery_mask.float().mean()

        G_common_new = G_common.clone()
        if surgery_mask.any():
            coeff = (dot[surgery_mask] / denom_ref).unsqueeze(1)  # [n_bad,1]
            G_common_new[surgery_mask] = G_common_new[surgery_mask] - coeff * v_ref.unsqueeze(0)

        # ---- Write back per-task grads (NOT merged) ----
        G_new = G_work.clone()
        G_new[:, cmask] = G_common_new
        new_g_merged_common = self._merge_from_G(G_new)[cmask] # [Pc]
        self._ema_update(new_g_merged_common)

        st = _to_tdict(
            {
                **st_norm,
                "applied": applied,
                "dot.min": dot.min(),
                "dot.mean": dot.mean(),
                "dot.neg_frac": (dot < 0).float().mean(),
                "v_ref_norm": v_ref.norm() + eps,
                "g_pop_norm": g_pop.norm() + eps,
                "merge_kind": 0.0 if self.cfg.merge_kind == "sum" else 1.0,
                "denom_used": denom_used,
                "centered_used": centered_used,
                "damping": float(self.cfg.cov_inv_damping) if kind == "cov_inv" else 0.0,
            },
            device=device,
            dtype=G.dtype,
        )

        return G_new, _prefix(st, f"common_gpop_surgery.{kind}")

    # State I/O
    def state_dict(self):
        return {
            "Gpop_common": None if self.g_common_pop is None else self.g_common_pop.detach().cpu(),
        }

    def load_state_dict(self, st: dict, device=None, dtype=None):
        if st is None:
            self.g_common_pop = None
            return
        gc = st.get("Gpop_common", None)
        if gc is None:
            self.g_common_pop = None
            return
        device = gc.device if device is None else device
        self.g_common_pop = gc.to(device=device, dtype=gc.dtype)