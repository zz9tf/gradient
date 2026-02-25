import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple


# ----------------------------- utils -----------------------------

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


# ----------------------------- config -----------------------------

@dataclass
class GradAggConfig:
    mode: str = "pgrs"   # sum | pcgrad | graddrop | pgrs | pgrs_lambda | htdir | pgrs_stage | pgrs_lpf1 | pgrs_common_gate
    eps: float = 1e-12

    # PGRS
    beta: float = 0.999
    tau: float = 0.2

    # heavy-tailed direction sampling (optional mode)
    dir_beta: float = 5.0
    dir_k: float = 2.0
    dir_reject_max: int = 64
    # --- NEW: stage switch ---
    loss_switch: float = 10.0   # overall_loss_raw < loss_switch => late-stage

    def validate(self):
        m = self.mode.lower()
        if m in ("pgrs", "pgrs_stage", "pgrs_lambda", "pgrs_lpf1", "pgrs_common_gate"):
            if not (0.0 <= float(self.tau) <= 1.0):
                raise ValueError(f"[{m}] tau must be in [0,1], got {self.tau}")
            if not (0.0 < float(self.beta) < 1.0):
                raise ValueError(f"[{m}] beta must be in (0,1), got {self.beta}")
            if float(self.loss_switch) <= 0:
                raise ValueError(f"[{m}] loss_switch must be > 0, got {self.loss_switch}")
        if m == "htdir":
            if self.dir_beta <= 0:
                raise ValueError(f"[htdir] dir_beta must be > 0, got {self.dir_beta}")
            if self.dir_k <= 0:
                raise ValueError(f"[htdir] dir_k must be > 0, got {self.dir_k}")


# ----------------------------- aggregator -----------------------------

class GradAggregator:
    """
    Single-space gradient aggregation on all selected params.
    Writes .grad directly (no "fill then overwrite" logic).

    Modes:
      - sum
      - pcgrad
      - graddrop
      - pgrs (GLOBAL EMA Gpop + rho gate)
      - htdir
    """

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

        if common_param_filter is None:
            raise ValueError("You want split management, so common_param_filter must be provided.")

        named_common = [(n, p) for (n, p) in named_all if common_param_filter(n, p)]
        named_priv   = [(n, p) for (n, p) in named_all if not common_param_filter(n, p)]

        if len(named_common) == 0:
            raise ValueError("common_param_filter selected 0 params.")
        if len(named_priv) == 0:
            raise ValueError("private params are empty (common selected everything).")

        self.common_names = [n for n, _ in named_common]
        self.common_params = [p for _, p in named_common]

        self.priv_names = [n for n, _ in named_priv]
        self.priv_params = [p for _, p in named_priv]

        # for global stats only (optional)
        self.all_names = [n for n, _ in named_all]
        self.params = [p for _, p in named_all]
        # ---- NEW: mask over flattened FULL parameter vector for common subset ----
        mask = []
        for (n, p) in named_all:
            # 这里用 common_param_filter 来决定这个 param 属不属于 common
            is_common = bool(common_param_filter(n, p))
            mask.append(torch.full((p.numel(),), is_common, dtype=torch.bool))
        self._common_mask_cpu = torch.cat(mask, dim=0)  # [P] on CPU

        if verbose:
            print("[grad] common tensors:", len(self.common_params), "examples:", self.common_names[:5])
            print("[grad] priv   tensors:", len(self.priv_params),   "examples:", self.priv_names[:5])

        # state
        self.Gpop_common: Optional[torch.Tensor] = None  # only for common space
        self.Gpop: Optional[torch.Tensor] = None         # keep if you still want old modes

        self.last_stats: Dict[str, torch.Tensor] = {}
        self.overall_loss_raw: Optional[torch.Tensor] = None
        self.shrink_ratio: Optional[torch.Tensor] = None
        self.cos_raw_final: Optional[torch.Tensor] = None
        self._is_late = False
        
        # --- NEW: correlation tracking state (per loss key) ---
        self.corr_beta = 0.99  # 你也可以放到 cfg 里
        self._corr_state = {}  # key -> dict of running moments
        self._prev_loss = {}  # key -> last loss scalar

    def _common_mask(self, device):
        return self._common_mask_cpu.to(device=device)
    
    @torch.no_grad()
    def _ema_corr_update(self, key: str, x: torch.Tensor, y: torch.Tensor, beta: float, eps: float):
        """
        Track EMA Pearson correlation between scalar x and y.
        Keeps running mean, var, cov, and returns corr in [-1,1].
        """
        st = self._corr_state.get(key, None)
        if st is None:
            st = {
                "mx": x.detach(),
                "my": y.detach(),
                "vx": torch.zeros_like(x),
                "vy": torch.zeros_like(y),
                "cxy": torch.zeros_like(x),
                "corr": torch.zeros_like(x),
            }

        mx, my = st["mx"], st["my"]
        # update means
        mx_new = beta * mx + (1 - beta) * x
        my_new = beta * my + (1 - beta) * y

        # centered
        dx = x - mx_new
        dy = y - my_new

        # update variances/cov
        vx_new = beta * st["vx"] + (1 - beta) * (dx * dx)
        vy_new = beta * st["vy"] + (1 - beta) * (dy * dy)
        cxy_new = beta * st["cxy"] + (1 - beta) * (dx * dy)

        corr = cxy_new / (vx_new.sqrt() * vy_new.sqrt() + eps)

        st.update({"mx": mx_new, "my": my_new, "vx": vx_new, "vy": vy_new, "cxy": cxy_new, "corr": corr})
        self._corr_state[key] = st
        return corr

    def state_dict(self):
        return {
            "mode": self.cfg.mode,
            "Gpop": None if self.Gpop is None else self.Gpop.detach().cpu(),
            "Gpop_common": None if self.Gpop_common is None else self.Gpop_common.detach().cpu(),  # NEW
        }

    def load_state_dict(self, st, strict: bool = False):
        if st is None:
            self.Gpop = None
            self.Gpop_common = None
            return
        try:
            dev, dt = self.params[0].device, self.params[0].dtype

            g = st.get("Gpop", None)
            self.Gpop = None if g is None else g.to(device=dev, dtype=dt)

            gc = st.get("Gpop_common", None)
            self.Gpop_common = None if gc is None else gc.to(device=dev, dtype=dt)
        except Exception:
            if strict:
                raise
            self.Gpop = None
            self.Gpop_common = None

    # ------------------------- strategies -------------------------

    @torch.no_grad()
    def _pcgrad(self, G: torch.Tensor) -> torch.Tensor:
        eps = float(self.cfg.eps)
        T = G.shape[0]
        Gm = G.clone()
        for i in range(T):
            gi = Gm[i]
            perm = torch.randperm(T, device=G.device)
            for j in perm.tolist():
                if i == j:
                    continue
                gj = Gm[j]  # NOTE: use modified gradients for symmetry
                dot = torch.dot(gi, gj)
                if dot < 0:
                    gi = gi - (dot / (torch.dot(gj, gj) + eps)) * gj
            Gm[i] = gi
        return Gm

    @torch.no_grad()
    def _graddrop(self, G: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        eps = float(self.cfg.eps)
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
    def _sample_ht_direction(self, G: torch.Tensor, w: torch.Tensor):
        eps = float(self.cfg.eps)
        b = float(self.cfg.dir_beta)
        k = float(self.cfg.dir_k)
        max_it = int(self.cfg.dir_reject_max)

        device, dtype = G.device, G.dtype

        norms = G.norm(dim=1) + eps
        Gh = G / norms[:, None]
        a = (w.squeeze(1) * norms)
        Amax = a.sum() + eps

        it_used = 0
        v = None
        last_acc = None
        for it in range(max_it):
            it_used = it + 1
            x = torch.randn(G.shape[1], device=device, dtype=dtype)
            v_try = x / (x.norm() + eps)

            z = Gh @ v_try
            phi = (1.0 + b * (1.0 - z)).pow(-k)
            Aval = torch.dot(a, phi)
            acc = Aval / Amax
            last_acc = acc

            if torch.rand((), device=device, dtype=dtype) < acc:
                v = v_try
                break
            v = v_try

        stats = {
            "ht_iters": torch.tensor(it_used, device=device),
            "ht_acc": last_acc.detach() if last_acc is not None else torch.tensor(0.0, device=device),
        }
        return v, stats

    @torch.no_grad()
    def _update_gpop_common(self, g_for_gpop_common: torch.Tensor) -> torch.Tensor:
        """EMA update for common reference gradient."""
        beta = float(self.cfg.beta)
        if self.Gpop_common is None:
            raise ValueError("Gpop_common is not initialized")
        self.Gpop_common = beta * self.Gpop_common + (1.0 - beta) * g_for_gpop_common.detach()
        return self.Gpop_common

    @torch.no_grad()
    def _update_gpop(self, g_for_gpop: torch.Tensor) -> torch.Tensor:
        """EMA update of population/reference gradient."""
        beta = float(self.cfg.beta)
        if self.Gpop is None:
            raise ValueError("Gpop is not initialized")
        self.Gpop = beta * self.Gpop + (1.0 - beta) * g_for_gpop.detach()
        return self.Gpop

    @torch.no_grad()
    def _rho(self, G: torch.Tensor, Gpop: torch.Tensor) -> torch.Tensor:
        """cosine alignment rho_t between each task gradient and Gpop. returns [T]."""
        eps = float(self.cfg.eps)
        Gpop_norm = Gpop.norm() + eps
        G_norm = G.norm(dim=1) + eps
        return (G @ Gpop) / (G_norm * Gpop_norm)

    @torch.no_grad()
    def _pgrs_projection_surgery(self, G: torch.Tensor, Gpop: torch.Tensor, tau: float):
        """
        Surgery (projection-based, 3-case):
          rho<0: drop
          0<=rho<tau: remove projection onto Gpop
          rho>=tau: keep
        returns (Gprime [T,P], stats)
        """
        eps = float(self.cfg.eps)
        rho = self._rho(G, Gpop)

        den = (Gpop @ Gpop) + eps
        alpha = (G @ Gpop) / den
        proj = alpha[:, None] * Gpop[None, :]

        Gprime = G.clone()
        neg = (rho < 0)
        mid = (rho >= 0) & (rho < tau)

        Gprime[neg] = 0.0
        Gprime[mid] = G[mid] - proj[mid]

        st = {
            "rho_mean": rho.mean(),
            "rho_min": rho.min(),
            "rho_max": rho.max(),
            "kept_frac": (rho >= tau).float().mean(),
            "mid_frac": mid.float().mean(),
            "drop_frac": neg.float().mean(),
        }
        return Gprime, st

    @torch.no_grad()
    def _conf_lambda(self, G: torch.Tensor, Gpop: torch.Tensor):
        """
        lambda/conf = ||Gpop|| / ( (1/T) * sum ||g_i|| )
        returns (lam in [0,1], conf unclamped, mean_g_norm)
        """
        eps = float(self.cfg.eps)
        eps_conf = float(getattr(self.cfg, "eps_conf", eps))

        Gpop_norm = Gpop.norm() + eps
        mean_g_norm = (G.norm(dim=1) + eps).mean()
        conf = Gpop_norm / (mean_g_norm + eps)
        lam = conf.clamp(0.0, 1.0)
        if lam.item() < eps_conf:
            lam = lam.new_tensor(0.0)
        return lam, conf, mean_g_norm

    # ------------------------- main entry -------------------------

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
        gpop_key: Optional[str] = None,
        gpop_use_weight: bool = True,
    ):
        if weights is None:
            weights = {k: 1.0 for k in losses.keys()}

        # clear grads
        for p in self.params:
            p.grad = None

        names = list(losses.keys())
        T = len(names)
        device = next(iter(losses.values())).device
        eps = float(self.cfg.eps)

        # raw objective for logging
        with torch.no_grad():
            self.overall_loss_raw = sum(losses[k].detach() * float(weights.get(k, 1.0)) for k in names)

        # per-task gradients (global space)
        w_list = [float(weights.get(k, 1.0)) for k in names]
        g_list = [gradvec(losses[k], self.params, retain_graph=True) for k in names]
        G = torch.stack(g_list, dim=0)  # [T, P]
        w = torch.tensor(w_list, device=device, dtype=G.dtype).view(T, 1)
        g_raw = (w * G).sum(dim=0)      # [P]

        mode = self.cfg.mode.lower()
        stats: Dict[str, torch.Tensor] = {
            "mode": torch.tensor({"sum":0,"pcgrad":1,"graddrop":2,"pgrs":3,"htdir":4}.get(mode, -1), device=device)
        }

        if mode == "sum":
            g_final = g_raw

        elif mode == "pcgrad":
            Gm = self._pcgrad(G)          # [T,P]
            g_final = (w * Gm).sum(dim=0)

        elif mode == "graddrop":
            g_final, st = self._graddrop(G, w)
            stats.update(st)

        elif mode == "pgrs":
            tau = float(self.cfg.tau)
            # initialize Gpop if not initialized
            if self.Gpop is None:
                self.Gpop = g_raw
            Gprime, st = self._pgrs_projection_surgery(G, self.Gpop, tau=tau)
            g_final = (w * Gprime).sum(dim=0)
            self._update_gpop(g_final)

            stats.update(st)

        elif mode == "htdir":
            g_norm = g_raw.norm() + eps
            if g_norm.item() == 0.0:
                g_final = g_raw
            else:
                v, st = self._sample_ht_direction(G, w)
                g_final = g_norm * v
                stats.update(st)

        elif mode == "pgrs_lambda":
            tau = float(self.cfg.tau)
            # initialize Gpop if not initialized
            if self.Gpop is None:
                self.Gpop = g_raw
            Gpop = self.Gpop

            # 2) compute lambda
            lam, conf, mean_g_norm = self._conf_lambda(G, Gpop)

            # 3) projection surgery branch
            Gprime, st_route = self._pgrs_projection_surgery(G, Gpop, tau=tau)
            g_surgery = (w * Gprime).sum(dim=0)

            # 4) fallback branch: PCGrad
            Gpc = self._pcgrad(G)
            g_pc = (w * Gpc).sum(dim=0)

            # 5) mix on aggregated gradients (NOT per-task mix)
            g_final = lam * g_surgery + (1.0 - lam) * g_pc
            
            Gpop = self._update_gpop(g_final)

            stats.update(st_route)
            stats.update({
                "lambda": lam.detach(),
                "conf": conf.detach(),
                "mean_g_norm": mean_g_norm.detach(),
                "Gpop_norm": (Gpop.norm() + eps).detach(),
                "cos_surgery_pc": torch.dot(g_surgery, g_pc) / ((g_surgery.norm()+eps) * (g_pc.norm()+eps)),
            })

        elif mode == "pgrs_lpf1":
            """
            Single-loss low-pass reference (EMA) + global projection surgery.

            - Pick ONE loss (gpop_key) to generate the reference input x_t.
            - Update Gpop with EMA(x_t) ONLY (this is the low-pass filter).
            - Apply projection surgery to ALL task gradients using the OLD Gpop.
            - Update happens at the END.
            """
            tau = float(self.cfg.tau)
            eps_ = float(self.cfg.eps)

            if gpop_key is None:
                raise ValueError("gpop_key is required for pgrs_lpf1 mode")
            if gpop_key not in losses:
                raise ValueError(f"gpop_key='{gpop_key}' not found in losses keys={list(losses.keys())}")
            idx = names.index(gpop_key)

            # x_t: the ONLY signal that goes into EMA low-pass filter
            # choose raw grad of that ONE loss (optionally weighted)
            x = (w[idx] * G[idx]) if gpop_use_weight else G[idx]  # [P]

            # init Gpop (filter state)
            if self.Gpop is None:
                self.Gpop = x.detach().clone()

            Gpop = self.Gpop

            # global surgery uses the OLD Gpop
            Gprime, st = self._pgrs_projection_surgery(G, Gpop, tau=tau)
            g_final = (w * Gprime).sum(dim=0)

            # update low-pass filter state at END
            Gpop = self._update_gpop(x)

            stats.update(st)
            stats.update({
                "gpop_key_idx": torch.tensor(idx, device=device, dtype=torch.long),
                "lpf_in_norm": (x.norm() + eps_).detach(),
                "Gpop_norm": (Gpop.norm() + eps_).detach(),
            })
        
        elif mode == "pgrs_stage":
            tau = float(self.cfg.tau)

            # init Gpop with raw global gradient
            if self.Gpop is None:
                self.Gpop = g_raw.detach().clone()

            # stage decision
            loss_sw = float(getattr(self.cfg, "loss_switch", 10.0))
            loss_hi = loss_sw * 1.1
            loss_val = float(self.overall_loss_raw.item())

            if not self._is_late and loss_val < loss_sw:
                self._is_late = True
            elif self._is_late and loss_val > loss_hi:
                self._is_late = False
            late = self._is_late

            if late:
                # late-stage: degenerate to "instant surgery projection"
                # (no EMA reference, no lag)
                Gm = self._pcgrad(G)          # [T,P]
                g_final = (w * Gm).sum(dim=0)

                # keep Gpop tracking raw gradient (optional but recommended)
                # avoids stale Gpop if you ever switch back or for logging
                self._update_gpop(g_raw)

                stats.update({
                    "late_stage": torch.tensor(1.0, device=device, dtype=G.dtype),
                    "loss_switch": torch.tensor(loss_sw, device=device, dtype=G.dtype),
                    "pgrs_stage_late_pcgrad": torch.tensor(1.0, device=device, dtype=G.dtype),
                })

            else:
                # early-stage: normal PGRS surgery w/ EMA reference (OLD Gpop)
                Gpop = self.Gpop
                Gprime, st = self._pgrs_projection_surgery(G, Gpop, tau=tau)
                g_final = (w * Gprime).sum(dim=0)

                # IMPORTANT FIX: update EMA with RAW (not g_final) to avoid self-locking
                self._update_gpop(g_raw)

                stats.update(st)
                stats.update({
                    "late_stage": torch.tensor(0.0, device=device, dtype=G.dtype),
                    "loss_switch": torch.tensor(loss_sw, device=device, dtype=G.dtype),
                    "pgrs_stage_late_pcgrad": torch.tensor(0.0, device=device, dtype=G.dtype),
                })
          
        elif mode == "pgrs_common_gate":
            """
            No surgery.

            - Use plain summed gradient for parameter update: g_final = g_raw.
            - Maintain ONLY Gpop_common on common params.
            - Update Gpop_common only when common-subspace rho has NO negatives.
            """
            cmask = self._common_mask(device=G.device)  # [P] bool

            # common subspace grads
            Gc = G[:, cmask]                      # [T, Pc]
            g_raw_c = (w * Gc).sum(dim=0)         # [Pc]

            # init Gpop_common
            if self.Gpop_common is None:
                self.Gpop_common = g_raw_c.detach().clone()

            # rho in common subspace
            rho_c = self._rho(Gc, self.Gpop_common)  # [T]
            # alignment signal (dot, not cosine)
            align = (Gc @ self.Gpop_common).detach()  # [T]
            can_update = (rho_c.min() >= 0)          # strict: any negative -> freeze

            # parameter update uses raw full gradient (no surgery)
            g_final = g_raw

            # gated update of Gpop_common (use RAW common grad to avoid self-locking)
            if can_update.item():
                self._update_gpop_common(g_raw_c)
                
            beta = getattr(self, "corr_beta", 0.99)
            for i, k in enumerate(names):
                li = losses[k].detach()
                prev = self._prev_loss.get(k, None)
                if prev is None:
                    dli = torch.zeros_like(li)
                else:
                    dli = (li - prev)
                self._prev_loss[k] = li

                # corr(Δloss, align)
                ci = self._ema_corr_update(f"dloss_align/{k}", dli, align[i], beta=beta, eps=eps)
                stats[f"corr_dloss_align/{k}"] = ci.detach()

            stats.update({
                "rho_c_mean": rho_c.mean().detach(),
                "rho_c_min": rho_c.min().detach(),
                "rho_c_max": rho_c.max().detach(),
                "align_mean": align.mean().detach(),
                "align_min": align.min().detach(),
                "align_neg_frac": (align < 0).float().mean().detach(),
                "rho_c_neg_frac": (rho_c < 0).float().mean().detach(),
                "gpop_common_updated": can_update.float().detach(),
                "Gpop_common_norm": (self.Gpop_common.norm() + eps).detach(),
                "common_frac_params": cmask.float().mean().detach(),
            })
 
        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")

        # commit grads
        grads_final = unflatten(g_final.detach(), self.params)
        for p, g in zip(self.params, grads_final):
            p.grad = g

        # diagnostics
        with torch.no_grad():
            raw_n = g_raw.norm() + eps
            fin_n = g_final.norm() + eps
            self.shrink_ratio = fin_n / raw_n
            self.cos_raw_final = torch.dot(g_raw, g_final) / (raw_n * fin_n)

        stats.update({
            "overall_loss_raw": self.overall_loss_raw.detach(),
            "shrink_ratio": self.shrink_ratio.detach(),
            "cos_raw_final": self.cos_raw_final.detach(),
        })
        self.last_stats = {k: (v.detach() if torch.is_tensor(v) else torch.tensor(v, device=device))
                           for k, v in stats.items()}
        return self.last_stats