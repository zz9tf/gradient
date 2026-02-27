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
    mode: str = "pgrs_common_gate"   # sum | pcgrad | graddrop | pgrs_common_gate
    eps: float = 1e-8

    # for common-gate EMA update of Gpop_common
    beta: float = 0.999

    # common gate threshold in cosine space
    common_gate_rho_thr: float = 0.0

    def validate(self):
        m = self.mode.lower()
        if m not in ("sum", "pcgrad", "graddrop", "pgrs_common_gate"):
            raise ValueError(f"Unknown mode: {self.mode}")

        if m == "pgrs_common_gate":
            if not (0.0 < float(self.beta) < 1.0):
                raise ValueError(f"[pgrs_common_gate] beta must be in (0,1), got {self.beta}")
            thr = float(self.common_gate_rho_thr)
            if not (-1.0 <= thr <= 1.0):
                raise ValueError(
                    f"[pgrs_common_gate] common_gate_rho_thr must be in [-1,1], got {thr}"
                )


# ----------------------------- aggregator -----------------------------

class GradAggregator:
    """
    Single-space gradient aggregation on all selected params.
    Writes .grad directly.

    Modes:
      - sum
      - pcgrad
      - graddrop
      - pgrs_common_gate
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

        # full space (for actual optimizer step)
        self.all_names = [n for n, _ in named_all]
        self.params = [p for _, p in named_all]

        # ---- mask over flattened FULL parameter vector for common subset ----
        mask = []
        for (n, p) in named_all:
            is_common = bool(common_param_filter(n, p))
            mask.append(torch.full((p.numel(),), is_common, dtype=torch.bool))
        self._common_mask_cpu = torch.cat(mask, dim=0)  # [P] on CPU

        if verbose:
            print("[grad] common tensors:", len(self.common_params), "examples:", self.common_names[:5])
            print("[grad] priv   tensors:", len(self.priv_params),   "examples:", self.priv_names[:5])

        # state (ONLY common reference)
        self.Gpop_common: Optional[torch.Tensor] = None  # [Pc]

        # logging / diagnostics
        self.last_stats: Dict[str, torch.Tensor] = {}
        self.overall_loss_raw: Optional[torch.Tensor] = None
        self.shrink_ratio: Optional[torch.Tensor] = None
        self.cos_raw_final: Optional[torch.Tensor] = None

        # --- correlation tracking state (per loss key) ---
        self.corr_beta = 0.99
        self._corr_state = {}   # key -> dict of running moments
        self._prev_loss = {}    # key -> last loss scalar

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
        mx_new = beta * mx + (1 - beta) * x
        my_new = beta * my + (1 - beta) * y

        dx = x - mx_new
        dy = y - my_new

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
            "Gpop_common": None if self.Gpop_common is None else self.Gpop_common.detach().cpu(),
        }

    def load_state_dict(self, st, strict: bool = False):
        if st is None:
            self.Gpop_common = None
            return
        try:
            dev, dt = self.params[0].device, self.params[0].dtype
            gc = st.get("Gpop_common", None)
            self.Gpop_common = None if gc is None else gc.to(device=dev, dtype=dt)
        except Exception:
            if strict:
                raise
            self.Gpop_common = None

    # ------------------------- strategies -------------------------

    @torch.no_grad()
    def _pcgrad(self, G: torch.Tensor) -> torch.Tensor:
        eps = float(self.cfg.eps)
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

        return Gm

    @torch.no_grad()
    def _graddrop(self, G: torch.Tensor, w: torch.Tensor):
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
    def _rho(self, G: torch.Tensor, Gpop: torch.Tensor) -> torch.Tensor:
        eps = float(self.cfg.eps)
        Gpop_norm = Gpop.norm() + eps
        G_norm = G.norm(dim=1) + eps
        return (G @ Gpop) / (G_norm * Gpop_norm)

    @torch.no_grad()
    def _update_gpop_common(self, g_for_gpop_common: torch.Tensor) -> torch.Tensor:
        beta = float(self.cfg.beta)
        if self.Gpop_common is None:
            raise ValueError("Gpop_common is not initialized")
        self.Gpop_common = beta * self.Gpop_common + (1.0 - beta) * g_for_gpop_common.detach()
        return self.Gpop_common

    @torch.no_grad()
    def _common_gate(
        self,
        G: torch.Tensor,                   # [T, P]
        w: torch.Tensor,                   # [T, 1]
        losses: Dict[str, torch.Tensor],
        names: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Common-gated update:

        - Update parameters with summed gradient g_raw = sum_t w_t g_t.
        - But if common-subspace rho (vs EMA reference) has negatives (< thr),
          we zero-out common-part gradients for this step.
        - Update EMA reference Gpop_common only when gate passes.
        """
        eps = float(self.cfg.eps)
        device = G.device
        cmask = self._common_mask(device=device)  # [P] bool

        # raw full gradient
        g_raw = (w * G).sum(dim=0)  # [P]

        # common subspace
        Gc = G[:, cmask]            # [T, Pc]
        g_raw_c = (w * Gc).sum(dim=0)  # [Pc]

        # init ref
        if self.Gpop_common is None:
            self.Gpop_common = g_raw_c.detach().clone()

        # gate
        rho_c = self._rho(Gc, self.Gpop_common)          # [T]
        align = (Gc @ self.Gpop_common).detach()         # [T]
        rho_thr = g_raw_c.new_tensor(float(self.cfg.common_gate_rho_thr))
        can_update = (rho_c.min() >= rho_thr)

        g_final = g_raw.clone()
        if not bool(can_update.item()):
            g_final[cmask] = 0.0

        if bool(can_update.item()):
            self._update_gpop_common(g_raw_c)

        # stats
        stats: Dict[str, torch.Tensor] = {}
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
            "rho_c_gate_thr": rho_thr.detach(),
        })

        # corr(Δloss, align)
        beta = float(getattr(self, "corr_beta", 0.99))
        for i, k in enumerate(names):
            li = losses[k].detach()
            prev = self._prev_loss.get(k, None)
            dli = torch.zeros_like(li) if prev is None else (li - prev)
            self._prev_loss[k] = li
            ci = self._ema_corr_update(f"dloss_align/{k}", dli, align[i], beta=beta, eps=eps)
            stats[f"corr_dloss_align/{k}"] = ci.detach()

        # pairwise cosine among task grads in common subspace
        Gc_norm = Gc.norm(dim=1) + eps
        Gc_unit = Gc / Gc_norm[:, None]
        cos_tt = Gc_unit @ Gc_unit.T
        triu = torch.triu(torch.ones_like(cos_tt, dtype=torch.bool), diagonal=1)
        cos_vals = cos_tt[triu]
        stats.update({
            "cos_tt_mean": cos_vals.mean().detach() if cos_vals.numel() else cos_tt.new_tensor(0.0),
            "cos_tt_min":  cos_vals.min().detach()  if cos_vals.numel() else cos_tt.new_tensor(0.0),
            "cos_tt_neg_frac": (cos_vals < 0).float().mean().detach() if cos_vals.numel() else cos_tt.new_tensor(0.0),
        })

        # cosine task vs summed common grad
        g_raw_c_norm = g_raw_c.norm() + eps
        cos_to_rawc = (Gc @ g_raw_c) / (Gc_norm * g_raw_c_norm)
        stats.update({
            "cos_to_rawc_mean": cos_to_rawc.mean().detach(),
            "cos_to_rawc_min":  cos_to_rawc.min().detach(),
            "cos_to_rawc_neg_frac": (cos_to_rawc < 0).float().mean().detach(),
        })

        return g_final, stats

    # ------------------------- main entry -------------------------

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
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
        g_list = []
        for i, k in enumerate(names):
            rg = (i != len(names) - 1)
            g_list.append(gradvec(losses[k], self.params, retain_graph=rg))
        G = torch.stack(g_list, dim=0)  # [T, P]
        w = torch.tensor(w_list, device=device, dtype=G.dtype).view(T, 1)
        g_raw = (w * G).sum(dim=0)      # [P]

        mode = self.cfg.mode.lower()
        stats: Dict[str, torch.Tensor] = {
            "mode": torch.tensor({"sum": 0, "pcgrad": 1, "graddrop": 2, "pgrs_common_gate": 3}.get(mode, -1),
                                 device=device)
        }

        if mode == "sum":
            g_final = g_raw

        elif mode == "pcgrad":
            Gm = self._pcgrad(G)
            g_final = (w * Gm).sum(dim=0)

        elif mode == "graddrop":
            g_final, st = self._graddrop(G, w)
            stats.update(st)

        elif mode == "pgrs_common_gate":
            g_final, st = self._common_gate(G, w, losses=losses, names=names)
            stats.update(st)

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

        self.last_stats = {
            k: (v.detach() if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in stats.items()
        }
        return self.last_stats