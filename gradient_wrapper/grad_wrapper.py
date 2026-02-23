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
    mode: str = "pgrs"   # sum | pcgrad | graddrop | pgrs | pgrs_lambda | htdir
    eps: float = 1e-12

    # PGRS
    beta: float = 0.999
    tau: float = 0.2

    # heavy-tailed direction sampling (optional mode)
    dir_beta: float = 5.0
    dir_k: float = 2.0
    dir_reject_max: int = 64

    def validate(self):
        m = self.mode.lower()
        if m == "pgrs":
            if not (0.0 <= float(self.tau) <= 1.0):
                raise ValueError(f"[pgrs] tau must be in [0,1], got {self.tau}")
            if not (0.0 < float(self.beta) < 1.0):
                raise ValueError(f"[pgrs] beta must be in (0,1), got {self.beta}")
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
        verbose: bool = True,
    ):
        self.model = model
        self.cfg = cfg
        self.cfg.validate()

        named_all = named_params(model, only=param_filter)
        self.all_names = [n for n, _ in named_all]
        self.params = [p for _, p in named_all]

        if len(self.params) == 0:
            raise ValueError("No trainable params selected (param_filter might filter everything).")

        if verbose:
            print("[grad] tensors:", len(self.params))
            print("[grad] examples:", self.all_names[:5])

        # PGRS state (EMA of raw global gradient)
        self.Gpop: Optional[torch.Tensor] = None

        # stats
        self.last_stats: Dict[str, torch.Tensor] = {}
        self.overall_loss_raw: Optional[torch.Tensor] = None
        self.shrink_ratio: Optional[torch.Tensor] = None
        self.cos_raw_final: Optional[torch.Tensor] = None

    def state_dict(self):
        return {
            "mode": self.cfg.mode,
            "Gpop": None if self.Gpop is None else self.Gpop.detach().cpu(),
        }

    def load_state_dict(self, st, strict: bool = False):
        if st is None:
            self.Gpop = None
            return
        try:
            g = st.get("Gpop", None)
            if g is None:
                self.Gpop = None
            else:
                dev, dt = self.params[0].device, self.params[0].dtype
                self.Gpop = g.to(device=dev, dtype=dt)
        except Exception:
            if strict:
                raise
            self.Gpop = None

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
    def _update_gpop(self, g_raw: torch.Tensor) -> torch.Tensor:
        """EMA update of population/reference gradient."""
        beta = float(self.cfg.beta)
        if self.Gpop is None:
            self.Gpop = g_raw.detach().clone()
        else:
            self.Gpop = beta * self.Gpop + (1.0 - beta) * g_raw.detach()
        return self.Gpop

    @torch.no_grad()
    def _rho(self, G: torch.Tensor, Gpop: torch.Tensor) -> torch.Tensor:
        """cosine alignment rho_t between each task gradient and Gpop. returns [T]."""
        eps = float(self.cfg.eps)
        Gpop_norm = Gpop.norm() + eps
        G_norm = G.norm(dim=1) + eps
        return (G @ Gpop) / (G_norm * Gpop_norm)

    @torch.no_grad()
    def _pgrs_routing_keep(self, G: torch.Tensor, Gpop: torch.Tensor, tau: float):
        """Routing: keep full g_t if rho_t>=tau else drop. returns (Gm [T,P], stats)."""
        rho = self._rho(G, Gpop)
        keep = (rho >= tau).float()
        Gm = G * keep[:, None]
        st = {
            "rho_mean": rho.mean(),
            "rho_min": rho.min(),
            "rho_max": rho.max(),
            "kept_frac": keep.mean(),
        }
        return Gm, st

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

    def backward(self, losses: Dict[str, torch.Tensor], weights: Optional[Dict[str, float]] = None):
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

            Gpop = self._update_gpop(g_raw)
            Gprime, st = self._pgrs_projection_surgery(G, Gpop, tau=tau)
            g_final = (w * Gprime).sum(dim=0)

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

            # 1) update reference
            Gpop = self._update_gpop(g_raw)

            # 2) compute lambda
            lam, conf, mean_g_norm = self._conf_lambda(G, Gpop)

            # 3) routing branch (your EMA-based complex routing; currently hard-keep routing)
            Grouting, st_route = self._pgrs_routing_keep(G, Gpop, tau=tau)
            g_route = (w * Grouting).sum(dim=0)

            # 4) fallback branch: PCGrad
            Gpc = self._pcgrad(G)
            g_pc = (w * Gpc).sum(dim=0)

            # 5) mix on aggregated gradients (NOT per-task mix)
            g_final = lam * g_route + (1.0 - lam) * g_pc

            stats.update(st_route)
            stats.update({
                "lambda": lam.detach(),
                "conf": conf.detach(),
                "mean_g_norm": mean_g_norm.detach(),
                "Gpop_norm": (Gpop.norm() + eps).detach(),
                "cos_route_pc": torch.dot(g_route, g_pc) / ((g_route.norm()+eps) * (g_pc.norm()+eps)),
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