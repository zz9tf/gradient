import torch
from dataclasses import dataclass
from typing import Dict, List, Optional


def _params_from(model, only: Optional[callable] = None):
    ps = []
    for n, p in model.named_parameters():
        if p.requires_grad and (only is None or only(n, p)):
            ps.append(p)
    return ps


def _flatten(grads: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def _unflatten(vec: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    out, off = [], 0
    for p in params:
        n = p.numel()
        out.append(vec[off : off + n].view_as(p))
        off += n
    return out


def _safe_grads(grads, params):
    return [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]


@dataclass
class GradAggConfig:
    mode: str = "pgrs"          # "sum" | "pcgrad" | "graddrop" | "pgrs"
    beta: float = 0.999         # PGRS EMA beta
    tau: float = 0.2            # PGRS alignment threshold
    eps: float = 1e-12
    graddrop_p: float = 0.5     # (optional) reserved


class GradAggregator:
    """
    agg = GradAggregator(model, GradAggConfig(mode="pgrs"))
    agg.backward(losses, weights)   # writes .grad
    optimizer.step()
    """

    def __init__(self, model: torch.nn.Module, config: GradAggConfig,
                 param_filter: Optional[callable] = None):
        self.model = model
        self.cfg = config
        self.params = _params_from(model, only=param_filter)

        # PGRS state
        self.Gpop: Optional[torch.Tensor] = None  # EMA of raw gradient (flattened)

        # logging/cache
        self.g_final: Optional[torch.Tensor] = None
        self.global_grad_norm: Optional[torch.Tensor] = None
        self.overall_loss_raw: Optional[torch.Tensor] = None   # Σ w_i L_i (detached)
        self.overall_obj: Optional[torch.Tensor] = None        # g_raw · g_final (detached)
        self.shrink_ratio: Optional[torch.Tensor] = None       # ||g_final|| / ||g_raw||
        self.last_stats: Dict[str, torch.Tensor] = {}

    def _per_loss_gradvec(self, loss: torch.Tensor, retain_graph: bool) -> torch.Tensor:
        grads = torch.autograd.grad(
            loss, self.params,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
        )
        grads = _safe_grads(grads, self.params)
        return _flatten(grads)

    @torch.no_grad()
    def _commit(self, g_final: torch.Tensor, g_raw: torch.Tensor,
                stats: Optional[Dict[str, torch.Tensor]] = None):
        # cache for logging
        self.g_final = g_final.detach()
        self.global_grad_norm = self.g_final.norm()
        raw_norm = g_raw.detach().norm()
        self.shrink_ratio = self.global_grad_norm / (raw_norm + self.cfg.eps)
        self.overall_obj = torch.dot(g_raw.detach(), self.g_final)
        self.last_stats = {} if stats is None else {k: v.detach() for k, v in stats.items()}

        # write gradients into parameters
        grads_final = _unflatten(self.g_final, self.params)
        for p, g in zip(self.params, grads_final):
            p.grad = g

    def backward(self, losses: Dict[str, torch.Tensor],
                 weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        if weights is None:
            weights = {k: 1.0 for k in losses.keys()}

        # clear old grads
        for p in self.params:
            p.grad = None

        names = list(losses.keys())
        T = len(names)
        device = next(iter(losses.values())).device

        # ---- raw overall loss (true objective value, for logging) ----
        with torch.no_grad():
            self.overall_loss_raw = sum(losses[k].detach() * float(weights.get(k, 1.0)) for k in names)

        # ---- per-loss gradients ----
        g_list, w_list = [], []
        for i, name in enumerate(names):
            g_list.append(self._per_loss_gradvec(losses[name], retain_graph=(i != T - 1)))
            w_list.append(float(weights.get(name, 1.0)))

        G = torch.stack(g_list, dim=0)                                  # [T, P]
        w = torch.tensor(w_list, device=device, dtype=G.dtype).view(T, 1)  # [T,1]
        g_raw = (w * G).sum(dim=0)                                      # [P]

        mode = self.cfg.mode.lower()

        # ---- SUM (baseline) ----
        if mode == "sum":
            self._commit(g_raw, g_raw, stats={"mode": torch.tensor(0, device=device)})
            return self.last_stats

        # ---- PCGrad ----
        if mode == "pcgrad":
            Gm = G.clone()
            for i in range(T):
                gi = Gm[i]
                for j in range(T):
                    if i == j:
                        continue
                    gj = Gm[j]
                    dot = torch.dot(gi, gj)
                    if dot < 0:
                        gi = gi - (dot / (torch.dot(gj, gj) + self.cfg.eps)) * gj
                Gm[i] = gi
            g_final = (w * Gm).sum(dim=0)
            self._commit(g_final, g_raw, stats={"mode": torch.tensor(1, device=device)})
            return self.last_stats

        # ---- GradDrop (simple sign-based drop) ----
        if mode == "graddrop":
            Gw = w * G
            signs = torch.sign(Gw)
            pos = (signs > 0).float().mean(dim=0)
            neg = (signs < 0).float().mean(dim=0)
            conflict = (pos > 0) & (neg > 0)

            prob_pos = pos / (pos + neg + self.cfg.eps)
            target_pos = (torch.rand_like(prob_pos) < prob_pos)

            mask = torch.ones_like(Gw)
            for t in range(T):
                st = signs[t]
                drop = conflict & (((st > 0) & (~target_pos)) | ((st < 0) & (target_pos)))
                mask[t, drop] = 0.0

            g_final = (Gw * mask).sum(dim=0)
            self._commit(g_final, g_raw, stats={
                "mode": torch.tensor(2, device=device),
                "conflict_frac": conflict.float().mean(),
            })
            return self.last_stats

        # ---- PGRS (task-level routing by alignment to EMA population gradient) ----
        if mode == "pgrs":
            # update EMA population direction with raw gradient
            if self.Gpop is None:
                self.Gpop = g_raw.detach()
            else:
                beta = self.cfg.beta
                self.Gpop = beta * self.Gpop + (1.0 - beta) * g_raw.detach()

            Gpop = self.Gpop
            Gpop_norm = Gpop.norm() + self.cfg.eps
            g_norm = G.norm(dim=1) + self.cfg.eps
            rho = (G @ Gpop) / (g_norm * Gpop_norm)   # [T]

            keep = (rho >= self.cfg.tau).float()      # [T]
            Gm = G * keep[:, None]
            g_final = (w * Gm).sum(dim=0)

            self._commit(g_final, g_raw, stats={
                "mode": torch.tensor(3, device=device),
                "rho_mean": rho.mean(),
                "rho_min": rho.min(),
                "rho_max": rho.max(),
                "kept_frac": keep.mean(),
            })
            return self.last_stats

        raise ValueError(f"Unknown mode: {self.cfg.mode}")
