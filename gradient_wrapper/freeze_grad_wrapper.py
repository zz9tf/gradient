import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple


# ----------------------------- utils -----------------------------

def named_params(model: torch.nn.Module,
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


# ----------------------------- configs -----------------------------

@dataclass
class GradAggConfig:
    mode: str = "pgrs"   # sum | pcgrad | graddrop | pgrs | htdir
    eps: float = 1e-12

    # PGRS thresholds
    tau: float = 0.2

    # EMA reference (if fast_update off)
    beta_ref: float = 0.999   # for Gpop_ref
    beta_op: float = 0.999    # for Gpop_op (used for OP surgery direction)

    # fast-update (optional)
    use_fast_update_ref: bool = False
    use_fast_update_op: bool = False
    alpha_fast: float = 0.99
    c_fast: float = 1.0

    # heavy-tailed direction sampling (optional mode)
    dir_beta: float = 5.0
    dir_k: float = 2.0
    dir_reject_max: int = 64

    def validate(self):
        m = self.mode.lower()
        if m in ("pgrs",):
            if not (0.0 <= float(self.tau) <= 1.0):
                raise ValueError(f"[pgrs] tau must be in [0,1], got {self.tau}")
            if not (0.0 < float(self.beta_ref) < 1.0):
                raise ValueError(f"[pgrs] beta_ref must be in (0,1), got {self.beta_ref}")
            if not (0.0 < float(self.beta_op) < 1.0):
                raise ValueError(f"[pgrs] beta_op must be in (0,1), got {self.beta_op}")
        if m == "htdir":
            if self.dir_beta <= 0:
                raise ValueError(f"[htdir] dir_beta must be > 0, got {self.dir_beta}")
            if self.dir_k <= 0:
                raise ValueError(f"[htdir] dir_k must be > 0, got {self.dir_k}")


# ----------------------------- reference updater -----------------------------

class RefUpdater:
    """
    Maintains a reference vector Gpop in some parameter subspace.
    Supports:
      - EMA: G <- beta G + (1-beta) g
      - Fast-update (your Step 2-5)
    """

    def __init__(self, cfg: GradAggConfig, *, which: str):
        self.cfg = cfg
        self.which = which  # "ref" or "op"
        self.G: Optional[torch.Tensor] = None
        self.m: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None  # scalar tensor

    def state_dict(self):
        return {
            "G": None if self.G is None else self.G.detach().cpu(),
            "m": None if self.m is None else self.m.detach().cpu(),
            "v": None if self.v is None else self.v.detach().cpu(),
        }

    def load_state_dict(self, st, device, dtype):
        if st is None:
            self.G = self.m = self.v = None
            return
        G = st.get("G", None)
        m = st.get("m", None)
        v = st.get("v", None)
        self.G = None if G is None else G.to(device=device, dtype=dtype)
        self.m = None if m is None else m.to(device=device, dtype=dtype)
        self.v = None if v is None else v.to(device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, g: torch.Tensor) -> Dict[str, torch.Tensor]:
        eps = float(self.cfg.eps)
        device, dtype = g.device, g.dtype

        use_fast = (self.cfg.use_fast_update_ref if self.which == "ref"
                    else self.cfg.use_fast_update_op)

        if self.G is None:
            self.G = g.detach().clone()
            if use_fast:
                self.m = torch.zeros_like(self.G)
                self.v = torch.zeros((), device=device, dtype=dtype)
            return {
                f"{self.which}_S": torch.zeros((), device=device, dtype=dtype),
                f"{self.which}_beta": torch.ones((), device=device, dtype=dtype),
                f"{self.which}_e_norm": torch.zeros((), device=device, dtype=dtype),
                f"{self.which}_m_norm": torch.zeros((), device=device, dtype=dtype),
                f"{self.which}_v": torch.zeros((), device=device, dtype=dtype),
            } if use_fast else {}

        if not use_fast:
            beta = float(self.cfg.beta_ref if self.which == "ref" else self.cfg.beta_op)
            self.G = beta * self.G + (1.0 - beta) * g.detach()
            return {}

        # fast update
        alpha = float(self.cfg.alpha_fast)
        c = float(self.cfg.c_fast)

        if self.m is None:
            self.m = torch.zeros_like(self.G)
        if self.v is None:
            self.v = torch.zeros((), device=device, dtype=dtype)

        e = g - self.G
        self.m = alpha * self.m + (1.0 - alpha) * e
        e2 = torch.dot(e, e)
        self.v = alpha * self.v + (1.0 - alpha) * e2

        m_norm = self.m.norm()
        S = m_norm / (torch.sqrt(self.v + eps) + eps)
        beta_t = torch.exp(-c * S)  # (0,1]

        self.G = self.G + (1.0 - beta_t) * self.m

        return {
            f"{self.which}_S": S.detach(),
            f"{self.which}_beta": beta_t.detach(),
            f"{self.which}_e_norm": torch.sqrt(e2 + eps).detach(),
            f"{self.which}_m_norm": m_norm.detach(),
            f"{self.which}_v": self.v.detach(),
        }


# ----------------------------- main aggregator -----------------------------

class HybridGradAggregator:
    """
    Clean separation:
      - overall_loss grads fill all params
      - op grads overwritten by strategy aggregation (sum/pcgrad/graddrop/pgrs/htdir)
      - PGRS gating uses REF-space rho, acts on OP-space gradients
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: GradAggConfig,
        keeper_filter: Callable[[str, torch.nn.Parameter], bool],
        ref_filter: Callable[[str, torch.nn.Parameter], bool],
        param_filter: Optional[Callable[[str, torch.nn.Parameter], bool]] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.cfg = cfg
        self.cfg.validate()

        # all considered params
        named_all = named_params(model, only=param_filter)
        self.all_names = [n for n, _ in named_all]
        self.all_params = [p for _, p in named_all]

        # keep/op split (both subsets of all_params)
        self.keep_names, self.keep_params = [], []
        self.op_names, self.op_params = [], []
        for n, p in named_all:
            if bool(keeper_filter(n, p)):
                self.keep_names.append(n)
                self.keep_params.append(p)
            else:
                self.op_names.append(n)
                self.op_params.append(p)

        # reference params (can overlap keep/op; we only use them to compute rho + update Gpop_ref)
        named_ref = named_params(model, only=ref_filter)
        self.ref_names = [n for n, _ in named_ref]
        self.ref_params = [p for _, p in named_ref]

        if len(self.op_params) == 0:
            raise ValueError("op_params is empty. keeper_filter might be selecting everything.")
        if len(self.ref_params) == 0:
            raise ValueError("ref_params is empty. ref_filter selects nothing (but pgrs needs ref).")

        if verbose:
            print("[grad] all tensors:", len(self.all_params))
            print("[grad] keep tensors:", len(self.keep_params), "op tensors:", len(self.op_params), "ref tensors:", len(self.ref_params))
            print("[grad] keep examples:", self.keep_names[:5])
            print("[grad] op examples:", self.op_names[:5])
            print("[grad] ref examples:", self.ref_names[:5])

        # references
        self.ref_updater = RefUpdater(cfg, which="ref")
        self.op_updater = RefUpdater(cfg, which="op")  # for OP surgery direction stability

        # stats
        self.last_stats: Dict[str, torch.Tensor] = {}
        self.overall_loss_raw: Optional[torch.Tensor] = None
        self.op_cos_raw_final: Optional[torch.Tensor] = None
        self.op_shrink_ratio: Optional[torch.Tensor] = None

    def state_dict(self):
        return {
            "mode": self.cfg.mode,
            "ref_updater": self.ref_updater.state_dict(),
            "op_updater": self.op_updater.state_dict(),
        }

    def load_state_dict(self, st, strict: bool = False):
        if st is None:
            self.ref_updater = RefUpdater(self.cfg, which="ref")
            self.op_updater = RefUpdater(self.cfg, which="op")
            return
        try:
            # load ref updater
            dev_ref, dt_ref = self.ref_params[0].device, self.ref_params[0].dtype
            self.ref_updater.load_state_dict(st.get("ref_updater", None), dev_ref, dt_ref)

            # load op updater
            dev_op, dt_op = self.op_params[0].device, self.op_params[0].dtype
            self.op_updater.load_state_dict(st.get("op_updater", None), dev_op, dt_op)
        except Exception:
            if strict:
                raise
            self.ref_updater = RefUpdater(self.cfg, which="ref")
            self.op_updater = RefUpdater(self.cfg, which="op")

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
                gj = G[j]
                dot = torch.dot(gi, gj)
                if dot < 0:
                    gi = gi - (dot / (torch.dot(gj, gj) + eps)) * gj
            Gm[i] = gi
        return Gm.sum(dim=0)

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
        stats = {
            "conflict_frac": conflict.float().mean(),
            "P_mean": P.mean(),
        }
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
    def _pgrs_ref_guided(self, Gref: torch.Tensor, Gop: torch.Tensor, w: torch.Tensor, device):
        """
        PGRS decisions computed in REF space, applied in OP space.
        """
        eps = float(self.cfg.eps)
        tau = float(self.cfg.tau)

        # 1) update reference in REF space
        g_raw_ref = (w.to(Gref.dtype) * Gref).sum(dim=0)
        fast_ref = self.ref_updater.update(g_raw_ref.detach())
        Gpop_ref = self.ref_updater.G
        Gpop_ref_norm = Gpop_ref.norm() + eps

        # 2) rho in REF space
        gref_norm = Gref.norm(dim=1) + eps
        rho = (Gref @ Gpop_ref) / (gref_norm * Gpop_ref_norm)

        harmful = rho < 0
        borderline = (rho >= 0) & (rho < tau)
        aligned = rho >= tau

        # 3) apply to OP grads
        Gm = Gop.clone()
        Gm[harmful] = 0.0

        # surgery direction in OP space: keep stable via op_updater
        g_raw_op = (w.to(Gop.dtype) * Gop).sum(dim=0)
        fast_op = self.op_updater.update(g_raw_op.detach())
        Gpop_op = self.op_updater.G
        denom = (Gpop_op.norm() ** 2) + eps

        if borderline.any():
            alpha = (Gm[borderline] @ Gpop_op) / denom
            Gm[borderline] = Gm[borderline] - alpha[:, None] * Gpop_op[None, :]

        g_final_op = (w.to(Gop.dtype) * Gm).sum(dim=0)

        stats = {
            "rho_mean": rho.mean(),
            "rho_min": rho.min(),
            "rho_max": rho.max(),
            "drop_frac": harmful.float().mean(),
            "surgery_frac": borderline.float().mean(),
            "kept_frac": aligned.float().mean(),
            **fast_ref,
            **fast_op,
        }
        return g_final_op, stats

    # ------------------------- main entry -------------------------

    def backward(self, losses: Dict[str, torch.Tensor], weights: Optional[Dict[str, float]] = None):
        """
        losses: dict task_name -> scalar loss tensor
        weights: dict task_name -> float
        """
        if weights is None:
            weights = {k: 1.0 for k in losses.keys()}

        # clear grads
        for p in self.all_params:
            p.grad = None

        names = list(losses.keys())
        T = len(names)
        device = next(iter(losses.values())).device

        # overall loss for normal grads
        overall_loss = sum(losses[k] * float(weights.get(k, 1.0)) for k in names)

        with torch.no_grad():
            self.overall_loss_raw = sum(losses[k].detach() * float(weights.get(k, 1.0)) for k in names)

        # ---- compute per-task grads for OP space (always needed for strategy) ----
        w_list = [float(weights.get(k, 1.0)) for k in names]
        gop_list = [gradvec(losses[k], self.op_params, retain_graph=True) for k in names]
        Gop = torch.stack(gop_list, dim=0)  # [T, P_op]
        w = torch.tensor(w_list, device=device, dtype=Gop.dtype).view(T, 1)

        mode = self.cfg.mode.lower()
        stats: Dict[str, torch.Tensor] = {"mode": torch.tensor({"sum":0,"pcgrad":1,"graddrop":2,"pgrs":3,"htdir":4}.get(mode, -1), device=device)}

        # ---- strategy aggregation in OP space ----
        if mode == "sum":
            g_final_op = (w * Gop).sum(dim=0)

        elif mode == "pcgrad":
            # PCGrad ignores weights per original paper; if you want weights, apply to G first.
            g_final_op = self._pcgrad(Gop)

        elif mode == "graddrop":
            g_final_op, st = self._graddrop(Gop, w)
            stats.update(st)

        elif mode == "pgrs":
            # compute REF per-task grads ONLY for pgrs
            gref_list = [gradvec(losses[k], self.ref_params, retain_graph=True) for k in names]
            Gref = torch.stack(gref_list, dim=0)  # [T, P_ref]
            g_final_op, st = self._pgrs_ref_guided(Gref, Gop, w, device=device)
            stats.update(st)

        elif mode == "htdir":
            g_raw = (w * Gop).sum(dim=0)
            g_norm = g_raw.norm() + float(self.cfg.eps)
            if g_norm.item() == 0.0:
                g_final_op = g_raw
            else:
                v, st = self._sample_ht_direction(Gop, w)
                g_final_op = g_norm * v
                stats.update(st)

        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")

        # diagnostics in OP space
        with torch.no_grad():
            g_raw_op = (w * Gop).sum(dim=0)
            raw_n = g_raw_op.norm() + float(self.cfg.eps)
            fin_n = g_final_op.norm() + float(self.cfg.eps)
            self.op_shrink_ratio = fin_n / raw_n
            self.op_cos_raw_final = torch.dot(g_raw_op, g_final_op) / (raw_n * fin_n)

        # ---- fill all grads with normal backprop on overall_loss ----
        grads_all = torch.autograd.grad(
            overall_loss, self.all_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        grads_all = safe_grads(grads_all, self.all_params)
        for p, g in zip(self.all_params, grads_all):
            p.grad = g

        # ---- overwrite OP grads with strategy result ----
        grads_op = unflatten(g_final_op.detach(), self.op_params)
        for p, g in zip(self.op_params, grads_op):
            p.grad = g

        # stats
        stats.update({
            "op_shrink_ratio": self.op_shrink_ratio.detach(),
            "op_cos_raw_final": self.op_cos_raw_final.detach(),
            "overall_loss_raw": self.overall_loss_raw.detach(),
        })
        self.last_stats = {k: (v.detach() if torch.is_tensor(v) else torch.tensor(v, device=device)) for k, v in stats.items()}
        return self.last_stats