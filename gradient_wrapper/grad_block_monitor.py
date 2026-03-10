import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional


@dataclass
class MonitorConfig:
    prefix: str = ""
    eps: float = 1e-8
    detach: bool = True

    # enable switches
    enable_global: bool = True
    enable_block_energy: bool = True      # norm / eff / dominance
    enable_block_cov: bool = True         # Gram spectrum + erank + u1 stability
    enable_drift: bool = True             # g_mean drift
    enable_block_gpop: bool = True        # gpop stability (rho stats + gpop drift)

    # Gram / Cov
    cov_unbiased: bool = True             # / (T-1) else / T
    cov_mode_k: int = 3                   # top-k modes tracked for stability (k>=1)

    # Gpop
    gpop_beta: float = 0.99
    gpop_update: bool = True
    gpop_warmup_steps: int = 0

    def validate(self):
        if not (0.0 < float(self.gpop_beta) < 1.0):
            raise ValueError(f"gpop_beta must be in (0,1), got {self.gpop_beta}")
        if float(self.eps) <= 0:
            raise ValueError(f"eps must be >0, got {self.eps}")
        if int(self.cov_mode_k) < 1:
            raise ValueError(f"cov_mode_k must be >= 1, got {self.cov_mode_k}")


class GradientMonitor:
    """
    Refactored monitor:
      - NO pairwise cosine stats (saves compute + avoids high-dim cosine pitfalls)
      - Main signals: energy stats, merge efficiency, Gram spectrum, mode stability,
        temporal drift, and gpop stability.

    Inputs:
      - G: [T, P] per-task gradients (flattened param vector)
      - g_ref: [P] reference merged gradient (e.g., raw sum/mean OR final update direction)
              used only for violation fraction (optional but useful).
    """

    def __init__(
        self,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        block_split_fn: Callable[[str], str],
        cfg: Optional[MonitorConfig] = None,
    ):
        self.cfg = cfg or MonitorConfig()
        self.cfg.validate()

        self.param_slices = self._build_param_slices(named_params, block_split_fn)

        # state
        self.prev_block_gmean: Dict[str, torch.Tensor] = {}
        self.prev_block_u: Dict[str, torch.Tensor] = {}       # top-k Gram eigenvectors (T-dim)
        self.block_gpop: Dict[str, torch.Tensor] = {}
        self.prev_block_gpop: Dict[str, torch.Tensor] = {}    # for gpop drift
        self._step: int = 0

    # -------------------
    # helpers
    # -------------------
    def _apply_prefix(self, stats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p = (self.cfg.prefix or "").strip()
        if p == "":
            return stats
        if not p.endswith("."):
            p = p + "."
        return {p + k: v for k, v in stats.items()}

    def _zero(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_tensor(0.0)

    def _build_param_slices(self, named_params, block_split_fn):
        param_slices: Dict[str, List[Tuple[int, int]]] = {}
        offset = 0
        for name, p in named_params:
            if not p.requires_grad:
                continue
            block = block_split_fn(name)
            n = p.numel()
            param_slices.setdefault(block, []).append((offset, offset + n))
            offset += n
        return param_slices

    def _collect_block(self, g: torch.Tensor, block: str) -> torch.Tensor:
        return torch.cat([g[s:e] for s, e in self.param_slices[block]], dim=0)

    def _build_blocks_cache(self, G: torch.Tensor, g_ref: Optional[torch.Tensor]):
        """
        Build per-block tensors once:
          - G_block: [T, Pb]
          - g_ref_block: [Pb] or None
        """
        T = G.shape[0]
        blocks: Dict[str, Dict[str, torch.Tensor]] = {}
        for block in self.param_slices:
            Gb = torch.stack([self._collect_block(G[i], block) for i in range(T)], dim=0)  # [T,Pb]
            out = {"G": Gb}
            if g_ref is not None:
                out["g_ref"] = self._collect_block(g_ref, block)
            blocks[block] = out
        return blocks

    @staticmethod
    def _safe_std(x: torch.Tensor) -> torch.Tensor:
        if x.numel() <= 1:
            return x.new_tensor(0.0)
        return x.std(unbiased=False)

    # -------------------
    # global stats (cheap)
    # -------------------
    def compute_global(self, G: torch.Tensor) -> Dict[str, torch.Tensor]:
        eps = float(self.cfg.eps)
        stats: Dict[str, torch.Tensor] = {}
        T = G.shape[0]

        # per-task norms
        n = G.norm(dim=1)  # [T]
        n_sum = n.sum() + eps

        # merge efficiency (cancellation measure)
        g_sum = G.sum(dim=0)
        eff = g_sum.norm() / n_sum

        stats["global.T"] = G.new_tensor(float(T))
        stats["global.norm_mean"] = n.mean() if n.numel() else self._zero(G)
        stats["global.norm_std"] = self._safe_std(n) if n.numel() else self._zero(G)
        stats["global.norm_cv"] = stats["global.norm_std"] / (stats["global.norm_mean"] + eps)
        stats["global.norm_max_frac"] = (n.max() / n_sum) if n.numel() else self._zero(G)

        stats["global.eff_sum"] = eff
        stats["global.sum_norm"] = n_sum
        stats["global.sum_vec_norm"] = g_sum.norm()

        return stats

    # -------------------
    # block stats (main)
    # -------------------
    def compute_blocks(self, blocks_cache: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        eps = float(cfg.eps)
        k = int(cfg.cov_mode_k)
        stats: Dict[str, torch.Tensor] = {}

        for block, pack in blocks_cache.items():
            Gb: torch.Tensor = pack["G"]  # [T,Pb]
            T = Gb.shape[0]
            denom = max(T - 1, 1) if cfg.cov_unbiased else max(T, 1)

            # -------------------
            # energy / merge efficiency
            # -------------------
            if cfg.enable_block_energy:
                n = Gb.norm(dim=1)                 # [T]
                n_sum = n.sum() + eps
                g_sum = Gb.sum(dim=0)              # [Pb]
                eff = g_sum.norm() / n_sum

                stats[f"{block}.norm_mean"] = n.mean() if n.numel() else self._zero(Gb)
                stats[f"{block}.norm_std"] = self._safe_std(n) if n.numel() else self._zero(Gb)
                stats[f"{block}.norm_cv"] = stats[f"{block}.norm_std"] / (stats[f"{block}.norm_mean"] + eps)
                stats[f"{block}.norm_max_frac"] = (n.max() / n_sum) if n.numel() else self._zero(Gb)

                stats[f"{block}.eff_sum"] = eff
                stats[f"{block}.sum_norm"] = n_sum
                stats[f"{block}.sum_vec_norm"] = g_sum.norm()

                # optional: violation fraction vs g_ref (if provided)
                if "g_ref" in pack:
                    gref = pack["g_ref"]                      # [Pb]
                    dot = Gb @ gref                           # [T]
                    stats[f"{block}.viol_frac"] = (dot < 0).float().mean() if dot.numel() else self._zero(Gb)

            # -------------------
            # Gram spectrum + effective rank + mode stability
            # -------------------
            if cfg.enable_block_cov:
                Gc = Gb - Gb.mean(dim=0, keepdim=True)        # [T,Pb]
                Gram = (Gc @ Gc.T) / float(denom)             # [T,T]

                # eigendecomp (symmetric)
                eigvals, eigvecs = torch.linalg.eigh(Gram)    # ascending
                eigvals = torch.clamp(eigvals, min=0.0)

                trace = eigvals.sum() + eps
                stats[f"{block}.trace"] = trace

                # top eigen ratios
                stats[f"{block}.lambda1_ratio"] = eigvals[-1] / trace
                stats[f"{block}.lambda2_ratio"] = (eigvals[-2] / trace) if eigvals.numel() > 1 else self._zero(trace)

                # effective rank: exp(entropy(p))
                p = eigvals / trace
                # avoid log(0)
                ent = -(p * torch.log(p + eps)).sum()
                stats[f"{block}.erank"] = torch.exp(ent)

                # "condition-ish": lambda1 / mean(lambda)
                mean_lambda = trace / float(max(eigvals.numel(), 1))
                stats[f"{block}.condish"] = eigvals[-1] / (mean_lambda + eps)

                # mode stability: track top-k eigenvectors (T-dim). Use abs dot to ignore sign flips.
                # NOTE: top eigenvectors are last columns in ascending eigvecs.
                kk = min(k, eigvecs.shape[1])
                u_now = eigvecs[:, -kk:]  # [T,kk]

                u_prev = self.prev_block_u.get(block, None)
                if u_prev is None:
                    stats[f"{block}.u1_stab"] = self._zero(trace)
                    if kk > 1:
                        stats[f"{block}.uK_stab_mean"] = self._zero(trace)
                else:
                    # align by absolute dot per mode
                    u_now = eigvecs[:, -kk:]
                    dots = (u_prev * u_now).sum(dim=0).abs()
                    stats[f"{block}.u1_stab"] = dots[-1] if dots.numel() else self._zero(trace)
                    if dots.numel() > 1:
                        stats[f"{block}.uK_stab_mean"] = dots.mean()
                    else:
                        stats[f"{block}.uK_stab_mean"] = self._zero(trace)

                self.prev_block_u[block] = u_now.detach() if cfg.detach else u_now

            # -------------------
            # temporal drift of block mean gradient
            # -------------------
            g_mean = None
            if cfg.enable_drift or cfg.enable_block_gpop:
                g_mean = Gb.mean(dim=0)  # [Pb]

            if cfg.enable_drift:
                if g_mean is None:
                    raise ValueError(f"g_mean is None for block {block} but enable_drift is True")
                prev = self.prev_block_gmean.get(block, None)
                if prev is None:
                    stats[f"{block}.gmean_drift"] = self._zero(g_mean)  
                else:
                    stats[f"{block}.gmean_drift"] = torch.dot(g_mean, prev) / (
                        (g_mean.norm() + eps) * (prev.norm() + eps)
                    )
                self.prev_block_gmean[block] = g_mean.detach() if cfg.detach else g_mean  

            # -------------------
            # gpop stability + task alignment to gpop
            # -------------------
            if cfg.enable_block_gpop:
                if g_mean is None:
                    raise ValueError(f"g_mean is None for block {block} but enable_block_gpop is True")
                beta = float(cfg.gpop_beta)

                if block not in self.block_gpop:
                    self.block_gpop[block] = g_mean.detach().clone()

                gpop = self.block_gpop[block]

                # task-to-gpop rho stats
                gpop_norm = gpop.norm() + eps
                task_norms = Gb.norm(dim=1) + eps
                rho = (Gb @ gpop) / (task_norms * gpop_norm)   # [T]

                stats[f"{block}.gpop_rho_mean"] = rho.mean() if rho.numel() else self._zero(Gb)
                stats[f"{block}.gpop_rho_min"] = rho.min() if rho.numel() else self._zero(Gb)
                stats[f"{block}.gpop_rho_std"] = self._safe_std(rho) if rho.numel() else self._zero(Gb)
                stats[f"{block}.gpop_neg_frac"] = (rho < 0).float().mean() if rho.numel() else self._zero(Gb)

                # gpop drift (stability of EMA itself)
                prev_gpop = self.prev_block_gpop.get(block, None)
                if prev_gpop is None:
                    stats[f"{block}.gpop_drift"] = self._zero(gpop)
                else:
                    stats[f"{block}.gpop_drift"] = torch.dot(gpop, prev_gpop) / (
                        (gpop.norm() + eps) * (prev_gpop.norm() + eps)
                    )
                self.prev_block_gpop[block] = gpop.detach() if cfg.detach else gpop

                # norm ratio: is gpop vanishing or exploding relative to current mean?
                stats[f"{block}.gpop_norm_ratio"] = (gpop.norm() + eps) / (g_mean.norm() + eps)

                # update EMA
                if cfg.gpop_update and (self._step >= int(cfg.gpop_warmup_steps)):
                    new = beta * gpop + (1.0 - beta) * (g_mean.detach() if cfg.detach else g_mean)
                    self.block_gpop[block] = new

        return stats

    # -------------------
    # main
    # -------------------
    def monitor(self, G: torch.Tensor, g_ref: Optional[torch.Tensor] = None, step: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            G: [T,P] per-task grads
            g_ref: [P] optional reference direction for violation fraction (per-block)
                  recommended: raw mean/sum OR final update direction after surgery.
        """
        if step is None:
            self._step += 1
        else:
            self._step = int(step)

        stats: Dict[str, torch.Tensor] = {}
        stats["monitor.step"] = G.new_tensor(float(self._step))

        if self.cfg.enable_global:
            stats.update(self.compute_global(G))

        blocks_cache = self._build_blocks_cache(G, g_ref)

        stats.update(self.compute_blocks(blocks_cache))

        if self.cfg.detach:
            stats = {k: v.detach() for k, v in stats.items()}

        return self._apply_prefix(stats)