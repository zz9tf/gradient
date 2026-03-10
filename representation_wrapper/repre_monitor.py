import torch
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RepresentationRankMonitorConfig:
    prefix: str = ""
    eps: float = 1e-8

    # covariance / spectrum
    cov_unbiased: bool = True
    mode_k: int = 3

    def validate(self):
        if float(self.eps) <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if int(self.mode_k) < 1:
            raise ValueError(f"mode_k must be >= 1, got {self.mode_k}")


class RepresentationRankMonitor:
    """
    Rank-focused monitor for block representations.

    Input:
        repr_blocks: Dict[str, Tensor]
            each tensor is the representation for one block, e.g.
              [B, D]
              [B, N, D]
              [B, C, H, W]

    Internal reduction:
        X -> [B, F] by flattening all non-batch dimensions.

    Main signals:
      - centered sample-Gram trace
      - effective rank (erank)
      - participation ratio (prank)
      - top eigenvalue ratios
      - top-k / tail energy ratio
      - top-k subspace stability over time

    Note:
      This monitor measures rank structure of block representations
      across the batch/sample dimension, not task-subspace rank.
    """

    def __init__(self, cfg: Optional[RepresentationRankMonitorConfig] = None):
        self.cfg = cfg or RepresentationRankMonitorConfig()
        self.cfg.validate()

        self.prev_block_u: Dict[str, torch.Tensor] = {}
        self.prev_block_u1: Dict[str, torch.Tensor] = {}
        self._step: int = 0

    def reset_state(self):
        self.prev_block_u.clear()
        self.prev_block_u1.clear()
        self._step = 0

    def _apply_prefix(self, stats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p = (self.cfg.prefix or "").strip()
        if p == "":
            return stats
        if not p.endswith("."):
            p = p + "."
        return {p + k: v for k, v in stats.items()}

    def _zero(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_tensor(0.0)

    def _reduce_repr(self, X: torch.Tensor) -> torch.Tensor:
        """
        Reduce representation to [B, F] by flattening
        all non-batch dimensions.
        """
        if X.ndim < 2:
            raise ValueError(f"Representation tensor must have ndim >= 2, got shape={tuple(X.shape)}")
        return X.reshape(X.shape[0], -1)

    def _compute_rank_stats(self, X: torch.Tensor, block_name: str) -> Dict[str, torch.Tensor]:
        """
        X: [B, F]
        Build centered sample-Gram:
            Gram = Xc Xc^T / denom   in R^{B x B}
        and compute spectral rank statistics.
        """
        cfg = self.cfg
        eps = float(cfg.eps)
        B = X.shape[0]
        denom = max(B - 1, 1) if cfg.cov_unbiased else max(B, 1)

        # Center across batch/sample axis
        Xc = X - X.mean(dim=0, keepdim=True)         # [B, F]
        Gram = (Xc @ Xc.T) / float(denom)            # [B, B]

        # Symmetric eigendecomposition, ascending order
        eigvals, eigvecs = torch.linalg.eigh(Gram)
        eigvals = torch.clamp(eigvals, min=0.0)

        trace = eigvals.sum()
        trace_safe = trace + eps

        # probability over spectrum
        p = eigvals / trace_safe

        # effective rank
        ent = -(p * torch.log(p + eps)).sum()
        erank = torch.exp(ent)

        # participation ratio
        lam_sq_sum = (eigvals ** 2).sum()
        prank = (trace ** 2) / (lam_sq_sum + eps)

        # leading ratios
        lambda1_ratio = eigvals[-1] / trace_safe
        lambda2_ratio = eigvals[-2] / trace_safe if eigvals.numel() > 1 else self._zero(trace)

        # top-k / tail ratios
        kk = min(int(cfg.mode_k), eigvals.numel())
        topk_ratio = eigvals[-kk:].sum() / trace_safe
        tail_ratio = (trace - eigvals[-1]) / trace_safe

        stats: Dict[str, torch.Tensor] = {}
        stats[f"{block_name}.B"] = X.new_tensor(float(B))
        stats[f"{block_name}.trace"] = trace
        stats[f"{block_name}.erank"] = erank
        stats[f"{block_name}.prank"] = prank
        stats[f"{block_name}.lambda1_ratio"] = lambda1_ratio
        stats[f"{block_name}.lambda2_ratio"] = lambda2_ratio
        stats[f"{block_name}.top{kk}_ratio"] = topk_ratio
        stats[f"{block_name}.tail_ratio"] = tail_ratio

        # temporal top-k subspace stability
        U_now = eigvecs[:, -kk:]   # [B, kk], top-k eigenspace basis
        u1_now = eigvecs[:, -1]    # [B], top-1 eigvec

        U_prev = self.prev_block_u.get(block_name, None)
        u1_prev = self.prev_block_u1.get(block_name, None)

        if U_prev is None or U_prev.shape != U_now.shape:
            subspace_stab = self._zero(trace)
        else:
            M = U_prev.T @ U_now
            subspace_stab = (M ** 2).sum() / float(kk)

        if u1_prev is None or u1_prev.shape != u1_now.shape:
            u1_stab = self._zero(trace)
        else:
            u1_stab = (u1_prev * u1_now).sum().abs()

        stats[f"{block_name}.subspace_stab"] = subspace_stab
        stats[f"{block_name}.u1_stab"] = u1_stab

        self.prev_block_u[block_name] = U_now.detach()
        self.prev_block_u1[block_name] = u1_now.detach()

        return stats

    def compute_blocks(self, repr_blocks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        stats: Dict[str, torch.Tensor] = {}

        for block, Xraw in repr_blocks.items():
            X = self._reduce_repr(Xraw)  # [B, F]
            stats.update(self._compute_rank_stats(X, block))

        return stats

    def monitor(
        self,
        repr_blocks: Dict[str, torch.Tensor],
        step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            repr_blocks:
                dict of {block_name: tensor}
                each tensor shape can be [B,D], [B,N,D], [B,C,H,W], ...

        Returns:
            Dict[str, Tensor] of rank-oriented monitor stats
        """
        if step is None:
            self._step += 1
        else:
            self._step = int(step)

        if len(repr_blocks) == 0:
            raise ValueError("repr_blocks cannot be empty")

        ref_tensor = next(iter(repr_blocks.values()))
        stats: Dict[str, torch.Tensor] = {
            "monitor.step": ref_tensor.new_tensor(float(self._step))
        }

        stats.update(self.compute_blocks(repr_blocks))
        return self._apply_prefix(stats)