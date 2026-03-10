import torch
from dataclasses import dataclass


@dataclass
class GradRPCorrConfig:
    enabled: bool = False
    eps: float = 1e-8
    weight_mode: str = "inv_sqrt"
    clamp_eig_min: float = 0.0
    low_rank_k: int = 0
    detach_repr: bool = True


def _flatten_batch_first(x: torch.Tensor):
    B = x.shape[0]
    return x.reshape(B, -1), x.shape


def _compute_S(eigvals, cfg):
    eps = cfg.eps
    lam = torch.clamp(eigvals, min=cfg.clamp_eig_min)

    if cfg.weight_mode == "inv":
        S = 1.0 / ((lam + eps) ** 2)

    elif cfg.weight_mode == "inv_sqrt":
        S = 1.0 / ((lam + eps) ** 1.5)

    elif cfg.weight_mode == "flat":
        S = 1.0 / (lam + eps)

    elif cfg.weight_mode == "log_inv":
        S = 1.0 / ((lam + eps) * torch.log1p(lam + eps))

    else:
        raise ValueError(cfg.weight_mode)

    return S


def _build_context(h, cfg):
    x, _ = _flatten_batch_first(h)

    if cfg.detach_repr:
        x = x.detach()

    B = x.shape[0]

    xc = x - x.mean(dim=0, keepdim=True)

    denom = max(B - 1, 1)
    K = (xc @ xc.T) / denom
    K = 0.5 * (K + K.T)

    eigvals, eigvecs = torch.linalg.eigh(K)

    eigvals = torch.clamp(eigvals, min=cfg.clamp_eig_min)

    if cfg.low_rank_k > 0 and cfg.low_rank_k < eigvals.numel():
        eigvals = eigvals[-cfg.low_rank_k:]
        eigvecs = eigvecs[:, -cfg.low_rank_k:]

    S = _compute_S(eigvals, cfg)

    return xc, eigvecs, S, B


def _apply_filter(grad, xc, V, S, B):

    g2, orig_shape = _flatten_batch_first(grad)

    A = g2 @ xc.T
    Bm = A @ V
    Cm = Bm * S.unsqueeze(0)
    Dm = Cm @ V.T

    denom = max(B - 1, 1)

    g_filtered = (Dm @ xc) / denom

    return g_filtered.reshape(orig_shape)


def make_repr_grad_hook(h, cfg: GradRPCorrConfig):

    if not cfg.enabled:
        return None

    xc, V, S, B = _build_context(h, cfg)

    def _hook(grad):
        return _apply_filter(
            grad,
            xc,
            V,
            S,
            B,
        )

    return _hook