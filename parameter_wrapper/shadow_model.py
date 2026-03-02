"""
parameter_wrapper/shadow_model.py

Shadow model (EMA) in PARAMETER space.

Unified stats naming:
- raw_step.*        : theta_t vs theta_{t-1}
- gap_to_shadow.*   : theta_t vs shadow_t
- shadow_step.*     : shadow_t vs shadow_{t-1}
- eval/raw/*, eval/shadow/* : user-defined metrics via eval_fn

Each block returns the SAME canonical keys:
  step_l2, theta_l2, rel            (for steps)
  gap_l2,  theta_l2, rel            (for gaps)
Plus optional grouped rel/<group> via group_fn(name)->str (e.g., "common"/"private").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple, Callable

import math
import torch
from torch import nn

from collections import OrderedDict
import torch.nn.functional as F

# -------------------------
# helpers
# -------------------------

def _iter_tensors(module: nn.Module, include_buffers: bool) -> Iterator[Tuple[str, torch.Tensor]]:
    for n, p in module.named_parameters(recurse=True):
        if p is None:
            continue
        yield n, p
    if include_buffers:
        for n, b in module.named_buffers(recurse=True):
            if b is None:
                continue
            yield f"__buffer__.{n}", b


def _is_float_tensor(_: str, t: torch.Tensor) -> bool:
    return torch.is_floating_point(t)


def _l2_sq(t: torch.Tensor) -> torch.Tensor:
    return torch.sum(t * t)


def _prefix_keys(d: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}.{k}": float(v) for k, v in d.items()}


def _shadow_group_fn(name: str) -> str:
    name = name[7:] if name.startswith("module.") else name
    return "private" if name.startswith("decoder.") else "common"


def get_or_create_shadow_ema(run_info, model):
    """
    Ensure a single ShadowEMA instance per run.
    """
    shadow_ema_cfg = run_info["net"]["extra_info"].get("shadow_ema_cfg")
    if shadow_ema_cfg is None:
        return None, None

    shadow_ema = run_info["net"].get("shadow_ema")
    if shadow_ema is None:
        shadow_ema = ShadowEMA(model, group_fn=_shadow_group_fn, **shadow_ema_cfg)
        run_info["net"]["shadow_ema"] = shadow_ema
    return shadow_ema, shadow_ema_cfg


def make_or_get_shadow_train_eval_fn(run_info):
    """
    Create once, reuse forever. 只缓存“函数壳子”，具体 batch 的 imgs/true 等仍在调用时传入。
    """
    extra = run_info["net"]["extra_info"]
    fn = extra.get("_shadow_train_eval_fn", None)
    if fn is not None:
        return fn

    def _eval_fn(m, *, imgs, true_dict, true_np_onehot, loss_opts, loss_func_dict):
        was_training_mode = bool(m.training)
        m.eval()
        with torch.no_grad():
            eval_pred = m(imgs)
            eval_pred = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in eval_pred.items()]
            )
            eval_pred["np"] = F.softmax(eval_pred["np"], dim=-1)
            if m.module.nr_types is not None:
                eval_pred["tp"] = F.softmax(eval_pred["tp"], dim=-1)

            eval_loss_branch = {}
            for branch_name in eval_pred.keys():
                L = 0.0
                for loss_name, loss_weight in loss_opts[branch_name].items():
                    loss_func = loss_func_dict[loss_name]
                    loss_args = [true_dict[branch_name], eval_pred[branch_name]]
                    if loss_name == "msge":
                        loss_args.append(true_np_onehot[..., 1])
                    term_loss = loss_func(*loss_args)
                    L = L + float(loss_weight) * term_loss
                eval_loss_branch[branch_name] = L
            eval_total_loss = sum(v for v in eval_loss_branch.values())

        if was_training_mode:
            m.train()

        out = {"overall_loss": float(eval_total_loss.detach().item())}
        for branch_name, branch_loss in eval_loss_branch.items():
            out[f"{branch_name}_loss"] = float(branch_loss.detach().item())
        return out

    extra["_shadow_train_eval_fn"] = _eval_fn
    return _eval_fn


def forward_pred_dict(model, imgs_nchw, *, for_loss: bool):
    """
    Returns pred_dict in HWC:
    - for_loss=True  : keep np/tp as prob distribution (softmax over last dim)
    - for_loss=False : return np as prob of positive class (..., 1) and tp as argmax map (float32)
    """
    pred_dict = model(imgs_nchw)
    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    )

    # np always exists
    if for_loss:
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
    else:
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1]

    # tp optional
    if getattr(model.module, "nr_types", None) is not None and "tp" in pred_dict:
        if for_loss:
            pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)
        else:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=False).type(torch.float32)
            pred_dict["tp"] = type_map

    return pred_dict

# -------------------------
# config
# -------------------------

@dataclass
class ShadowConfig:
    beta: float = 0.999
    eps: float = 1e-8
    include_buffers: bool = False


# -------------------------
# main
# -------------------------

class ShadowEMA:
    """
    EMA ("shadow") in PARAMETER space with unified logging.

    Canonical blocks:
      - raw_step:        live theta step size
      - gap_to_shadow:   live-vs-shadow distance
      - shadow_step:     shadow EMA step size
      - eval/raw/* and eval/shadow/*: user metrics

    Recommended loop:
        shadow = ShadowEMA(model, beta=0.999, group_fn=...)

        for step, batch in enumerate(loader):
            prev_raw = shadow.snapshot_raw(model)
            prev_sh  = shadow.snapshot_shadow()   # BEFORE shadow.update

            loss = ...
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            shadow.update(model)                  # AFTER step

            stats = shadow.on_step_end(
                model,
                prev_raw_snap=prev_raw,
                prev_shadow_snap=prev_sh,
                step=step,
                eval_fn=eval_fn,
                eval_every=200,
            )
    """

    def __init__(
        self,
        model: nn.Module,
        beta: float = 0.999,
        *,
        eps: float = 1e-8,
        include_buffers: bool = False,
        param_filter: Callable[[str, torch.Tensor], bool] = _is_float_tensor,
        shadow_device: Optional[torch.device] = None,
        group_fn: Optional[Callable[[str], str]] = None,   # e.g. name -> "common"/"private"
    ):
        if not (0.0 < float(beta) < 1.0):
            raise ValueError(f"beta must be in (0,1), got {beta}")

        self.cfg = ShadowConfig(beta=float(beta), eps=float(eps), include_buffers=bool(include_buffers))
        self.param_filter = param_filter
        self.shadow_device = shadow_device

        self.shadow: Dict[str, torch.Tensor] = {}
        self.group_fn = group_fn
        self._init_from(model)

    def _init_from(self, model: nn.Module) -> None:
        self.shadow.clear()
        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            s = t.detach().clone()
            if self.shadow_device is not None:
                s = s.to(self.shadow_device)
            self.shadow[name] = s

    def _group(self, name: str) -> str:
        return self.group_fn(name) if self.group_fn is not None else "all"

    # -------------------------
    # EMA update + snapshots
    # -------------------------

    @torch.no_grad()
    def update(self, model: nn.Module, *, beta: Optional[float] = None) -> None:
        b = float(self.cfg.beta if beta is None else beta)
        if not (0.0 < b < 1.0):
            raise ValueError(f"beta must be in (0,1), got {b}")

        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue

            if name not in self.shadow:
                s = t.detach().clone()
                if self.shadow_device is not None:
                    s = s.to(self.shadow_device)
                self.shadow[name] = s
                continue

            live = t.detach()
            if self.shadow_device is not None:
                live = live.to(self.shadow_device)

            self.shadow[name].mul_(b).add_(live, alpha=(1.0 - b))

    @torch.no_grad()
    def snapshot_raw(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Detached copy of current live parameters (and optional buffers) on their native devices."""
        snap: Dict[str, torch.Tensor] = {}
        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            snap[name] = t.detach().clone()
        return snap

    @torch.no_grad()
    def snapshot_shadow(self) -> Dict[str, torch.Tensor]:
        """Detached copy of current shadow tensors (on shadow_device if set)."""
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    # -------------------------
    # canonical stats blocks
    # -------------------------
    @torch.no_grad()
    def raw_step(self, prev_raw: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, float]:
        """
        raw step: theta_now vs theta_prev
          step_l2 = ||theta_now - theta_prev||
          theta_l2 = ||theta_now||
          rel = step_l2 / (theta_l2 + eps)
        plus rel/<group>
        """
        step_sq = 0.0
        theta_sq = 0.0
        step_sq_g: Dict[str, float] = {}
        theta_sq_g: Dict[str, float] = {}

        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            if name not in prev_raw:
                continue

            prev = prev_raw[name].to(t.device)
            diff = (t.detach() - prev).float()

            dsq = float(_l2_sq(diff).item())
            tsq = float(_l2_sq(t.detach().float()).item())

            step_sq += dsq
            theta_sq += tsq

            g = self._group(name)
            step_sq_g[g] = step_sq_g.get(g, 0.0) + dsq
            theta_sq_g[g] = theta_sq_g.get(g, 0.0) + tsq

        step_l2 = math.sqrt(step_sq + self.cfg.eps)
        theta_l2 = math.sqrt(theta_sq + self.cfg.eps)
        rel = step_l2 / (theta_l2 + self.cfg.eps)

        out: Dict[str, float] = {"step_l2": step_l2, "theta_l2": theta_l2, "rel": rel}
        for g, ssq in step_sq_g.items():
            sl2 = math.sqrt(ssq + self.cfg.eps)
            tl2 = math.sqrt(theta_sq_g.get(g, 0.0) + self.cfg.eps)
            out[f"rel/{g}"] = sl2 / (tl2 + self.cfg.eps)
        return out

    @torch.no_grad()
    def gap_to_shadow(self, model: nn.Module) -> Dict[str, float]:
        """
        gap: theta_now vs shadow_now
          gap_l2 = ||theta - shadow||
          theta_l2 = ||theta||
          rel = gap_l2 / (theta_l2 + eps)
        plus rel/<group>
        """
        gap_sq = 0.0
        theta_sq = 0.0
        gap_sq_g: Dict[str, float] = {}
        theta_sq_g: Dict[str, float] = {}

        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            if name not in self.shadow:
                continue

            s = self.shadow[name]
            if s.device != t.device:
                s = s.to(t.device)

            diff = (t.detach() - s).float()
            gsq = float(_l2_sq(diff).item())
            tsq = float(_l2_sq(t.detach().float()).item())

            gap_sq += gsq
            theta_sq += tsq

            g = self._group(name)
            gap_sq_g[g] = gap_sq_g.get(g, 0.0) + gsq
            theta_sq_g[g] = theta_sq_g.get(g, 0.0) + tsq

        gap_l2 = math.sqrt(gap_sq + self.cfg.eps)
        theta_l2 = math.sqrt(theta_sq + self.cfg.eps)
        rel = gap_l2 / (theta_l2 + self.cfg.eps)

        out: Dict[str, float] = {"gap_l2": gap_l2, "theta_l2": theta_l2, "rel": rel}
        for g, gsq in gap_sq_g.items():
            gl2 = math.sqrt(gsq + self.cfg.eps)
            tl2 = math.sqrt(theta_sq_g.get(g, 0.0) + self.cfg.eps)
            out[f"rel/{g}"] = gl2 / (tl2 + self.cfg.eps)
        return out

    @torch.no_grad()
    def shadow_step(self, prev_shadow: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        shadow step: shadow_now vs shadow_prev
          step_l2 = ||s_now - s_prev||
          theta_l2 = ||s_now||
          rel = step_l2 / (theta_l2 + eps)
        plus rel/<group>
        """
        step_sq = 0.0
        theta_sq = 0.0
        step_sq_g: Dict[str, float] = {}
        theta_sq_g: Dict[str, float] = {}

        for name, s_now in self.shadow.items():
            if name not in prev_shadow:
                continue
            s_prev = prev_shadow[name]
            if s_prev.device != s_now.device:
                s_prev = s_prev.to(s_now.device)

            diff = (s_now.detach() - s_prev.detach()).float()
            dsq = float(_l2_sq(diff).item())
            tsq = float(_l2_sq(s_now.detach().float()).item())

            step_sq += dsq
            theta_sq += tsq

            g = self._group(name)
            step_sq_g[g] = step_sq_g.get(g, 0.0) + dsq
            theta_sq_g[g] = theta_sq_g.get(g, 0.0) + tsq

        step_l2 = math.sqrt(step_sq + self.cfg.eps)
        theta_l2 = math.sqrt(theta_sq + self.cfg.eps)
        rel = step_l2 / (theta_l2 + self.cfg.eps)

        out: Dict[str, float] = {"step_l2": step_l2, "theta_l2": theta_l2, "rel": rel}
        for g, ssq in step_sq_g.items():
            sl2 = math.sqrt(ssq + self.cfg.eps)
            tl2 = math.sqrt(theta_sq_g.get(g, 0.0) + self.cfg.eps)
            out[f"rel/{g}"] = sl2 / (tl2 + self.cfg.eps)
        return out

    # -------------------------
    # eval convenience
    # -------------------------

    @torch.no_grad()
    def copy_shadow_to_raw(self, model: nn.Module) -> None:
        """Overwrite live params with shadow (useful for eval-time EMA weights)."""
        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            if name not in self.shadow:
                continue
            s = self.shadow[name]
            if s.device != t.device:
                s = s.to(t.device)
            t.copy_(s)

    @torch.no_grad()
    def eval_raw_and_shadow(
        self,
        model: nn.Module,
        eval_fn: Callable[[nn.Module], Dict[str, float]],
        *,
        prefix_raw: str = "eval/raw",
        prefix_shadow: str = "eval/shadow",
    ) -> Dict[str, float]:
        """
        Runs eval_fn on:
          1) current live model
          2) model with shadow weights temporarily copied in
        Restores weights afterwards.

        NOTE: eval_fn decides keys (e.g., {"acc":..., "loss":...}).
        """
        out: Dict[str, float] = {}

        # raw
        r_raw = eval_fn(model) or {}
        for k, v in r_raw.items():
            out[f"{prefix_raw}/{k}"] = float(v)

        # backup live
        backup = self.snapshot_raw(model)

        # shadow
        self.copy_shadow_to_raw(model)
        r_sh = eval_fn(model) or {}
        for k, v in r_sh.items():
            out[f"{prefix_shadow}/{k}"] = float(v)

        # restore
        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            if name in backup:
                t.copy_(backup[name].to(t.device))

        return out

    # -------------------------
    # actions (parameter space)
    # -------------------------

    @torch.no_grad()
    def pull_to_shadow(self, model: nn.Module, *, lam: float = 0.05) -> None:
        """theta <- (1-lam)*theta + lam*shadow"""
        lam = float(lam)
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"lam must be in [0,1], got {lam}")

        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            if name not in self.shadow:
                continue
            s = self.shadow[name]
            if s.device != t.device:
                s = s.to(t.device)
            t.mul_(1.0 - lam).add_(s, alpha=lam)

    @torch.no_grad()
    def correct_toward_shadow(self, model: nn.Module, *, lr: float, lam: float = 1.0) -> Dict[str, float]:
        """
        Small normalized correction step toward shadow:
            d = shadow - theta
            theta <- theta + lr*lam * d / (||d|| + eps)

        Returns: {"d_l2":..., "applied": 0/1}
        """
        lr = float(lr)
        lam = float(lam)
        if lr <= 0 or lam <= 0:
            return {"d_l2": 0.0, "applied": 0.0}

        d_sq = 0.0
        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            if name not in self.shadow:
                continue
            s = self.shadow[name]
            if s.device != t.device:
                s = s.to(t.device)
            d = (s - t.detach()).float()
            d_sq += float(_l2_sq(d).item())

        d_l2 = math.sqrt(d_sq + self.cfg.eps)
        if d_l2 <= 0:
            return {"d_l2": 0.0, "applied": 0.0}

        scale = (lr * lam) / (d_l2 + self.cfg.eps)

        for name, t in _iter_tensors(model, self.cfg.include_buffers):
            if not self.param_filter(name, t):
                continue
            if name not in self.shadow:
                continue
            s = self.shadow[name]
            if s.device != t.device:
                s = s.to(t.device)
            d = (s - t).float()
            t.add_(d, alpha=scale)

        return {"d_l2": d_l2, "applied": 1.0}

    # -------------------------
    # on_step_end (unified)
    # -------------------------

    @torch.no_grad()
    def on_step_end(
        self,
        model: nn.Module,
        *,
        prev_raw_snap: Optional[Dict[str, torch.Tensor]] = None,
        prev_shadow_snap: Optional[Dict[str, torch.Tensor]] = None,
        do_gap: bool = True,
        do_raw_step: bool = True,
        do_shadow_step: bool = True,
        eval_fn: Optional[Callable[..., Dict[str, float]]] = None,
        step: Optional[int] = None,
        eval_every: Optional[int] = 1,
        eval_kwargs: Optional[dict] = None,
    ) -> Dict[str, float]:
        stats: Dict[str, float] = {}

        if do_raw_step and (prev_raw_snap is not None):
            stats.update(_prefix_keys(self.raw_step(prev_raw_snap, model), "raw_step"))

        if do_gap:
            stats.update(_prefix_keys(self.gap_to_shadow(model), "gap_to_shadow"))

        if do_shadow_step and (prev_shadow_snap is not None):
            stats.update(_prefix_keys(self.shadow_step(prev_shadow_snap), "shadow_step"))

        if eval_fn is not None and eval_every and (step is not None) and (step % eval_every == 0):
            if eval_kwargs is None:
                eval_kwargs = {}
            stats.update(self.eval_raw_and_shadow(model, lambda m: eval_fn(m, **eval_kwargs)))

        return stats
    
    def state_dict(self) -> dict:
        return {
            "cfg": {
                "beta": self.cfg.beta,
                "eps": self.cfg.eps,
                "include_buffers": self.cfg.include_buffers,
            },
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, sd: dict, strict: bool = False):
        # cfg 可以选择不强制覆盖
        shadow = sd.get("shadow", {})
        missing = []
        for k, v in shadow.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))
            else:
                if strict:
                    missing.append(k)
                else:
                    # allow new keys
                    self.shadow[k] = v.to(self.shadow_device) if self.shadow_device is not None else v
        if strict and missing:
            raise KeyError(f"missing shadow keys: {missing[:5]} ... total={len(missing)}")

    # -------------------------
    # optional: list keys (for debugging)
    # -------------------------

    def list_stat_keys(self) -> None:
        """
        Print the key namespaces produced by on_step_end (excluding eval_fn custom keys).
        Call once at startup.
        """
        demo = {}
        demo.update(_prefix_keys({"step_l2": 0.0, "theta_l2": 0.0, "rel": 0.0, "rel/common": 0.0}, "raw_step"))
        demo.update(_prefix_keys({"gap_l2": 0.0, "theta_l2": 0.0, "rel": 0.0, "rel/common": 0.0}, "gap_to_shadow"))
        demo.update(_prefix_keys({"step_l2": 0.0, "theta_l2": 0.0, "rel": 0.0, "rel/common": 0.0}, "shadow_step"))
        keys = sorted(demo.keys())
        print(f"[ShadowEMA] base keys ({len(keys)}):")
        for k in keys:
            print("  -", k)