# train.py
# CIFAR10 MTL toy: cls + rot + recon
# Configurable:
#   - GradAggregator mode
#   - (optional) CommonGpopSurgery (per-task surgery BEFORE aggregation)
#   - (optional) block monitor (pre/post around gpop)
#
# GPU:
#   Control GPU ONLY via bash:
#     CUDA_VISIBLE_DEVICES=$GPU_ID python train.py ...
#   This script will NOT set CUDA_VISIBLE_DEVICES.

import os, sys
import argparse
import time
import random
from dataclasses import dataclass
from typing import Dict, Optional

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()

# ---- grad aggregation mode ----
parser.add_argument(
    "--mode",
    "--grad_mode",
    dest="grad_mode",
    type=str,
    default="sum",
    help="sum | pcgrad | graddrop | mgda | cagrad | dwa | gradnorm | uw_heuristic | nash_mtl",
)

# ---- gpop ----
parser.add_argument("--gpop", action="store_true", help="enable common gpop surgery (per-task, before aggregation)")
parser.add_argument("--gpop_policy", type=str, default="gg", help="gg | cov_mul | cov_inv")
parser.add_argument("--gpop_beta", type=float, default=0.999, help="EMA beta for gpop")
parser.add_argument("--gpop_merge", type=str, default="sum", help="sum | mean (merge per-task G for EMA base)")
parser.add_argument("--gpop_task_norm", action="store_true", help="normalize each task gradient before gpop (OFF by default)")
parser.add_argument("--gpop_task_norm_common_only", action="store_true", help="if set, compute norm on common dims only")
parser.add_argument("--gpop_cov_biased", action="store_true", help="use biased denom=T (default unbiased: T-1)")

# cov_inv only
parser.add_argument("--gpop_damping", type=float, default=1e-3, help="damping for cov_inv: (Cov + damp*I)")
parser.add_argument("--gpop_cg_iters", type=int, default=30, help="CG max iters for cov_inv")
parser.add_argument("--gpop_cg_tol", type=float, default=1e-6, help="CG tol for cov_inv")

# ---- monitor ----
parser.add_argument("--monitor", action="store_true", help="enable pre/post block monitor around gpop")
parser.add_argument("--monitor_detach", action="store_true", help="detach inside monitor (default False unless set)")
parser.add_argument("--monitor_eps", type=float, default=1e-8)
parser.add_argument("--monitor_cov_unbiased", action="store_true", help="unbiased cov in monitor (default False unless set)")
parser.add_argument("--monitor_gpop_beta", type=float, default=0.999)
parser.add_argument("--monitor_gpop_update", action="store_true")
parser.add_argument("--monitor_gpop_warmup", type=int, default=0)
parser.add_argument("--monitor_cov_k", type=int, default=3)

# ---- train misc ----
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--bs", "--batch_size", dest="batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=4)

# external loss weights
parser.add_argument("--w_cls", type=float, default=1.0)
parser.add_argument("--w_rot", type=float, default=1.0)
parser.add_argument("--w_rec", type=float, default=1.0)

# ---- representation-space gradient filter ----
parser.add_argument("--rp_corr", action="store_true", help="enable representation-space gradient filter")
parser.add_argument("--rp_layer", type=str, default="", help="target module name for representation filter")
parser.add_argument("--rp_weight", type=str, default="inv_sqrt", help="inv | inv_sqrt | flat | log_inv")
parser.add_argument("--rp_eps", type=float, default=1e-8)
parser.add_argument("--rp_lowrank_k", type=int, default=0)
parser.add_argument("--rp_detach_repr", action="store_true")

parser.add_argument("--rp_monitor", action="store_true", help="enable representation rank monitor (erank, lambda1_ratio per layer)")
parser.add_argument("--show_layers", action="store_true", help="show available layers for representation rank monitor")

# common filter
parser.add_argument("--common_prefix", type=str, default="backbone.", help="param name prefix treated as common")

args, _ = parser.parse_known_args()

# Do NOT set CUDA_VISIBLE_DEVICES here; bash controls it.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import JSONLLogger, save_json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

from model import MTLNet

# Your wrapper
from gradient_wrapper.grad_wrapper import GradAggregator, GradAggConfig
from gradient_wrapper.rp_hook import RepresentationHookManager
from gradient_wrapper.grad_rp_uwug import GradRPCorrConfig, make_repr_grad_hook
from representation_wrapper.repre_monitor import RepresentationRankMonitor, RepresentationRankMonitorConfig

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_rot_task(x: torch.Tensor):
    # x: [B,C,H,W]
    B = x.size(0)
    ks = torch.randint(0, 4, (B,), device=x.device)
    x_rot = x.clone()
    for k in range(4):
        mask = ks == k
        if mask.any():
            x_rot[mask] = torch.rot90(x[mask], k, dims=(2, 3))
    return x_rot, ks


def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item()) if x.numel() == 1 else x.detach().cpu().tolist()
    return x


def to_jsonable(d: Dict):
    return {k: to_float(v) for k, v in d.items()}


# ---------------------------------------------------------------------
# Train cfg
# ---------------------------------------------------------------------
@dataclass
class TrainCfg:
    seed: int = 0
    device: str = "cuda:0"
    batch_size: int = 16
    epochs: int = 30
    lr: float = 1e-4
    wd: float = 0.0
    momentum: float = 0.0
    num_workers: int = 4
    grad_mode: str = "sum"

    # losses weights (external)
    w_cls: float = 1.0
    w_rot: float = 1.0
    w_rec: float = 1.0


def main(cfg: TrainCfg):
    run_name = f"{cfg.grad_mode}_{int(time.time())}"
    run_dir = os.path.join("runs_cifar", run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_log = JSONLLogger(os.path.join(run_dir, "train.jsonl"))
    eval_log = JSONLLogger(os.path.join(run_dir, "eval.jsonl"))

    # Save config once at the beginning (include cli args; gpop_policy="none" when gpop off)
    _cli = dict(vars(args))
    if not _cli.get("gpop", False):
        _cli["gpop_policy"] = "none"
    save_json(os.path.join(run_dir, "config.json"), {**cfg.__dict__, "cli": _cli})

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # dataset
    tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    te = T.Compose([T.ToTensor()])

    trainset_full = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True, transform=tr)
    testset = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=te)

    # Split train into 90% train + 10% valid
    n_train = int(0.9 * len(trainset_full))
    n_valid = len(trainset_full) - n_train
    trainset, validset = random_split(trainset_full, [n_train, n_valid])

    train_loader = DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    valid_loader = DataLoader(
        validset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # Display steps per epoch at first initialization
    steps_per_epoch_train = len(train_loader)
    steps_per_epoch_valid = len(valid_loader)
    print(f"[init] train: {steps_per_epoch_train} steps/epoch (n_train={n_train}), valid: {steps_per_epoch_valid} steps/epoch (n_valid={n_valid})")

    model = MTLNet(width=64, num_classes=10).to(device)
    if bool(args.show_layers):
        for name, module in model.named_modules():
            print(name, "->", module.__class__.__name__)
        print("--------------------------------")
        input()
    else:
        print("\n[Available module names for --show_layers]")
    
    
    # ---------------- representation-space gradient filter ----------------
    rp_hooker = None
    rp_cfg = None

    if bool(args.rp_corr):
        if str(args.rp_layer).strip() == "":
            raise ValueError("--rp_corr is set but --rp_layer is empty")

        rp_hooker = RepresentationHookManager(
            detach=False,      # must keep graph
            retain_grad=False  # not needed for this method
        )
        rp_hooker.register_by_name(model, [str(args.rp_layer)])

        rp_cfg = GradRPCorrConfig(
            enabled=True,
            eps=float(args.rp_eps),
            weight_mode=str(args.rp_weight),
            clamp_eig_min=0.0,
            low_rank_k=int(args.rp_lowrank_k),
            detach_repr=bool(args.rp_detach_repr),
        )

    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.wd,
        nesterov=False,
    )

    # -----------------------------------------------------------------
    # Build GradAggConfig from CLI
    # NOTE: these fields exist in your GradAggConfig dataclass :contentReference[oaicite:1]{index=1}
    # -----------------------------------------------------------------
    agcfg = GradAggConfig(
        mode=str(args.grad_mode),

        # gpop knobs (policy is "none" when gpop is disabled so config is accurate)
        gpop_enabled=bool(args.gpop),
        gpop_policy_kind="none" if not args.gpop else str(args.gpop_policy).lower(),
        gpop_ema_beta=float(args.gpop_beta),
        gpop_merge_kind=str(args.gpop_merge).lower(),
        gpop_task_grad_norm=bool(args.gpop_task_norm),
        gpop_task_grad_norm_common_only=bool(args.gpop_task_norm_common_only),
        gpop_unbiased=(not bool(args.gpop_cov_biased)),
        gpop_cov_inv_damping=float(args.gpop_damping),
        gpop_cov_inv_max_iter=int(args.gpop_cg_iters),
        gpop_cov_inv_tol=float(args.gpop_cg_tol),

        # monitor knobs
        gpop_monitor=bool(args.monitor),
        monitor_detach=bool(args.monitor_detach),
        monitor_eps=float(args.monitor_eps),
        monitor_cov_unbiased=bool(args.monitor_cov_unbiased),
        monitor_gpop_beta=float(args.monitor_gpop_beta),
        monitor_gpop_update=bool(args.monitor_gpop_update),
        monitor_gpop_warmup_steps=int(args.monitor_gpop_warmup),
        monitor_cov_mode_k=int(args.monitor_cov_k),
    )

    def common_param_filter(name: str, p: torch.nn.Parameter) -> bool:
        # grad_wrapper 内部会 strip module. 后再传 filter（你那份实现是这样写的） :contentReference[oaicite:2]{index=2}
        return name.startswith(str(args.common_prefix))

    agg = GradAggregator(model, agcfg, common_param_filter=common_param_filter, verbose=True)

    # Representation rank monitor (optional): erank / lambda1_ratio per layer
    repr_mon = None
    repr_rank_hooker = None
    if bool(args.rp_monitor):
        repr_mon_cfg = RepresentationRankMonitorConfig(
            prefix="repr",
            cov_unbiased=True,
            mode_k=3,
        )
        repr_mon = RepresentationRankMonitor(repr_mon_cfg)
        repr_rank_hooker = RepresentationHookManager(
            detach=True,
            retain_grad=False,
        )
        repr_rank_layers = [
            "backbone.stem",
            "backbone.stage1",
            "backbone.stage2",
            "backbone.stage3",
            "dec",
            "cls_pool",
            "rot_pool",
        ]
        repr_rank_hooker.register_by_name(model, repr_rank_layers)

    best_acc = 0.0
    global_step = 0
    t0 = time.time()

    n_iters_per_epoch = len(train_loader)
    for ep in range(cfg.epochs):
        model.train()
        for it, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x_rot, y_rot = make_rot_task(x)

            if rp_hooker is not None:
                rp_hooker.clear()
            if repr_rank_hooker is not None:
                repr_rank_hooker.clear()

            # ---------------- forward 1: original image (cls + rec) ----------------
            logits_cls, _, recon = model(x)

            # Representation rank stats from current batch/block outputs (only when --rp_monitor)
            repr_stats = (
                repr_mon.monitor(repr_rank_hooker.repr_cache, step=global_step)
                if repr_mon is not None and repr_rank_hooker is not None
                else {}
            )

            h_main = None
            if rp_hooker is not None:
                if args.rp_layer not in rp_hooker.repr_cache:
                    raise ValueError(
                        f"Target rp layer '{args.rp_layer}' not found in repr cache after model(x)"
                    )
                h_main = rp_hooker.repr_cache[args.rp_layer]

            # ---------------- forward 2: rotated image (rot) ----------------
            if rp_hooker is not None:
                rp_hooker.clear()

            _, logits_rot, _ = model(x_rot)

            h_rot = None
            if rp_hooker is not None:
                if args.rp_layer not in rp_hooker.repr_cache:
                    raise ValueError(
                        f"Target rp layer '{args.rp_layer}' not found in repr cache after model(x_rot)"
                    )
                h_rot = rp_hooker.repr_cache[args.rp_layer]

            loss_cls = F.cross_entropy(logits_cls, y)
            loss_rot = F.cross_entropy(logits_rot, y_rot)
            loss_rec = F.mse_loss(recon, x)

            losses = {"cls": loss_cls, "rot": loss_rot, "rec": loss_rec}
            weights = {"cls": cfg.w_cls, "rot": cfg.w_rot, "rec": cfg.w_rec}

            if rp_cfg is not None:
                if h_main is not None:
                    hook_main = make_repr_grad_hook(h_main, rp_cfg)
                    if hook_main is not None:
                        h_main.register_hook(hook_main)

                if h_rot is not None:
                    hook_rot = make_repr_grad_hook(h_rot, rp_cfg)
                    if hook_rot is not None:
                        h_rot.register_hook(hook_rot)

            opt.zero_grad(set_to_none=True)
            grad_stats = agg.backward(losses, weights=weights)
            opt.step()

            stats = {}
            stats.update(grad_stats)
            stats.update(repr_stats)

            if it % 50 == 0 and not (ep == 0 and it == 0):
                rec = {
                    "t": time.time() - t0,
                    "epoch": ep,
                    "iter": it,
                    "step": global_step,
                    "mode": cfg.grad_mode,
                    "loss_cls": float(loss_cls.detach().cpu().item()),
                    "loss_rot": float(loss_rot.detach().cpu().item()),
                    "loss_rec": float(loss_rec.detach().cpu().item()),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "stats": to_jsonable(stats),
                }
                train_log.write(rec)

                msg = (
                    f"ep {ep:03d} it {it:04d} step {global_step:06d} "
                    f"Lcls {loss_cls.item():.3f} Lrot {loss_rot.item():.3f} Lrec {loss_rec.item():.3f} "
                )

                def _add(k, fmt=".3f"):
                    nonlocal msg
                    if k in stats and torch.is_tensor(stats[k]) and stats[k].numel() == 1:
                        msg += f"{k}={float(stats[k]):{fmt}} "

                # pre/post quick signals (depends on your monitor; your block_monitor produces these keys)
                _add("pre.common.eff_sum", ".3f")
                _add("post.common.eff_sum", ".3f")
                _add("pre.common.viol_frac", ".3f")
                _add("post.common.viol_frac", ".3f")

                _add("pre.common.erank", ".2f")
                _add("post.common.erank", ".2f")
                _add("pre.common.condish", ".2f")
                _add("post.common.condish", ".2f")

                _add("pre.common.gpop_rho_mean", "+.3f")
                _add("post.common.gpop_rho_mean", "+.3f")
                _add("pre.common.gpop_drift", ".3f")
                _add("post.common.gpop_drift", ".3f")

                _add("pre.global.eff_sum", ".3f")
                _add("post.global.eff_sum", ".3f")
                
                _add("repr.backbone.stem.erank", ".2f")
                _add("repr.backbone.stage1.erank", ".2f")
                _add("repr.backbone.stage2.erank", ".2f")
                _add("repr.backbone.stage3.erank", ".2f")
                _add("repr.dec.erank", ".2f")

                _add("repr.backbone.stage1.lambda1_ratio", ".3f")
                _add("repr.backbone.stage2.lambda1_ratio", ".3f")
                _add("repr.backbone.stage3.lambda1_ratio", ".3f")

                print(msg)

            global_step += 1

        # Log last step of epoch if it was not already logged (it % 50 != 0)
        if n_iters_per_epoch > 0 and (it % 50) != 0:
            rec = {
                "t": time.time() - t0,
                "epoch": ep,
                "iter": it,
                "step": global_step - 1,
                "mode": cfg.grad_mode,
                "loss_cls": float(loss_cls.detach().cpu().item()),
                "loss_rot": float(loss_rot.detach().cpu().item()),
                "loss_rec": float(loss_rec.detach().cpu().item()),
                "lr": float(opt.param_groups[0]["lr"]),
                "stats": to_jsonable(stats),
            }
            train_log.write(rec)

        # eval on valid (all three heads)
        model.eval()
        correct_cls_v, total_cls_v = 0, 0
        correct_rot_v, total_rot_v = 0, 0
        rec_loss_sum_v, rec_count_v = 0.0, 0
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # classification + reconstruction head on original image
                logits_cls_v, _, recon_v = model(x)
                pred_cls_v = logits_cls_v.argmax(dim=1)
                correct_cls_v += (pred_cls_v == y).sum().item()
                total_cls_v += y.numel()

                # reconstruction loss (per-pixel MSE)
                rec_loss_v = F.mse_loss(recon_v, x, reduction="sum")
                rec_loss_sum_v += float(rec_loss_v.item())
                rec_count_v += x.numel()

                # rotation head: build rotation task and evaluate accuracy
                x_rot_v, y_rot_v = make_rot_task(x)
                _, logits_rot_v, _ = model(x_rot_v)
                pred_rot_v = logits_rot_v.argmax(dim=1)
                correct_rot_v += (pred_rot_v == y_rot_v).sum().item()
                total_rot_v += y_rot_v.numel()

        acc_cls_valid = correct_cls_v / max(total_cls_v, 1)
        acc_rot_valid = correct_rot_v / max(total_rot_v, 1)
        loss_rec_valid = rec_loss_sum_v / max(rec_count_v, 1)

        # keep classification valid accuracy as the main metric for early stopping / checkpointing
        acc_valid = acc_cls_valid

        print(
            f"[eval] ep {ep:03d} "
            f"acc_cls_valid={acc_cls_valid:.4f} "
            f"acc_rot_valid={acc_rot_valid:.4f} "
            f"loss_rec_valid={loss_rec_valid:.6f} "
            f"lr={cfg.lr:.5f}"
        )

        eval_log.write(
            {
                "t": time.time() - t0,
                "epoch": ep,
                "mode": cfg.grad_mode,
                "acc_cls_valid": acc_cls_valid,
                "acc_rot_valid": acc_rot_valid,
                "loss_rec_valid": loss_rec_valid,
            }
        )

        ckpt = {
            "model": model.state_dict(),
            "epoch": ep,
            "acc_cls_valid": acc_valid,
            "best_acc": best_acc,
            "mode": cfg.grad_mode,
            "cfg": cfg.__dict__,
            "cli": vars(args),
        }
        torch.save(ckpt, os.path.join(run_dir, "last.pt"))

        if acc_valid > best_acc:
            best_acc = acc_valid
            ckpt["best_acc"] = best_acc
            torch.save(ckpt, os.path.join(run_dir, "best.pt"))

    # Final evaluation on test set (all three heads)
    model.eval()
    correct_cls, total_cls = 0, 0
    correct_rot, total_rot = 0, 0
    rec_loss_sum, rec_count = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # classification + reconstruction head on original image
            logits_cls, _, recon = model(x)
            pred_cls = logits_cls.argmax(dim=1)
            correct_cls += (pred_cls == y).sum().item()
            total_cls += y.numel()

            # reconstruction loss (per-pixel MSE)
            rec_loss = F.mse_loss(recon, x, reduction="sum")
            rec_loss_sum += float(rec_loss.item())
            rec_count += x.numel()

            # rotation head: build rotation task and evaluate accuracy
            x_rot, y_rot = make_rot_task(x)
            _, logits_rot, _ = model(x_rot)
            pred_rot = logits_rot.argmax(dim=1)
            correct_rot += (pred_rot == y_rot).sum().item()
            total_rot += y_rot.numel()

    acc_cls_test = correct_cls / max(total_cls, 1)
    acc_rot_test = correct_rot / max(total_rot, 1)
    loss_rec_test = rec_loss_sum / max(rec_count, 1)

    final_results = {
        "acc_cls_test": acc_cls_test,
        "acc_rot_test": acc_rot_test,
        "loss_rec_test": loss_rec_test,
        "best_acc_valid_cls": best_acc,
        "mode": cfg.grad_mode,
        "epochs": cfg.epochs,
    }
    print(
        f"[FINAL] cls_acc={acc_cls_test:.4f} "
        f"rot_acc={acc_rot_test:.4f} "
        f"rec_loss={loss_rec_test:.6f} "
        f"(best_valid_cls={best_acc:.4f})"
    )
    save_json(os.path.join(run_dir, "final_results.json"), final_results)
    eval_log.write(
        {
            "t": time.time() - t0,
            "epoch": "final",
            "mode": cfg.grad_mode,
            "acc_cls_test": acc_cls_test,
            "acc_rot_test": acc_rot_test,
            "loss_rec_test": loss_rec_test,
        }
    )

    print("done.")


if __name__ == "__main__":
    cfg = TrainCfg(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        wd=float(args.wd),
        momentum=float(args.momentum),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        grad_mode=str(args.grad_mode),
        w_cls=float(args.w_cls),
        w_rot=float(args.w_rot),
        w_rec=float(args.w_rec),
    )
    main(cfg)