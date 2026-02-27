import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import JSONLLogger, save_json
import time

import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from model import MTLNet

from gradient_wrapper.grad_wrapper import GradAggregator, GradAggConfig


# -------------------------
# Utils
# -------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rotate_batch(x: torch.Tensor, k: int) -> torch.Tensor:
    # k in {0,1,2,3}: rotate 0/90/180/270
    return torch.rot90(x, k, dims=(2, 3))


def make_rot_task(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # returns (x_rot, y_rot) where y_rot in [0..3]
    bs = x.size(0)
    ks = torch.randint(0, 4, (bs,), device=x.device)
    x_rot = x.clone()
    for i in range(bs):
        x_rot[i] = rotate_batch(x_rot[i:i+1], int(ks[i].item()))[0]
    return x_rot, ks

def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item()) if x.numel() == 1 else x.detach().cpu().tolist()
    return x

def to_jsonable(d):
    out = {}
    for k, v in d.items():
        out[k] = to_float(v)
    return out

# -------------------------
# Train
# -------------------------

@dataclass
class TrainCfg:
    seed: int = 0
    device: str = "cuda"
    batch_size: int = 256
    epochs: int = 30
    lr: float = 0.1
    wd: float = 5e-4
    num_workers: int = 4

    # losses weights (external)
    w_cls: float = 1.0
    w_rot: float = 1.0
    w_rec: float = 1.0

    # grad-aggregation mode
    grad_mode: str = "mgda"   # sum | pcgrad | graddrop | mgda | cagrad | dwa | gradnorm | uw_heuristic | nash_mtl


def main(cfg: TrainCfg):
    run_dir = os.path.join("runs_cifar", cfg.grad_mode)
    os.makedirs(run_dir, exist_ok=True)

    train_log = JSONLLogger(os.path.join(run_dir, "train.jsonl"))
    eval_log  = JSONLLogger(os.path.join(run_dir, "eval.jsonl"))

    # 保存 config（一次）
    save_json(os.path.join(run_dir, "config.json"), cfg.__dict__)
    
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # dataset
    tr = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    te = T.Compose([T.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tr)
    testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=te)

    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    model = MTLNet(width=64, num_classes=10).to(device)

    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.0, weight_decay=0.0, nesterov=False)
    # # cosine schedule
    # def lr_at_epoch(ep):
    #     return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * ep / cfg.epochs))

    # your aggregator
    agcfg = GradAggConfig(mode=cfg.grad_mode)
    def common_param_filter(n, p):
        return n.startswith("backbone.")
    agg = GradAggregator(model, agcfg, common_param_filter=common_param_filter, verbose=True)
    
    best_acc = 0.0
    global_step = 0
    t0 = time.time()

    for ep in range(cfg.epochs):
        model.train()
        for it, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # build rotation task from same images (same batch)
            x_rot, y_rot = make_rot_task(x)

            logits_cls, _, recon = model(x)
            _, logits_rot, _ = model(x_rot)

            loss_cls = F.cross_entropy(logits_cls, y)
            loss_rot = F.cross_entropy(logits_rot, y_rot)
            loss_rec = F.mse_loss(recon, x)

            losses = {"cls": loss_cls, "rot": loss_rot, "rec": loss_rec}
            weights = {"cls": cfg.w_cls, "rot": cfg.w_rot, "rec": cfg.w_rec}

            opt.zero_grad(set_to_none=True)
            stats = agg.backward(losses, weights=weights)
            opt.step()

            if it % 50 == 0:
                rec = {
                    "t": time.time() - t0,
                    "epoch": ep,
                    "iter": it,
                    "step": global_step,
                    "mode": cfg.grad_mode,
                    "loss_cls": float(loss_cls.detach().cpu().item()),
                    "loss_rot": float(loss_rot.detach().cpu().item()),
                    "loss_rec": float(loss_rec.detach().cpu().item()),
                    "lr": opt.param_groups[0]["lr"],
                    "stats": to_jsonable(stats),
                }
                train_log.write(rec)

                msg = (f"ep {ep:03d} it {it:04d} "
                    f"Lcls {loss_cls.item():.3f} Lrot {loss_rot.item():.3f} Lrec {loss_rec.item():.3f} "
                    f"shrink {stats['shrink_ratio'].item():.3f} cosRF {stats['cos_raw_final'].item():.3f}")
                print(msg)

            global_step += 1

        # update LR
        # lr_now = lr_at_epoch(ep + 1)
        for pg in opt.param_groups:
            # pg["lr"] = lr_now
            pg["lr"] = cfg.lr

        # quick eval (classification only)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits_cls, _, _ = model(x)
                pred = logits_cls.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / max(total, 1)
        # print(f"[eval] ep {ep:03d} acc_cls={acc:.4f} lr={lr_now:.5f}")
        print(f"[eval] ep {ep:03d} acc_cls={acc:.4f} lr={cfg.lr:.5f}")
        
        # 记录 eval
        eval_log.write({
            "t": time.time() - t0,
            "epoch": ep,
            "mode": cfg.grad_mode,
            "acc_cls": acc,
            # "lr": lr_now,
            "lr": cfg.lr,
        })

        # save last
        torch.save({
            "model": model.state_dict(),
            "epoch": ep,
            "acc_cls": acc,
            "best_acc": best_acc,
            "mode": cfg.grad_mode,
            "cfg": cfg.__dict__,
        }, os.path.join(run_dir, "last.pt"))

        # save best
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model": model.state_dict(),
                "epoch": ep,
                "acc_cls": acc,
                "best_acc": best_acc,
                "mode": cfg.grad_mode,
                "cfg": cfg.__dict__,
            }, os.path.join(run_dir, "best.pt"))

    print("done.")


if __name__ == "__main__":
    grad_modes = [
        "sum", "pcgrad", "graddrop", "mgda", "cagrad", "dwa", "gradnorm", "uw_heuristic", "nash_mtl"
    ]
    cfg = TrainCfg(
        epochs=30,
        batch_size=8,
        grad_mode=grad_modes[1],
    )
    main(cfg)