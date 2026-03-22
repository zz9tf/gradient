import torch.nn as nn
import torch.nn.functional as F

# =========================
# Basic shared residual block
# =========================
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.short = None
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        skip = x if self.short is None else self.short(x)
        out = F.relu(out + skip, inplace=True)
        return out


# =========================
# Task-specific block adapter
# h -> h + Adapter_t(h)
# =========================
class TaskAdapter(nn.Module):
    def __init__(self, channels, bottleneck=16):
        super().__init__()
        hidden = max(channels // bottleneck, 4)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        # start near zero so initial model is close to shared backbone
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x):
        return x + self.net(x)


# =========================
# One shared block + optional task-specific adapters
# =========================
class TaskAwareBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, tasks=("cls", "rot", "rec"), bottleneck=16):
        super().__init__()
        self.shared_block = BasicBlock(in_ch, out_ch, stride=stride)
        self.adapters = nn.ModuleDict({
            t: TaskAdapter(out_ch, bottleneck=bottleneck) for t in tasks
        })

    def forward(self, x, task, use_lora=True):
        h = self.shared_block(x)
        if use_lora:
            h = self.adapters[task](h)
        return h


# =========================
# Shared backbone + optional task-specific adapters
# =========================
class TinyResNetTaskAdapter(nn.Module):
    def __init__(self, width=64, tasks=("cls", "rot", "rec"), bottleneck=16, use_lora=True):
        super().__init__()
        self.tasks = tasks
        self.use_lora = use_lora

        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList([
            TaskAwareBlock(width,   width,   stride=1, tasks=tasks, bottleneck=bottleneck),
            TaskAwareBlock(width,   width,   stride=1, tasks=tasks, bottleneck=bottleneck),

            TaskAwareBlock(width,   width*2, stride=2, tasks=tasks, bottleneck=bottleneck),
            TaskAwareBlock(width*2, width*2, stride=1, tasks=tasks, bottleneck=bottleneck),

            TaskAwareBlock(width*2, width*4, stride=2, tasks=tasks, bottleneck=bottleneck),
            TaskAwareBlock(width*4, width*4, stride=1, tasks=tasks, bottleneck=bottleneck),
        ])

        self.out_ch = width * 4

    def set_use_lora(self, use_lora: bool):
        self.use_lora = use_lora

    def forward(self, x, task):
        assert task in self.tasks
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x, task=task, use_lora=self.use_lora)
        return x


# =========================
# Multi-head model
# =========================
class MTLNet(nn.Module):
    def __init__(
        self,
        width=64,
        num_classes=10,
        tasks=("cls", "rot", "rec"),
        bottleneck=16,
        use_lora=True,
    ):
        super().__init__()
        self.tasks = tasks
        self.use_lora = use_lora

        self.backbone = TinyResNetTaskAdapter(
            width=width,
            tasks=tasks,
            bottleneck=bottleneck,
            use_lora=use_lora,
        )

        C = self.backbone.out_ch

        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc   = nn.Linear(C, num_classes)

        self.rot_pool = nn.AdaptiveAvgPool2d(1)
        self.rot_fc   = nn.Linear(C, 4)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(C, C // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C // 2, C // 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 4, 3, 3, padding=1),
        )

    def set_use_lora(self, use_lora: bool):
        self.use_lora = use_lora
        self.backbone.set_use_lora(use_lora)

    def forward_task(self, x, task):
        feat = self.backbone(x, task=task)

        if task == "cls":
            z = self.cls_pool(feat).flatten(1)
            return self.cls_fc(z)

        elif task == "rot":
            z = self.rot_pool(feat).flatten(1)
            return self.rot_fc(z)

        elif task == "rec":
            return self.dec(feat)

        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x):
        # no matter use_lora is True/False, still run 3 task-specific forwards
        logits_cls = self.forward_task(x, "cls")
        logits_rot = self.forward_task(x, "rot")
        recon      = self.forward_task(x, "rec")
        return logits_cls, logits_rot, recon
