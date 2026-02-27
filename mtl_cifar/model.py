import torch.nn as nn
import torch.nn.functional as F

# Tiny ResNet for CIFAR (ResNet-14-ish)
# -------------------------

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


class TinyResNet(nn.Module):
    """
    CIFAR version:
      conv3x3 stride1, no maxpool
      stages: [64,128,256] with strides [1,2,2]
    """
    def __init__(self, width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        # ResNet-14-ish: 2 blocks per stage (total 6 blocks)
        self.stage1 = nn.Sequential(
            BasicBlock(width, width, stride=1),
            BasicBlock(width, width, stride=1),
        )
        self.stage2 = nn.Sequential(
            BasicBlock(width, width*2, stride=2),
            BasicBlock(width*2, width*2, stride=1),
        )
        self.stage3 = nn.Sequential(
            BasicBlock(width*2, width*4, stride=2),
            BasicBlock(width*4, width*4, stride=1),
        )
        self.out_ch = width * 4

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x  # [B,C,H,W] ~ [B,256,8,8] if width=64


# -------------------------
# Multi-head model
# -------------------------

class MTLNet(nn.Module):
    def __init__(self, width=64, num_classes=10):
        super().__init__()
        self.backbone = TinyResNet(width=width)

        C = self.backbone.out_ch

        # head 1: classification
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc   = nn.Linear(C, num_classes)

        # head 2: rotation classification (4-way)
        self.rot_pool = nn.AdaptiveAvgPool2d(1)
        self.rot_fc   = nn.Linear(C, 4)

        # head 3: reconstruction (light decoder)
        # upsample 8x8 -> 32x32 (x4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(C, C//2, 4, stride=2, padding=1),  # 8->16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C//2, C//4, 4, stride=2, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.Conv2d(C//4, 3, 3, padding=1),
        )

    def forward(self, x):
        feat = self.backbone(x)             # [B,C,8,8]
        z1 = self.cls_pool(feat).flatten(1) # [B,C]
        z2 = self.rot_pool(feat).flatten(1)

        logits_cls = self.cls_fc(z1)
        logits_rot = self.rot_fc(z2)
        recon      = self.dec(feat)         # [B,3,32,32]
        return logits_cls, logits_rot, recon