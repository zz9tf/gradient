# plot_model_blocks.py
# Draw model skeleton by parameter-name blocks (same notion as plots_blockwise / block_monitor).
# Output: PNG (and optional Mermaid) of block-level diagram.
#
# Block IDs match grad_wrapper._default_monitor_block_fn:
#   name "backbone.stem.0.weight" -> block "backbone.stem"
#   name "cls_fc.weight" -> block "cls_fc"

import os
import sys
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import MTLNet


def block_id_from_param_name(name: str) -> str:
    """
    Same logic as gradient_wrapper.grad_wrapper._default_monitor_block_fn.
    E.g. 'backbone.stem.0.weight' -> 'backbone.stem', 'cls_fc.weight' -> 'cls_fc'.
    """
    n = name[7:] if name.startswith("module.") else name
    base = n.replace(".weight", "").replace(".bias", "")
    parts = base.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "unknown")


def collect_blocks_from_model(model: "torch.nn.Module") -> list[str]:
    """
    Collect unique block IDs from model.named_parameters() in traversal order.
    """
    seen = set()
    order = []
    for name, _ in model.named_parameters():
        bid = block_id_from_param_name(name)
        if bid not in seen:
            seen.add(bid)
            order.append(bid)
    return order


# ---------------------------------------------------------------------------
# MTLNet-specific block graph (data flow between blocks)
# backbone.stem -> stage1 -> stage2 -> stage3 -> [cls_fc, rot_fc, dec.0->dec.2->dec.4]
# ---------------------------------------------------------------------------
def mtlnet_block_edges():
    """Return list of (from_block, to_block) for MTLNet."""
    return [
        ("input", "backbone.stem"),
        ("backbone.stem", "backbone.stage1"),
        ("backbone.stage1", "backbone.stage2"),
        ("backbone.stage2", "backbone.stage3"),
        ("backbone.stage3", "cls_fc"),
        ("backbone.stage3", "rot_fc"),
        ("backbone.stage3", "dec.0"),
        ("dec.0", "dec.2"),
        ("dec.2", "dec.4"),
    ]


def block_display_name(block_id: str) -> str:
    """Short label for diagram so it fits in box (e.g. backbone.stage1 -> stage1)."""
    if block_id == "input":
        return "input"
    if block_id.startswith("backbone."):
        return block_id.split(".", 1)[1]  # stem, stage1, stage2, stage3
    return block_id  # cls_fc, rot_fc, dec.0, dec.2, dec.4


def draw_block_diagram(
    blocks: list[str],
    edges: list[tuple[str, str]],
    outpath: str,
    title: str = "MTLNet (parameter-name blocks)",
):
    """
    Draw a simple block diagram: one box per block, arrows for edges.
    Layout: backbone chain left-to-right, then three heads below.
    """
    # Node positions: we'll place by hand for clarity
    # backbone: x=0,1,2,3 for stem, s1, s2, s3
    # heads: from (3, 0) branch to cls_fc (4, 0.5), rot_fc (4, 0), dec chain (4, -0.5) then (5, -0.5), (6, -0.5)
    pos = {}
    idx = 0
    for b in ["input", "backbone.stem", "backbone.stage1", "backbone.stage2", "backbone.stage3"]:
        if b in blocks or b == "input":
            pos[b] = (idx, 0)
            idx += 1
    # heads
    if "cls_fc" in blocks:
        pos["cls_fc"] = (idx, 0.8)
    if "rot_fc" in blocks:
        pos["rot_fc"] = (idx, 0)
    if "dec.0" in blocks:
        pos["dec.0"] = (idx, -0.8)
    if "dec.2" in blocks:
        pos["dec.2"] = (idx + 1, -0.8)
    if "dec.4" in blocks:
        pos["dec.4"] = (idx + 2, -0.8)
    # any block not yet placed (e.g. global)
    for b in blocks:
        if b not in pos:
            pos[b] = (idx + 1, -1.5)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_aspect("equal")
    ax.axis("off")

    # Colors: backbone vs heads
    def color_for(b):
        if b == "input":
            return "#e0e0e0"
        if b.startswith("backbone."):
            return "#c8e6c9"  # light green
        if b in ("cls_fc", "rot_fc"):
            return "#bbdefb"  # light blue
        if b.startswith("dec."):
            return "#ffe0b2"  # light orange
        return "#f5f5f5"

    box_w, box_h = 0.58, 0.28
    for b in blocks:
        if b not in pos:
            continue
        x, y = pos[b]
        fc = color_for(b)
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02",
            facecolor=fc,
            edgecolor="gray",
            linewidth=1,
        )
        ax.add_patch(box)
        label = "input" if b == "input" else block_display_name(b)
        t = ax.text(x, y, label, ha="center", va="center", fontsize=9)
        t.set_clip_path(box)  # clip text inside box so it does not spill out

    # input node if used in edges
    if "input" in [e[0] for e in edges] and "input" not in blocks:
        x, y = pos.get("input", (0, 0))
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02",
            facecolor="#e0e0e0",
            edgecolor="gray",
            linewidth=1,
        )
        ax.add_patch(box)
        t = ax.text(x, y, "input", ha="center", va="center", fontsize=9)
        t.set_clip_path(box)

    # Arrows
    for (a, b) in edges:
        if a not in pos or b not in pos:
            continue
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        ax.annotate(
            "",
            xy=(x2 - box_w / 2 if x2 > x1 else x2 + box_w / 2, y2),
            xytext=(x1 + box_w / 2 if x1 < x2 else x1 - box_w / 2, y1),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        )

    ax.set_xlim(-0.6, idx + 2.5)
    ax.set_ylim(-1.4, 1.2)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_model_blocks] Saved: {outpath}")


def emit_mermaid(blocks: list[str], edges: list[tuple[str, str]], outpath: str):
    """Emit Mermaid flowchart text (same blocks/edges)."""
    lines = ["flowchart LR", "  %% MTLNet parameter-name blocks (same as plots_blockwise)"]
    for b in blocks:
        bid = b.replace(".", "_")
        label = block_display_name(b)
        lines.append(f'  {bid}["{label}"]')
    for (a, b) in edges:
        aid = a.replace(".", "_")
        bid = b.replace(".", "_")
        lines.append(f"  {aid} --> {bid}")
    text = "\n".join(lines)
    with open(outpath, "w") as f:
        f.write(text)
    print(f"[plot_model_blocks] Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Draw MTLNet skeleton by parameter-name blocks")
    parser.add_argument("-o", "--out", default=None, help="Output PNG path (default: mtl_cifar/model_blocks.png)")
    parser.add_argument("--mermaid", default=None, help="Also write Mermaid .mmd to this path")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    model = MTLNet(width=args.width, num_classes=args.num_classes)
    blocks = collect_blocks_from_model(model)
    edges = mtlnet_block_edges()

    outpath = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_blocks.png")
    draw_block_diagram(blocks, edges, outpath, title="MTLNet (parameter-name blocks)")

    if args.mermaid:
        emit_mermaid(blocks, edges, args.mermaid)
    else:
        mermaid_path = os.path.splitext(outpath)[0] + ".mmd"
        emit_mermaid(blocks, edges, mermaid_path)


if __name__ == "__main__":
    main()
