"""
Check class distribution in train and test .npy patches.
Run from hover_net project root.
"""
import numpy as np
from pathlib import Path
from collections import defaultdict

TYPE_NAMES = {0: "background/unknown", 1: "epithelial", 2: "lymphocyte", 3: "macrophage", 4: "neutrophil"}

def count_type_map(type_ch):
    """type_ch: H*W array from type_map."""
    uniq, cnt = np.unique(type_ch, return_counts=True)
    return dict(zip(uniq.tolist(), cnt.tolist()))

def stats_for_split(npy_dir, max_files=None):
    npy_dir = Path(npy_dir)
    files = sorted(npy_dir.glob("*.npy"))
    if max_files is not None:
        files = files[:max_files]
    pixel_counts = defaultdict(int)
    inst_counts = defaultdict(int)  # count instances per type (by patch)
    for path in files:
        data = np.load(path)
        if data.ndim != 3 or data.shape[2] < 5:
            continue
        type_map = data[..., 4].astype(np.int32)
        inst_map = data[..., 3].astype(np.int32)
        for tid in np.unique(type_map):
            pixel_counts[tid] += (type_map == tid).sum()
        for inst_id in np.unique(inst_map):
            if inst_id == 0:
                continue
            mask = inst_map == inst_id
            types_in_inst = type_map[mask]
            tid = int(np.median(types_in_inst))
            inst_counts[tid] += 1
    total_pix = sum(pixel_counts.values())
    total_inst = sum(inst_counts.values())
    return {
        "n_patches": len(files),
        "pixel_counts": dict(pixel_counts),
        "inst_counts": dict(inst_counts),
        "total_pixels": total_pix,
        "total_instances": total_inst,
    }

def print_ratio(name, s):
    print(f"\n=== {name} (n_patches={s['n_patches']}) ===")
    print("By pixel:")
    for tid in sorted(s["pixel_counts"].keys()):
        c = s["pixel_counts"][tid]
        r = c / (s["total_pixels"] or 1) * 100
        print(f"  type {tid} ({TYPE_NAMES.get(tid, '?')}): {c} px ({r:.2f}%)")
    print("By instance (nucleus):")
    for tid in sorted(s["inst_counts"].keys()):
        c = s["inst_counts"][tid]
        r = c / (s["total_instances"] or 1) * 100
        print(f"  type {tid} ({TYPE_NAMES.get(tid, '?')}): {c} inst ({r:.2f}%)")

if __name__ == "__main__":
    train_dir = "dataset/monusac/train/256x256_164x164"
    test_dir = "dataset/monusac/test/256x256_164x164"
    for dir_path in [train_dir, test_dir]:
        if not Path(dir_path).exists():
            print(f"Skip (not found): {dir_path}")
            continue
        s = stats_for_split(dir_path)
        print_ratio(dir_path, s)