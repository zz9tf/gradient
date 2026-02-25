"""
extract_patches_monusac.py
"""

import tqdm
import numpy as np
from pathlib import Path
import random

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir
from dataset import get_dataset

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    type_classification = True

    win_size = [256, 256]
    step_size = [164, 164]
    extract_type = "valid"
    print(f"MoNuSAC dataset extraction mode: {extract_type}")

    dataset_name = "monusac"
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)

    # Your data root directory
    data_root = Path("/home/zheng/zheng/gradient/hover_net/MoNuSAC")

    save_root = Path("dataset/monusac")

    # Valid split ratio from train dataset
    valid_ratio = 0.2
    random_seed = 42

    def process_split(tif_list, split_name):
        """
        Process a list of tif files and extract patches.
        
        Args:
            tif_list: List of tif file paths
            split_name: Name of the split (e.g., 'train', 'valid', 'test')
        """
        print(f"{split_name} files:", len(tif_list))

        out_dir = save_root / split_name / f"{win_size[0]}x{win_size[1]}_{step_size[0]}x{step_size[1]}"
        rm_n_mkdir(str(out_dir))

        pbarx = tqdm.tqdm(total=len(tif_list), ascii=True, desc=f"{split_name} files")
        
        # Calculate how many images need to be padded
        sizes = []
        small_count = 0
        pad_area_ratio = []

        target_h, target_w = win_size[0], win_size[1]

        for tif_path in tif_list:
            img = parser.load_img(str(tif_path))
            H, W = img.shape[:2]
            sizes.append((H, W))

            # 是否需要 padding
            if H < target_h or W < target_w:
                small_count += 1

                # 计算 padding 占 patch 的比例（更有意义）
                padded_area = target_h * target_w
                real_area = H * W
                pad_ratio = 1 - real_area / padded_area
                pad_area_ratio.append(pad_ratio)

        sizes = np.array(sizes)

        print("min:", sizes.min(axis=0))
        print("median:", np.median(sizes, axis=0))
        print("max:", sizes.max(axis=0))

        if extract_type == "padding":
            total = len(sizes)
            print("\n--- padding stats ---")
            print(f"total images: {total}")
            print(f"need padding: {small_count} ({small_count/total:.2%})")

            if pad_area_ratio:
                pad_area_ratio = np.array(pad_area_ratio)
                print(f"mean pad ratio (only small images): {pad_area_ratio.mean():.2%}")
                print(f"median pad ratio: {np.median(pad_area_ratio):.2%}")
                print(f"max pad ratio: {pad_area_ratio.max():.2%}")

        # For valid mode: track how much is abandoned (images with 0 patches)
        abandoned_count = 0
        total_patches_this_split = 0
        n_processed = 0

        for tif_path in tif_list:

            xml_path = tif_path.with_suffix(".xml")
            if not xml_path.exists():
                pbarx.update()
                continue

            n_processed += 1
            base_name = tif_path.stem

            # Read image
            img = parser.load_img(str(tif_path))

            # Read annotation (need to pass img.shape)
            ann = parser.load_ann(
                str(xml_path),
                type_classification,
                img_hw=img.shape[:2]
            )

            stacked = np.concatenate([img.astype(np.int32), ann.astype(np.int32)], axis=-1)
            sub_patches = xtractor.extract(stacked, extract_type)

            if extract_type == "valid" and len(sub_patches) == 0:
                abandoned_count += 1
            total_patches_this_split += len(sub_patches)
            
            for idx, patch in enumerate(sub_patches):
                # 只对 test 做 “移除 ambiguous nuclei（seg+cls 都不要）”
                if split_name == "test" and type_classification:
                    inst = patch[..., 3]
                    tp   = patch[..., 4]

                    # ambiguous nuclei pixels: 有实例但 type==0
                    amb_pix = (inst > 0) & (tp == 0)

                    if amb_pix.any():
                        amb_ids = np.unique(inst[amb_pix])   # 这些是需要删除的实例 id
                        # 把这些实例从 inst_map 清掉（整块实例都删）
                        inst[np.isin(inst, amb_ids)] = 0
                        # 同步 type_map：背景/被删区域都设 0
                        tp[inst == 0] = 0

                    patch[..., 3] = inst
                    patch[..., 4] = tp

                np.save(out_dir / f"{base_name}_{idx:03d}.npy", patch)

            pbarx.update()

        pbarx.close()

        if extract_type == "valid":
            print(f"[valid] {split_name}: processed {n_processed} images, "
                  f"abandoned {abandoned_count} images (too small → 0 patches), "
                  f"total patches saved = {total_patches_this_split}")

    # Process train/test splits
    for split in ["train", "test"]:
        tif_list = sorted((data_root / split).rglob("*.tif"))
        
        # For train split, randomly split into train and valid
        if split == "train":
            # Set random seed for reproducibility
            random.seed(random_seed)
            np.random.seed(random_seed)
            
            # Shuffle and split
            tif_list_shuffled = tif_list.copy()
            random.shuffle(tif_list_shuffled)
            
            valid_size = int(len(tif_list_shuffled) * valid_ratio)
            train_tif_list = tif_list_shuffled[valid_size:]
            valid_tif_list = tif_list_shuffled[:valid_size]
            
            print(f"\n=== Splitting train dataset ===")
            print(f"Total train files: {len(tif_list_shuffled)}")
            print(f"Train files: {len(train_tif_list)} ({len(train_tif_list)/len(tif_list_shuffled):.2%})")
            print(f"Valid files: {len(valid_tif_list)} ({len(valid_tif_list)/len(tif_list_shuffled):.2%})")
            
            # Process train and valid separately
            process_split(train_tif_list, "train")
            process_split(valid_tif_list, "valid")
        else:
            # Process test split normally
            process_split(tif_list, split)

    print("Done.")
