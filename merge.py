# -*- coding: utf-8 -*-
# @Date     : 2025/05/27
# @Author   : Chao Shuai
# @File     : merge.py
import cv2
import numpy as np
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def fuse_mask(fname, mask_dir_1, mask_dir_2, output_dir):
    try:
        mask_path_1 = mask_dir_1 / fname
        mask_path_2 = mask_dir_2 / fname

        if not mask_path_2.exists():
            return f"Warning: Missing {fname}"

        mask1 = cv2.imread(str(mask_path_1), cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(str(mask_path_2), cv2.IMREAD_GRAYSCALE)

        if mask1 is None or mask2 is None or mask1.shape != mask2.shape:
            return f"Error: Mask load/shape failed: {fname}"

        bin_mask1 = (mask1 > 127).astype(np.uint8) * 255
        bin_mask2 = (mask2 > 127).astype(np.uint8) * 255

        fused = np.maximum(bin_mask1, bin_mask2)

        cv2.imwrite(str(output_dir / fname), fused)
        return None
    except Exception as e:
        return f"Exception in {fname}: {str(e)}"

def parse_args():
    parser = argparse.ArgumentParser(description="Fuse two mask folders with max operation and copy prediction.txt.")
    parser.add_argument('--mask_dir_1', type=str, required=True, help='Path to first mask folder')
    parser.add_argument('--mask_dir_2', type=str, required=True, help='Path to second mask folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save fused masks')
    return parser.parse_args()

def main():
    args = parse_args()
    mask_dir_1 = Path(args.mask_dir_1) / "mask"
    mask_dir_2 = Path(args.mask_dir_2) / "mask"
    output_dir = Path(args.output_dir) / "mask"
    output_dir.mkdir(parents=True, exist_ok=True)

    file_list = sorted(mask_dir_1.glob("*.png"))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(fuse_mask, path.name, mask_dir_1, mask_dir_2, output_dir)
            for path in file_list
        ]

        for f in tqdm(futures, desc="Fusing masks"):
            result = f.result()
            if result:
                print(result)

    # Copy prediction.txt if provided
    pred_path = Path(args.mask_dir_1 + "/prediction.txt")
    if pred_path.exists():
        shutil.copy(pred_path, args.output_dir + "/prediction.txt")
        print(f"\nCopied prediction.txt to {args.output_dir}")
    else:
        print(f"\nWarning: prediction.txt not found at {pred_path}")

if __name__ == "__main__":
    main()
