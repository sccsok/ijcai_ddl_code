# -*- coding: utf-8 -*-
# @Date     : 2025/05/26
# @Author   : Chao Shuai
# @File     : get_ijcai_json.py

from __future__ import annotations
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import os
import json
import numpy as np
from tqdm import tqdm

# ========= Dataset root directories =========
# dataset_root/
# ├── fake/           
# │   ├── 001.jpg
# │   ├── 002.jpg
# │   └── ...
# ├── real/        
# │   ├── 101.jpg
# │   ├── 102.jpg
# │   └── ...
# ├── mask/           # masks for fake images
# │   ├── 002.jpg
# │   └── ...
# ---------- Utility Functions ----------

def img_size(path: Path) -> tuple[int, int] | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h

def mask_is_all_black(mask_path: Path) -> bool | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return not np.any(mask)

def process_item(item: dict) -> dict | None:
    img_path: Path = item["img_path"]
    sz = img_size(img_path)
    if sz is None:
        print(f"⚠️ Cannot read image: {img_path}")
        return None
    w, h = sz

    if item["type"] == "fake":
        mask_path: Path = item["mask_path"]
        if not mask_path.exists():
            print(f"⚠️ Missing mask: {mask_path}")
            return None

        # NOTE Face with all_pixel_black mask is also real with label = 0
        all_black = mask_is_all_black(mask_path)
        if all_black is None:
            print(f"⚠️ Cannot read mask: {mask_path}")
            return None

        label = 0 if all_black else 1
        return {
            "file_name": str(img_path.resolve()),
            "width": w,
            "height": h,
            "mask_path": str(mask_path.resolve()),
            "label": label,
        }

    else:  # real image
        return {
            "file_name": str(img_path.resolve()),
            "width": w,
            "height": h,
            "mask_path": "",
            "label": 0,
        }

# ---------- Main Function ----------
def main(args):
    input_dirs = [Path(os.path.join(args.self_root_dir, p)) for p in os.listdir(args.self_root_dir)]
    input_dirs.append(Path(args.face_root_dir))
    
    output_json = Path(args.output_json)

    tasks: list[dict] = []
    for root in input_dirs:
        fake_dir, mask_dir, real_dir = root / "fake", root / "mask", root / "real"

        for img_path in fake_dir.glob("*.*"):
            mask_name = img_path.name
            tasks.append({
                "type": "fake",
                "img_path": img_path,
                "mask_path": mask_dir / mask_name,
            })

        tasks.extend({
            "type": "real",
            "img_path": img_path,
            "mask_path": None,
        } for img_path in real_dir.glob("*.*"))

    print(f"Total images queued: {len(tasks)}")

    records: list[dict] = []
    max_workers = min(32, (os.cpu_count() or 8) * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process_item, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            res = fut.result()
            if res is not None:
                records.append(res)

    if output_json.exists():
        try:
            output_json.unlink()
            print(f"🗑️ Removed existing JSON file: {output_json}")
        except Exception as e:
            print(f"⚠️ Failed to delete existing JSON file: {e}")

    with open(output_json, 'w') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Metadata written to {output_json.resolve()}  (Total: {len(records)} entries)")

# ---------- Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata.json from fake/real/mask datasets.")
    parser.add_argument(
        "--face_root_dir",
        type=str,
        required=True,
        help="Path to all face images extracted from IJCAI_DDL training dataset (each must contain fake/, real/, mask/)"
    )
    parser.add_argument(
        "--self_root_dir",
        type=str,
        required=True,
        help="Root path of self-collected dataset root containing multiple subfolders (each contains fake/, real/, mask/)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to output metadata.json file"
    )

    args = parser.parse_args()
    main(args)