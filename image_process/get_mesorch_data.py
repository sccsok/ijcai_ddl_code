# -*- coding: utf-8 -*-
# @Date     : 2025/05/27
# @Author   : Chao Shuai
# @File     : get_mesorch_data.py
import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Supported image file extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')


def is_image_file(filename, extensions=image_extensions):
    """Check if the given filename is an image file."""
    return filename.lower().endswith(extensions)


def process_image_file(filename, src_dir, label_dir=None):
    """
    Process a single image file.
    If `label_dir` is given, return [image_path, label_path] if label exists.
    Otherwise, return [image_path, 'Negative'].
    """
    if not is_image_file(filename):
        return None

    image_path = os.path.join(src_dir, filename)

    if label_dir:
        label_path = os.path.join(label_dir, filename)
        if os.path.exists(label_path):
            return [str(Path(image_path).resolve()), str(Path(label_path).resolve())]
        else:
            print(f"⚠️ Missing GT file for: {filename}")
            return None
    else:
        return [str(Path(image_path).resolve()), "Negative"]


def process_directory(fake_dir, mask_dir, real_dir):
    """
    Process a dataset directory with 'fake', 'mask', and 'real' subdirectories.
    Returns a list of [image_path, label_path or 'Negative'] pairs.
    """
    results = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        # Process fake images
        fake_futures = [executor.submit(process_image_file, f, fake_dir, mask_dir)
                        for f in os.listdir(fake_dir)]
        # Process real images
        real_futures = [executor.submit(process_image_file, f, real_dir)
                        for f in os.listdir(real_dir)]

        for fut in fake_futures + real_futures:
            pair = fut.result()
            if pair:
                results.append(pair)

    return results


def generate_and_save_pairs(flag: str, root_dir: str, self_root_dir: str, output_json: str):
    """
    Generate (image_path, label_path) pairs for fake images,
    and (image_path, 'Negative') pairs for real images.
    Also optionally include additional self-collected subfolders for training.
    """
    # Paths to standard dataset directories
    fake_dir = os.path.join(root_dir, flag, "fake")
    mask_dir = os.path.join(root_dir, flag, "mask")
    real_dir = os.path.join(root_dir, flag, "real")

    result = process_directory(fake_dir, mask_dir, real_dir)

    # Count initial fake and real samples
    fake_count = sum(1 for p in result if p[1] != "Negative")
    real_count = sum(1 for p in result if p[1] == "Negative")

    # If training and additional data is provided
    if flag == "train" and os.path.exists(self_root_dir):
        print(f"\n🔄 Processing self_root_dir: {self_root_dir}")
        subdirs = [os.path.join(self_root_dir, d) for d in os.listdir(self_root_dir)
                   if os.path.isdir(os.path.join(self_root_dir, d))]

        for subdir in subdirs:
            print(f"📁 Processing subdir: {subdir}")
            sub_fake = os.path.join(subdir, "fake")
            sub_mask = os.path.join(subdir, "mask")
            sub_real = os.path.join(subdir, "real")

            if not all(os.path.exists(d) for d in [sub_fake, sub_mask, sub_real]):
                print(f"❌ Skipped invalid subdir: {subdir}")
                continue

            sub_result = process_directory(sub_fake, sub_mask, sub_real)
            result.extend(sub_result)

            fake_count += sum(1 for p in sub_result if p[1] != "Negative")
            real_count += sum(1 for p in sub_result if p[1] == "Negative")

    # Save result to JSON
    save_path = output_json
    try:
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\n✅ Saved to: {save_path}")
        print(f"📊 Total: {len(result)} (Fake: {fake_count} / Real: {real_count})")
    except Exception as e:
        print(f"❌ Failed to save JSON: {e}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fake/real image path pairs for Mesorch dataset.")
    parser.add_argument("--flag", type=str, required=True, help="Dataset split: train / valid / test")
    parser.add_argument("--root_dir", type=str, default="",
                        help="Root path of dataset containing train/valid/test folders")
    parser.add_argument("--self_root_dir", type=str, default="",
                        help="Root path of self-collected dataset root containing multiple subfolders (each contains fake/, real/, mask/)")
    parser.add_argument("--output_json", type=str, default="/data1/home/shuaichao/ijcai_ddl_code/Mesorch/datasets",
                        help="Path to save the resulting JSON file")

    args = parser.parse_args()
    generate_and_save_pairs(args.flag, args.root_dir, args.self_root_dir, args.output_json)
