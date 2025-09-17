# -*- coding: utf-8 -*-
# @Date     : 2025/05/26
# @Author   : Chao Shuai
# @File     : ijcai_data_process.py
'''
Multi-process face cropping tool
Output:
  dataset_root/
    ├── fake/
    ├── real/
    └── mask/
'''

import sys, os, cv2, torch, argparse, json
from pathlib import Path
from multiprocessing import Process, get_context, Manager
from tqdm import tqdm
from facedetector.face_utils import FaceDetector, norm_crop_mask
import numpy as np


def list_images(folder: Path):
    """List image files in a directory with common image extensions."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]


@torch.inference_mode()
def worker(proc_id, files, gpu, root_dir, out_dir, img_size):
    """Worker function for face detection and cropping.

    Args:
        proc_id: Process ID for tqdm display.
        files: A list of (kind, filename) tuples.
        gpu: GPU index to use.
        root_dir: Input image root directory.
        out_dir: Output directory for saving results.
        img_size: Target face crop size.
    """
    torch.cuda.set_device(gpu)

    # Initialize face detector
    det = FaceDetector(device=f"cuda:{gpu}", _reinit_cuda=False, _max_inds=False, confidence_threshold=0.7)
    det.load_checkpoint("RetinaFace-Resnet50-fixed.pth")

    # Create output subdirectories
    (out_dir / "fake").mkdir(parents=True, exist_ok=True)
    (out_dir / "real").mkdir(parents=True, exist_ok=True)
    (out_dir / "mask").mkdir(parents=True, exist_ok=True)

    # Progress bar for each process
    pbar = tqdm(files, position=proc_id, desc=f"P{proc_id}", ncols=100)
    for kind, fname in pbar:
        img_path = root_dir / kind / fname
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        if img is None:
            continue

        mask = None
        if kind == "fake":
            # Load corresponding mask for fake images
            mask_path = root_dir / "mask" / fname
            if not mask_path.exists():
                continue
            mask = cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_BGR2GRAY)

        # Detect landmarks
        _, lms = det.detect(img)

        # Case 1: No face detected
        if lms is None or len(lms) == 0:
            k = 1
            base = f"{Path(fname).stem}_{k:03d}.png"
            save_face = img
            mask_crop = mask
            save_dir = out_dir / kind
            full_path = str((save_dir / base).resolve())
            cv2.imwrite(full_path, cv2.cvtColor(save_face, cv2.COLOR_RGB2BGR))
            if kind == "fake" and mask_crop is not None:
                mask_path = str((out_dir / "mask" / base).resolve())
                cv2.imwrite(mask_path, mask_crop)

        # Case 2: Single face detected and image is square
        # Case 3: Multiple faces or non-square image
        else:
            for k, lm in enumerate(lms, 1):
                lm = lm.reshape(5, 2).astype(np.int32)
                face, _, loc = norm_crop_mask(img, lm, image_size=img_size)
                if face is None:
                    continue
                t, b, l, r = map(int, loc)
                face_area = (b - t) * (r - l)
                img_area = img.shape[0] * img.shape[1]
                face_ratio = face_area / img_area

                base = f"{Path(fname).stem}_{k:03d}.png"
                save_dir = out_dir / kind

                # Save original image or cropped face
                if face_ratio > 0.7:
                    save_face = img
                else:
                    save_face = face
                full_path = str((save_dir / base).resolve())
                cv2.imwrite(full_path, cv2.cvtColor(save_face, cv2.COLOR_RGB2BGR))

                # Save cropped mask if fake
                if kind == "fake" and mask is not None:
                    if face_ratio > 0.7:
                        mask_crop = mask
                    else:
                        mask_crop = mask[t:b, l:r]
                    mask_path = str((out_dir / "mask" / base).resolve())
                    cv2.imwrite(mask_path, mask_crop)

    pbar.close()


def main():
    """Main function: setup args, spawn workers, and process dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data1/home/pankun/ddl_data/phase1/track1/track1/train"),
                        help="Input root directory")
    parser.add_argument("--out", type=Path, default=Path("/data2/ijcai_temp/train_0.7_th0.7"),
                        help="Output directory for cropped results")
    parser.add_argument("--worker", type=int, default=12, help="Number of processes")
    parser.add_argument("--gpu", type=int, default=6, help="GPU index")
    parser.add_argument("--size", type=int, default=300, help="Face crop size")
    args = parser.parse_args()

    # Build task list (kind, filename)
    tasks = [("fake", f.name) for f in list_images(args.root / "fake")]
    tasks += [("real", f.name) for f in list_images(args.root / "real")]
    print(f"Total images queued: {len(tasks)} | processes: {args.worker}")

    # Split tasks among workers
    chunk = len(tasks) // args.worker + 1
    ctx = get_context("spawn")
    procs = []

    for i in range(args.worker):
        sub = tasks[i*chunk : (i+1)*chunk]
        if not sub:
            continue
        p = ctx.Process(target=worker, args=(i, sub, args.gpu, args.root, args.out, args.size))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("All done. Images saved to:", args.out)

if __name__ == "__main__":
    main()
