# -*- coding: utf-8 -*-
# @Date     : 2025/05/27
# @Author   : Chao Shuai
# @File     : test_image_process.py
"""
Multi-process face cropping script for teh test dataset (no mask, no real/fake split).
Input: --root 
Output:
  ├── images/
  └── face_locations.json
"""

import os, cv2, torch, argparse, json
from pathlib import Path
from multiprocessing import Process, get_context, Manager
from tqdm import tqdm
from facedetector.face_utils import FaceDetector, norm_crop_mask
import numpy as np


# NOTE
def list_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".png")
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]

@torch.inference_mode()
def worker(proc_id, files, gpu, root_dir, out_dir, img_size, return_dict):
    torch.cuda.set_device(gpu)
    det = FaceDetector(device=f"cuda:{gpu}", _reinit_cuda=False, _max_inds=False, confidence_threshold=0.7)
    det.load_checkpoint("RetinaFace-Resnet50-fixed.pth")

    image_dir = out_dir / "test"
    image_dir.mkdir(parents=True, exist_ok=True)

    local_dict = {}
    pbar = tqdm(files, position=proc_id, desc=f"P{proc_id}", ncols=100)
    for fname in pbar:
        try:
            img_path = root_dir / fname
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            if img is None:
                continue

            _, lms = det.detect(img)
            if lms is None or len(lms) == 0:
                base = f"{Path(fname).stem}_001.png"
                save_path = (image_dir / base).resolve()
                cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                local_dict[str(save_path)] = [0, img.shape[0], 0, img.shape[1]]
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
                    save_face = img if face_ratio > 0.7 else face
                    save_path = (image_dir / base).resolve()
                    cv2.imwrite(str(save_path), cv2.cvtColor(save_face, cv2.COLOR_RGB2BGR))
                    local_dict[str(save_path)] = [t, b, l, r]

        except Exception as e:
            pbar.set_postfix(error=str(e)[:40])
    return_dict.update(local_dict)
    pbar.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ijcai/ddl_data/phase1/track1/test"))
    parser.add_argument("--out", type=Path, default=Path("/data2/ijcai_temp/test_0.7_th0.7"))
    parser.add_argument("--worker", type=int, default=12)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--size", type=int, default=300)
    args = parser.parse_args()

    tasks = [f.name for f in list_images(args.root)[0:1000]]
    print(f"Total images queued: {len(tasks)} | processes: {args.worker}")

    chunk = len(tasks) // args.worker + 1
    ctx = get_context("spawn")
    manager = Manager()
    return_dict = manager.dict()

    procs = []
    for i in range(args.worker):
        sub = tasks[i*chunk : (i+1)*chunk]
        if not sub:
            continue
        p = ctx.Process(target=worker, args=(i, sub, args.gpu, args.root, args.out, args.size, return_dict))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    json_path = args.out / "face_locations.json"
    with open(json_path, "w") as f:
        json.dump(dict(return_dict), f, indent=2)

    print("All done. Images saved to:", args.out)
    print("Face locations saved to:", json_path)

if __name__ == "__main__":
    main()
