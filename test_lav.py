# -*- coding: utf-8 -*-
# @Date     : 2025/05/27
# @Author   : Chao Shuai
# @File     : test_lav.py
import argparse
import yaml
import os
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from Lav.training.detectors import DETECTOR


def cleanup_state_dict(state_dict):
    """Clean model state dict by removing 'module.' prefix"""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


class FaceCropDataset(Dataset):
    def __init__(self, location_json, transform):
        with open(location_json, "r") as f:
            self.data = list(json.load(f).items())
        self.data.sort()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, _ = self.data[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return {"image": tensor, "image_path": path}


def run_detection(args, detect_model, preprocess, location_json):
    image_folder = args.image_folder
    original_images_root = args.original_images_root
    output_prediction_txt = Path(args.save_dir) / "prediction.txt"
    output_mask_dir = Path(args.save_dir) / "mask"
    
    if output_prediction_txt.exists():
        output_prediction_txt.unlink()
    if output_mask_dir.exists():
        shutil.rmtree(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    with open(location_json, "r") as f:
        face_locations = json.load(f)

    dataset = FaceCropDataset(location_json, preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = {}
    for batch in tqdm(dataloader, desc="Detecting"):
        images = batch["image"].to(args.device)
        paths = batch["image_path"]

        with torch.no_grad():
            pred_dict = detect_model({"image": images})
            cls_probs = torch.softmax(pred_dict["cls"], dim=1)[:, 1].cpu().numpy()

            seg_masks_2 = F.interpolate(pred_dict["seg2"], size=images.shape[2:], mode="bilinear", align_corners=False)
            seg_masks_2 = torch.argmax(seg_masks_2, dim=1).cpu().numpy().astype(np.uint8) * 255

        for i in range(len(paths)):
            path = Path(paths[i])
            basename = path.stem
            try:
                img_id = basename.rsplit("_", 1)[0]
                face_id = basename.rsplit("_", 1)[1]
            except:
                img_id = basename
                face_id = '000'

            if img_id not in results:
                results[img_id] = {"preds": [], "masks": [], "masks_seg1": [], "size": None}

            results[img_id]["preds"].append(cls_probs[i])
            results[img_id]["masks"].append((face_id, seg_masks_2[i]))

            if results[img_id]["size"] is None:
                orig_path = original_images_root / f"{img_id}.png"
                if not orig_path.exists():
                    orig_path = original_images_root / f"{img_id}.jpg"
                if not orig_path.exists():
                    continue
                h, w = cv2.imread(str(orig_path)).shape[:2]
                results[img_id]["size"] = (h, w)

    with open(output_prediction_txt, "w") as f:
        for img_id, data in results.items():
            preds = np.array(data["preds"])
            if np.all(preds < 0.5):
                final_pred = preds.mean()
            else:
                final_pred = preds[preds >= 0.5].mean()
            f.write(f"{img_id}.png,{final_pred:.4f}\n")

            h, w = data["size"]
            full_mask = np.zeros((h, w), dtype=np.uint8)

            for face_name, face_mask in data["masks"]:
                key = str(image_folder / f"{img_id}_{face_name}.png")
                if key not in face_locations:
                    continue
                top, bottom, left, right = face_locations[key]
                face_mask_resized = cv2.resize(face_mask, (right - left, bottom - top))
                face_mask_resized = (face_mask_resized > 0.5).astype(np.uint8)
                full_mask[top:bottom, left:right] = np.maximum(full_mask[top:bottom, left:right], face_mask_resized)
                
            kernel = np.ones((5, 5), np.uint8)
            full_mask = cv2.dilate(full_mask, kernel, iterations=1)  
            full_mask = cv2.erode(full_mask, kernel, iterations=1)
            
            cv2.imwrite(str(output_mask_dir / f"{img_id}.png"), full_mask * 255)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multithreaded Deepfake Detection')
    parser.add_argument('--image_folder', type=str, required=True,)
    parser.add_argument('--location_json', type=str, required=True,)
    parser.add_argument('--original_images_root', type=str, required=True)
    parser.add_argument('--weights_path', type=str, default="Lav/training/checkpoints/two_stream_epoch_24.pth")
    parser.add_argument('--save_dir', type=str, default="./results/lav_outputs")
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--max_workers', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=160)
    args = parser.parse_args()

    args.detector_path = "./Lav/training/config/detector/dual_stream.yaml"
    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(exist_ok=True)

    args.original_images_root = Path(args.original_images_root)
    args.image_folder = Path(args.image_folder)

    with open(args.detector_path) as f:
        config = yaml.safe_load(f)

    config.update({
        'weights_path': args.weights_path
    })

    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # args.device = "cpu"

    _preprocess = T.Compose([
        T.Resize((config['resolution'], config['resolution']), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(config['data_aug']['mean']),
                    std=torch.tensor(config['data_aug']['std']), inplace=True)
    ])

    detect_model = DETECTOR[config['model_name']](config).to(args.device)
    ckpt = torch.load(config['weights_path'], map_location=args.device)
    try:
        state_dict = cleanup_state_dict(ckpt['model_state'])
    except:
        state_dict = cleanup_state_dict(ckpt)
    detect_model.load_state_dict(state_dict, strict=True)
    print(f'Loaded weights from {config["weights_path"]}')
    detect_model.eval()

    run_detection(args, detect_model, _preprocess, args.location_json)

    print("All Done!")