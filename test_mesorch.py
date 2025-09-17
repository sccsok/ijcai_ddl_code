# -*- coding: utf-8 -*-
# @Date     : 2025/05/27
# @Author   : Chao Shuai
# @File     : test_mesorch.py
import os
import types
import inspect
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import glob
from tqdm import tqdm
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from Mesorch.IMDLBenCo.IMDLBenCo.training_scripts.utils import misc
from Mesorch.IMDLBenCo.IMDLBenCo.registry import MODELS


class FaceCropDataset(Dataset):
    def __init__(self, image_folder, transform):
        self.data = glob.glob(image_folder + '/*.png')
        self.data.sort()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        
        return {"image": tensor, "image_path": img_path}
    

def cleanup_state_dict(state_dict):
    """Clean model state dict by removing 'module.' prefix"""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

 
def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBench testing launch!', add_help=True)
    # Model name
    parser.add_argument('--model', default="Mesorch", type=str,
                        help='The name of applied model', required=True)
    
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size of the images in datasets')
    
    parser.add_argument('--image_folder', type=str, required=True,)
    
    parser.add_argument('--if_padding', action='store_true',
                        help='padding all images to same resolution.')
    
    parser.add_argument('--if_resizing', action='store_true', 
                        help='resize all images to same resolution.')
    # ------------------------------------
    # Testing 相关的参数
    parser.add_argument('--weights_path', default = 'Mesorch/ckpts/checkpoint-37.pth', type=str, help='path to the dir where saving checkpoints')
    parser.add_argument('--batch_size', default=16, type=int,
                        help="batch size for testing")
    # -----------------------
    parser.add_argument('--device', default=6, type=int)
    parser.add_argument('--num_workers', default=12, type=int)

    parser.add_argument('--save_dir', type=str, default='./results/mesorch_outputs/mask', help='Path to save predictions')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                    help='Threshold for white pixel ratio (0-1)')
    
    args, remaining_args = parser.parse_known_args()
    model_class = MODELS.get(args.model)
    
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args


def main(args, model_args):
    device = torch.device(args.device)
    
    _preprocess = T.Compose([
        T.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
    ])

    model = MODELS.get(args.model)
    if isinstance(model,(types.FunctionType, types.MethodType)):
        model_init_params = inspect.signature(model).parameters
    else:
        model_init_params = inspect.signature(model.__init__).parameters
        
    combined_args = {k: v for k, v in vars(args).items() if k in model_init_params}
    combined_args.update({k: v for k, v in vars(model_args).items() if k in model_init_params})
    model = model(**combined_args)
    model.to(device)
    
    ckpt = torch.load(args.weights_path, map_location='cpu')
    try:
        state_dict = cleanup_state_dict(ckpt['model'])
    except:
        state_dict = cleanup_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=True)
    print(f'Loaded weights from {args.weights_path}')
    model.eval()
    
    dataset = FaceCropDataset(args.image_folder, _preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    for batch in tqdm(dataloader, desc="Detecting"):
        # move to device
        images = batch["image"].to(device)
        paths = batch["image_path"]
        # /data1/home/shuaichao/ijcai_ddl_code/Mesorch/IMDLBenCo/IMDLBenCo/model_zoo/mesorch/mesorch.py
        output_dict = model(images)
        mask_pred = output_dict['pred_mask']
        
        save_predicted_masks(mask_pred, paths, args.save_dir)


def save_predicted_masks(mask_pred, paths, output_dir):
    """
    Save predicted masks as binary images with morphological post-processing.
    
    Args:
        mask_preds (torch.Tensor): Tensor of shape (B, 1, H, W)
        paths (List[str]): Original image paths (used to extract filenames)
        output_dir (str): Directory to save output masks
    """
    bin_masks = (mask_pred > 0.5).float()

    # Morphological kernel
    kernel = np.ones((5, 5), np.uint8)

    for i in tqdm(range(len(paths))):
        # Convert tensor to numpy
        mask_np = bin_masks[i, 0].cpu().numpy() * 255  # (H, W), float32 → [0,255]
        mask_np = mask_np.astype(np.uint8)

        # Morphological operations
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        mask_np = cv2.erode(mask_np, kernel, iterations=1)

        # Get filename from path
        filename = Path(paths[i]).name
        save_path = os.path.join(output_dir, filename)

        # Save
        cv2.imwrite(save_path, mask_np)
        

if __name__ == '__main__':
    args, model_args = get_args_parser()
    
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        
    main(args, model_args)