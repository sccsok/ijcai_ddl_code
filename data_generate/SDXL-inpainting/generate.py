import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import PIL
from torchvision import transforms
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from diffusers import AutoPipelineForInpainting

# Define segmentation label colors (not directly used here, but kept for reference)
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
    (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128),
    (255, 165, 0), (255, 105, 180), (75, 0, 130), (240, 230, 140),
    (173, 216, 230), (255, 20, 147), (0, 0, 0), (255, 255, 255)
]

# Region group definitions (optional for potential extensions)
label_groups = {
    'nose': [2],
    'eyes': [4, 5], 
    'mouth': [10], 
    'ears': [6, 7], 
    'brows': [8, 9],
    'eyeglasses': [3],
    'lips': [11, 12],
    'hair': [13],
    'hat': [14],
    'ear_r': [15], 
    'neck_l': [16], 
    'neck': [17], 
    'cloth': [18]
}

# Set GPU device
device = "cuda:1"

# Load the Stable Diffusion XL Inpainting Pipeline
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir="./checkpoints"
).to(device)

# Load SegFormer-based face parser
processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = AutoModelForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").to("cpu")  # stays on CPU

def process(image_paths, out_fake_dir, out_mask_dir):
    os.makedirs(out_fake_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            width, height = image.size

            # Preprocess and predict semantic segmentation
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits.cpu()

                # Resize logits to match original image size
                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=image.size[::-1],
                    mode="bilinear",
                    align_corners=False,
                )

                pred_seg = upsampled_logits.argmax(dim=1)[0]
                pred_seg_np = pred_seg.detach().numpy().astype(np.uint8)
        
            # Convert grayscale mask to 3-channel
            pred_seg_bgr = cv2.cvtColor(pred_seg_np, cv2.COLOR_GRAY2BGR)
            mask = np.zeros_like(pred_seg_bgr)

            # Select one region to inpaint
            region_candidates = [[1], [2], [10, 11, 12], [13]]  # e.g. face, nose, mouth, hair
            i = random.randint(0, 3)
            i = 0  # Fixed to face region (label 1)

            for label in region_candidates[i]:
                mask = np.where(pred_seg_bgr == label, 255, mask)

            # Skip if mask is nearly empty
            if np.sum(mask) < 255:
                continue

            # Save binary mask (for debugging or training)
            out_mask_path = os.path.join(out_mask_dir, os.path.basename(image_path))
            cv2.imwrite(out_mask_path, mask.astype(np.uint8))

            # Convert to PIL for pipeline input
            mask = Image.fromarray(mask).convert("RGB")

            # Choose prompt based on masked region
            if i == 0 or i == 1:
                emos = ["sad", "confident", "angry", "surprised", "smiling"]
                prompt = f"a {random.choice(emos)} face"
            elif i == 2:
                emos = ["smiling", "grinning", "frowning", "pouting", "open-mouthed", "toothy"]
                prompt = f"a {random.choice(emos)} face"
            elif i == 3:
                clrs = ["black", "white", "yellow", "pink", "gray", "green"]
                prompt = f"a face with {random.choice(clrs)} hair"
            else:
                prompt = ""

            # Resize for SDXL
            image_resized = image.resize((1024, 1024))
            mask_resized = mask.resize((1024, 1024))

            generator = torch.Generator(device=device)  # random seed optional

            # Run inpainting
            with torch.no_grad():
                output = pipe(
                    prompt=prompt,
                    image=image_resized,
                    mask_image=mask_resized,
                    guidance_scale=8.0,
                    num_inference_steps=20,
                    strength=0.99,
                    generator=generator,
                ).images[0]

            # Resize result back to original resolution
            output = output.resize((width, height))

            # Save result
            out_image_path = os.path.join(out_fake_dir, os.path.basename(image_path))
            output.save(out_image_path)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == '__main__':
    real_image_dir = '../../face/train/real'
    fake_output_dir = '../../ddl_data/self/SDXL-inpainting/fake'
    mask_output_dir = '../../ddl_data/self/SDXL-inpainting/mask'

    # Load all image file paths
    image_paths = glob.glob(os.path.join(real_image_dir, "*.png"))
    process(image_paths, fake_output_dir, mask_output_dir)

    print("Done!")