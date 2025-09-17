import os
import json
import random
import numpy as np
import cv2
import random
from PIL import Image
import sys
from pathlib import Path

import torch
import albumentations as A
from torch.utils import data
from torchvision import transforms as T
from torchvision.utils import save_image
from torchvision.transforms import InterpolationMode

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.utils.albu import RandomCopyMove, RandomInpainting


class DeepfakeAbstractBaseDataset(data.Dataset):
    """Base dataset class for deepfake detection with data balancing and augmentation support"""
    
    def __init__(self, config=None, args=None, mode='train', preprocess=None):
        """
        Initialize dataset with configuration parameters and mode setting
        
        Args:
            config (dict): Configuration parameters including:
                - resolution: Target image resolution
                - data_aug: Data augmentation parameters
                - data_root: Root directory for dataset files
                - metadata_path: Path to metadata JSON file
            args (argparse.Namespace): Command line arguments containing:
                - local_rank: Process rank for distributed training
            mode (str): Dataset mode ('train' or 'test')
        """
        # Validate configuration integrity
        self._validate_config(config)
        # Initialize core attributes from config and arguments
        self._init_base_attributes(config, args, mode, preprocess)
        # Set up dataset paths from configuration
        self._init_dataset_paths()
        # Initialize data augmentation pipelines
        self._init_augmentations()
        # Collect and balance image paths with labels
        self.collect_img_and_label()
    
    
    def print_rank0(self, *args):
        """
        Print messages only from the main process in distributed training
        
        Args:
            *args: Variable length argument list for print function
        """
        if self.args is None or self.args.local_rank == 0:
            print(*args)

    def _validate_config(self, config):
        """
        Verify essential configuration parameters are present
        
        Args:
            config (dict): Configuration parameters to validate
            
        Raises:
            ValueError: If required configuration keys are missing
        """
        required_keys = ['resolution', 'data_aug']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

    def _init_base_attributes(self, config, args, mode, preprocess):
        """
        Initialize fundamental dataset attributes
        
        Args:
            config (dict): Configuration parameters
            args (argparse.Namespace): Command line arguments
            mode (str): Dataset mode ('train'/'test')
        """
        self.config = config
        self.args = args
        self.mode = mode
        self.preprocess = preprocess
        # Enable class balancing only in training mode
        self.balance = (mode == 'train')
        # Starting index for dataset sampling (used for large datasets)
        self.start = config.get('train_start', 0) if self.mode == 'train' else 0
        # Initialize containers for image paths and labels
        self.real_items = []
        self.fake_items = []
        self.image_items = []

    def _init_dataset_paths(self):
        """Initialize file system paths from configuration parameters"""
        # Root directory containing dataset files
        self.root = Path(self.config["data_root"])

    def collect_img_and_label(self):
        """
        Main entry point for dataset collection and balancing
        
        Handles:
        - Dataset file discovery (when root is a directory)
        - Real/fake label assignment
        - Class balancing operations
        - Final image/label list compilation
        """
        # Process all dataset files in directory mode
        with open(self.root, 'r') as f:
            data = json.load(f)
        
        for item in data:
            if int(item["label"]) == 0:
                self.real_items.append(item)
            else:
                self.fake_items.append(item)
        
        # Balance classes if required
        self.print_rank0(f"Pre-balance: real={len(self.real_items)}, fake={len(self.fake_items)}")
        if self.balance:
            self._balance_dataset()
        
        # Compile final dataset lists
        self.image_items = self.real_items + self.fake_items
        self.print_rank0(f"Post-balance: real={len(self.real_items)}, fake={len(self.fake_items)}")

    def _balance_dataset(self):
        """
        Balance real and fake samples through either:
        - Downsampling the majority class (real > fake)
        - Upsampling with duplication+random sampling (fake > real)
        
        Maintains exact class balance through either:
        - Random subsampling of real images (when real > fake)
        - Smart duplication with remainder handling (when fake > real)
        """
        real_len = len(self.real_items)
        fake_len = len(self.fake_items)
        
        if real_len == fake_len:
            return

        if real_len > fake_len:
            # Downsample real images to match fake count
            self.real_items = random.sample(self.real_items, fake_len)
        else:
            # Calculate duplication factors with remainder handling
            repeat_times = fake_len // real_len
            remainder = fake_len % real_len
            
            # Duplicate real samples and add random remainder
            self.real_items = (
                self.real_items * repeat_times +
                random.sample(self.real_items, remainder)
            )

    def __getitem__(self, index):
        """
        Retrieve and process a dataset item by index
        
        Args:
            index (int): Index of the item to retrieve
            
        Returns:
            dict: Containing:
                - image (Tensor): Processed image tensor
                - label (int): Class label (0=real, 1=fake)
                - path (str): Original image path
                
        Implements fault tolerance by returning first item on loading error
        """
        img_path = self.image_items[index]['file_name']
        mask_path = self.image_items[index]['mask_path']
        label = int(self.image_items[index]['label'])

        try:
            # Load image with PIL and convert to RGB
            img = Image.open(img_path).convert("RGB")
            if label == 0 or mask_path is None:
                mask = Image.new("L", img.size, 0)
            else:
                mask = Image.open(mask_path).convert("L") 
            img, mask = self._process_image(img, mask)
            
        except Exception as e:
            # Fallback to first item on loading failure
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(0)
        
        # Apply preprocessing and augmentation pipeline
        
        return {'image': img, 'label': label, 'mask': mask}

    def _process_image(self, img, mask):
        """
        Args:
            img (PIL.Image): Original input image
            
        Returns:
            Tensor: Processed image tensor ready for model input
        """
        img, mask = np.array(img).astype(np.uint8), np.array(mask).astype(np.uint8)
        # Training-specific preprocessing
        height, width = img.shape[:2] 
        mask = mask[:, :, None]
        
        assert img.shape[:2] == mask.shape[:2], "Image and mask dimensions must match"
        # img[:10, :, :] = 255  
        # mask[:10, :, :] = 255     
            
        # Apply albumentations augmentations
        if self.mode == 'train' and self.config['use_data_augmentation']:
            augmented = self.all_transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
            img = self.image_transform(image=img)['image']
         
        if self.preprocess is not None:
            img = self.preprocess(Image.fromarray(img))
        else:  
            img = self._after_augmentations(Image.fromarray(img))
            
        mask = self._after_mask_augmentations(Image.fromarray(mask.squeeze(-1)))
        # Convert to tensor and normalize
        
        return img, mask

    def _init_augmentations(self): 
        """
        Initialize complex augmentation pipeline using albumentations
        
        The pipeline includes:
        - Color space transformations
        - Geometric transformations
        - Resolution variations
        - Noise injection
        - Compression artifacts
        - Lighting variations
        
        Followed by standard normalization and tensor conversion
        """
        if 'mesorch' in self.config['model_name']:
            self.all_transform = A.Compose([
                # A.RandomScale(**self.config['data_aug']['random_scale']),
                RandomCopyMove(p=0.1),
                RandomInpainting(p=0.1),
                A.HorizontalFlip(p=0.5)
            ])
        else:
            self.all_transform = A.Compose([
                A.HorizontalFlip(p=0.5)
            ])
        self.image_transform = A.Compose([
            # Color transformations
            A.ToGray(p=0.2),  # 20% chance of grayscale conversion
            # Noise and quality degradation
            A.GaussNoise(**self.config['data_aug']['noise']),
            A.OneOf([
                A.GaussianBlur(**self.config['data_aug']['blur']),
                A.MotionBlur(p=0.2)
            ], p=0.5),  # 50% chance to apply blurring
            A.ImageCompression(**self.config['data_aug']['compression']),
            
            # Lighting and color variations
            A.OneOf([
                A.RandomBrightnessContrast(**self.config['data_aug']['brightness']),
                A.FancyPCA(),  # PCA-based color augmentation
                A.HueSaturationValue()
            ], p=0.5)
        ])
        
        self.all_transform.transforms.extend(A.Compose([
            # Color transformations
            # A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, 
                             **self.config['data_aug']['scale_rotate'])
        ]))
        
        # Post-augmentation processing
        self._after_augmentations = T.Compose([
            T.Resize((self.config['resolution'], self.config['resolution']), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),  # Convert to [0-1] range tensor
            T.Normalize(
                mean=torch.tensor(self.config['data_aug']['mean']),
                std=torch.tensor(self.config['data_aug']['std'])
            )
        ])
        
        self._after_mask_augmentations = T.Compose([
            T.Resize((self.config['resolution'], self.config['resolution']), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor()  # Convert to [0-1] range tensor
        ])
    
    def __len__(self):
        """Return total number of samples in the dataset"""
        return len(self.image_items)
        

if __name__ == "__main__":
    import time
    import yaml
    from tqdm import tqdm
    
    # Load detector configuration
    with open('/data1/home/shuaichao/ZJU_IMcert_DDP/training/config/detector/dual_stream.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['data_root'] = '/data1/home/shuaichao/ZJU_IMcert_DDP/training/dataset/ijcai_metadata.json'
    
    # Initialize training dataset
    train_set = DeepfakeAbstractBaseDataset(
        config=config,
        mode='train', 
    )
    
    # Create DataLoader with parallel loading
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=256,
        shuffle=True, 
        num_workers=16,
        pin_memory=True
    )

    # Denormalization parameters (should match training config)
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    
    # 'mean': [0.5482207536697388, 0.42340534925460815, 0.3654651641845703],
    # 'std': [0.2789176106452942, 0.2438540756702423, 0.23493893444538116]
    
    # Process batches and save sample images
    time1 = time.time()
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print("Batch tensor dimensions:", batch['image'].size())
        
        # Calculate processing time per batch
        time2 = time.time()
        print(f"Batch processing time: {(time2 - time1) * 1000:.2f}ms")
        time1 = time2
        
        # Denormalize images for visualization
        denormalized_image = batch['image'] * std[None, :, None, None] + mean[None, :, None, None]
        print(batch['label'])
        # Save batch as grid image
        save_image(denormalized_image, f'./visual/image/{iteration}.png', nrow=8)
        save_image(batch['mask'].float(), f'./visual/mask/{iteration}.png', nrow=8)
        
        exit()