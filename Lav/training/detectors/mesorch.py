"""
Referring from AAAI2025 paper:
"Mesoscopic Insights: Orchestrating Multi-scale & Hybrid Architecture for Image Manipulation Localization"
(https://arxiv.org/abs/2412.13753 )
"""

import sys
import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports
from components.frequency_feature import HighDctFrequencyExtractor, LowDctFrequencyExtractor
from metrics.base_metrics_class import calculate_f1_score, calculate_iou
from detectors.base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC


@DETECTOR.register_module(module_name='mesorch')
class Mesorch(AbstractDetector):
    """
    Mesoscopic IML Detector combining CNNs and Transformers for multi-scale feature extraction.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Build backbone and head modules
        self.build_backbone()

        # Define loss function
        self.loss_func = self.build_loss()

        # Reset tracking metrics
        self._reset_metrics()

    def build_backbone(self) -> None:
        """
        Builds the backbone network consisting of ConvNeXt and MixViT,
        along with frequency-based feature extractors and fusion modules.
        """
        # Use pretrained backbones
        self.convnext = BACKBONE['ConvNeXt'](conv_pretrain=self.config['backbone_config']['ConvNeXt']['pretrain_path'])
        self.segformer = BACKBONE['mixvit'](pretrain_path=self.config['backbone_config']['mixvit']['pretrain_path'])

        # Frequency feature extractors
        self.low_dct = LowDctFrequencyExtractor()
        self.high_dct = HighDctFrequencyExtractor()
        
        self.upsample = UpsampleConcatConv()

        # Feature reconstruction layers
        self.inverse = nn.ModuleList([nn.Conv2d(96, 1, 1) for _ in range(4)] +
            [nn.Conv2d(64, 1, 1) for _ in range(4)])

        # Attention gate generator
        self.gate = ScoreNetwork()

        # Final upsampling layer
        self.resize = nn.Upsample(
            size=(self.config['backbone_config']['image_size'], self.config['backbone_config']['image_size']),
            mode='bilinear',
            align_corners=True
        )

    def build_loss(self) -> nn.Module:
        """
        Builds the loss function(s) specified in the configuration.
        Supports both single and multiple losses.
        """
        loss_config = self.config['loss_func']
        if isinstance(loss_config, list):
            return nn.ModuleList([LOSSFUNC[name]() for name in loss_config])
        return LOSSFUNC[loss_config]()

    def _reset_metrics(self) -> None:
        """
        Resets internal metric buffers used during training and testing.
        """
        self.test_ious: List[float] = []
        self.test_f1s: List[float] = []

    @torch.no_grad()
    def get_train_metrics(self, data_dict: Dict, pred_dict: Dict) -> Dict:
        """
        Computes pixel-level metrics (IoU, F1 score) for the current batch during training.
        """
        iou = calculate_iou(pred_dict['pred_mask'].detach(), data_dict['mask'].detach())
        f1_score = calculate_f1_score(pred_dict['pred_mask'].detach(), data_dict['mask'].detach())
    
        return {
            'iou': iou,
            'f1_score': f1_score
        }

    @torch.no_grad()
    def get_test_metrics(self) -> Dict:
        """
        Computes average IoU and F1 score across all test samples.
        Also includes classification accuracy if available.
        """
        iou_avg = np.mean(self.test_ious)
        f1_avg = np.mean(self.test_f1s)
        acc = self.correct / self.total if self.total else 0.0

        self._reset_metrics()  # Clear metrics after evaluation

        return {
            'pixel_iou': iou_avg,
            'pixel_f1': f1_avg,
            'acc': acc
        }

    def features(self, data_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts multi-scale and multi-frequency features from the input image.
        Returns attention gates and fused features for mask prediction.
        """
        image = data_dict['image']

        # Extract high and low frequency components
        high_freq = self.high_dct(image)
        low_freq = self.low_dct(image)

        # Construct multi-input tensors
        input_high = torch.cat([image, high_freq], dim=1)
        input_low = torch.cat([image, low_freq], dim=1)
        input_all = torch.cat([image, high_freq, low_freq], dim=1)

        # Backbone feature extraction
        _, outs1 = self.convnext(input_high)
        _, outs2 = self.segformer(input_low)

        # Combine Transformer and CNN features
        inputs = outs1 + outs2

        # Fuse multi-scale features
        x, features = self.upsample(inputs)

        # Generate attention gates
        gate_outputs = self.gate(input_all)

        # Reconstruct per-level maps
        reduced = torch.cat([self.inverse[i](features[i]) for i in range(8)], dim=1)

        return gate_outputs, reduced

    def classifier(self, gate_outputs: torch.Tensor, reduced: torch.Tensor) -> torch.Tensor:
        """
        Applies attention-weighted fusion and generates the final binary mask.
        """
        logits_mask = torch.sum(gate_outputs * reduced, dim=1, keepdim=True)
        logits_mask = self.resize(logits_mask)
        return logits_mask

    def forward(self, data_dict: Dict, inference: bool = False) -> Dict:
        """
        Forward pass through the network.
        Optionally updates metrics during inference.
        """
        # Feature extraction
        gate_outputs, reduced = self.features(data_dict)

        # Mask prediction
        logits_mask = self.classifier(gate_outputs, reduced)

        # Sigmoid activation for output mask
        pred_mask = torch.sigmoid(logits_mask).float()

        pred_dict = {
            'logits_mask': logits_mask,
            'pred_mask': pred_mask
        }

        # Update metrics during inference
        if inference:
            self._update_inference_metrics(data_dict, pred_dict)

        return pred_dict

    def get_losses(self, data_dict: Dict, pred_dict: Dict) -> Dict:
        """
        Calculates the loss between predicted and ground truth masks.
        """
        masks = data_dict['mask']
        logits_mask = pred_dict['logits_mask']

        # Compute loss
        loss = self.loss_func(logits_mask, masks)

        return {'overall': loss}

    def _update_inference_metrics(self, data_dict: Dict, pred_dict: Dict) -> None:
        """
        Updates IoU and F1 score buffers during validation/testing.
        """
        true_mask = data_dict['mask'].detach()
        pred_mask = pred_dict['pred_mask'].detach()

        self.test_ious.append(calculate_iou(pred_mask, true_mask).item())
        self.test_f1s.append(calculate_f1_score(pred_mask, true_mask).item())


class LayerNorm2d(nn.LayerNorm):
    """
    Layer normalization for 2D spatial tensors (NCHW format).
    """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ScoreNetwork(nn.Module):
    """
    Attention gate generator using convolutional blocks.
    Outputs softmax-normalized weights over feature levels.
    """
    def __init__(self):
        super(ScoreNetwork, self).__init__()
        self.conv1 = nn.Conv2d(9, 192, kernel_size=7, stride=2, padding=3)
        self.invert = nn.Sequential(
            LayerNorm2d(192),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(192, 768, kernel_size=1),
            nn.Conv2d(768, 192, kernel_size=1),
            nn.GELU()
        )
        self.conv2 = nn.Conv2d(192, 8, kernel_size=7, stride=2, padding=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        shortcut = x
        x = self.invert(x)
        x = shortcut + x
        x = self.conv2(x)
        x = x.float()
        x = self.softmax(x)
        return x


class UpsampleConcatConv(nn.Module):
    """
    Multi-scale feature upsample and concatenation module.
    Used to fuse features from different stages of the backbone.
    """
    def __init__(self):
        super(UpsampleConcatConv, self).__init__()

        # Upsample paths for c2, c3, c4
        self.upsamplec2 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        self.upsamplec3 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        )
        self.upsamplec4 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        )

        # Upsample paths for s2, s3, s4
        self.upsamples2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upsamples3 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )
        self.upsamples4 = nn.Sequential(
            nn.ConvTranspose2d(512, 320, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, inputs):
        c1, c2, c3, c4, s1, s2, s3, s4 = inputs

        # Upsample and refine each level
        c2 = self.upsamplec2(c2)
        c3 = self.upsamplec3(c3)
        c4 = self.upsamplec4(c4)
        s2 = self.upsamples2(s2)
        s3 = self.upsamples3(s3)
        s4 = self.upsamples4(s4)

        # Concatenate all features
        x = torch.cat([c1, c2, c3, c4, s1, s2, s3, s4], dim=1)
        features = [c1, c2, c3, c4, s1, s2, s3, s4]

        return x, features
    
    
if __name__ == '__main__':
    # Example usage and testing
    import yaml
    from collections import OrderedDict
    
    def test_model_performance():
        """Test model FLOPs and parameters"""
        with open('/data1/home/shuaichao/ZJU_IMcert_DDP/training/config/detector/mesorch.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        model = Mesorch(config)
        dummy_data = {
            "image": torch.randn(4, 3, 512, 512),
            "label": torch.tensor([1, 0])
        }
        
        # 'logits_mask': logits_mask,
        # 'pred_mask': pred_mask
        # print(model(dummy_data)['logits_mask'])
        # print(model(dummy_data)['pred_mask'])
        for name, param in model.named_parameters():
            shape = list(param.shape)
            print(f"{name}: shape = {shape}")
            if shape == [1, 96, 1, 1]:
                print(f">>> FOUND parameter with shape [1, 96, 1, 1]: {name}")

    test_model_performance()