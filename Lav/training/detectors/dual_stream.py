# detectors/dual_stream_detector.py
import os
import sys
import logging
from typing import Dict, List
import numpy as np
from sklearn import metrics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from detectors.base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train
from components.srm_conv import SRMConv2d_simple
from components.linear_fusion import HdmProdBilinearFusion
from components.modules import *

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='dual_stream')
class DualStreamDetector(AbstractDetector):
    """
    Two-Stream Xception-based detector with SRM branch + RGB branch,
    classification head & segmentation head.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self._validate_config()

        self.params = {
            'location': {
                'size': 19,
                'channels': [64, 128, 256, 728, 728, 728],
                'mid_channel': 512
            },
            'cls_size': 10,
            'HBFusion': {
                'hidden_dim': 2048,
                'output_dim': 4096,
            }
        }

        self.build_backbone()
        self._build_heads()
        self.loss_func = self.build_loss()
        self._reset_metrics()


    def _validate_config(self) -> None:
        required = [
            'loss_func'              # str or [str, ...]
        ]
        for k in required:
            if k not in self.config:
                raise KeyError(f'Missing config key: {k}')


    def build_backbone(self):
        TransferModel = BACKBONE[self.config['backbone_name']]
        # RGB
        self.rgb_backbone = TransferModel(self.config['backbone_config'])
        self._load_pretrained_weights(self.rgb_backbone)
        # SRM
        self.srm_backbone = TransferModel(self.config['backbone_config'])
        self._load_pretrained_weights(self.srm_backbone)
        
        self.srm_conv0 = SRMConv2d_simple(inc=3)
        
        ch = self.params['location']['channels']
        mid = self.params['location']['mid_channel']
        self.score_layers = nn.ModuleList([
            BasicConv2d(ch[i], mid, kernel_size=1) for i in range(6)
        ])

        self.msff = MPFF(size=self.params['location']['size'])
        self.HBFusion = HdmProdBilinearFusion(
            dim1=sum(ch[:-1]), dim2=2048,
            hidden_dim=self.params['HBFusion']['hidden_dim'],
            output_dim=self.params['HBFusion']['output_dim']
        )

        # Cross-modal & Low-freq GA
        self.cmc_layers = nn.ModuleList([
            CMCE(in_channel=ch[i]) for i in range(3)
        ])
        self.lfe_layers = nn.ModuleList([
            LFGA(in_channel=ch[i+3]) for i in range(3)
        ])

    def _build_heads(self):
        out_dim = self.params['HBFusion']['output_dim']
        self.cls_header = nn.Sequential(
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(out_dim, 2)
        )

        seg_in = 728 + 64 + 16 + 4 + 1 + 1  
        self.seg_header1 = nn.Sequential(
            nn.BatchNorm2d(seg_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(seg_in, 2, kernel_size=1, bias=False)
        )
        
        self.seg_header2 = nn.Sequential(
            nn.Conv2d(728, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, bias=False)
        )

    def _load_pretrained_weights(self, model: nn.Module):
        """Load and adapt pretrained weights"""
        try:
            # state_dict = torch.load(self.config['pretrained'], map_location='cpu')
            current_file = Path(__file__).resolve()
            parent_dir = current_file.parent
            upper_dir = parent_dir.parent
            ckpt_path = upper_dir / "checkpoints" / "xception-b5690688.pth"
            state_dict = torch.load(ckpt_path, map_location='cpu')
            # Process pointwise convolution weights
            state_dict = {
                k: v.unsqueeze(-1).unsqueeze(-1) if 'pointwise' in k else v
                for k, v in state_dict.items()
                if 'fc' not in k
            }
            
            # Load weights with strict=False to allow partial loading
            load_result = model.load_state_dict(state_dict, strict=False)
            
            logger.info(f"Successfully loaded pretrained weights from {self.config['pretrained']}")
            logger.debug(f"Missing keys: {load_result.missing_keys}")
            logger.debug(f"Unexpected keys: {load_result.unexpected_keys}")
            
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {str(e)}")
            raise

    def build_loss(self):
        loss_config = self.config['loss_func']
        
        if isinstance(loss_config, list):
            return nn.ModuleList([
                LOSSFUNC[name]() for name in loss_config
            ])
        return LOSSFUNC[loss_config]()

    def _pad_max_pool(self, x):
        b, c, h, w = x.size()
        pad = (abs(h % self.params['cls_size'] - self.params['cls_size'])
               % self.params['cls_size'])
        pad2d = nn.ReplicationPad2d(
            (pad // 2, (pad + 1) // 2, pad // 2, (pad + 1) // 2)
        ).to(x.device)
        x = pad2d(x)
        pool = nn.MaxPool2d(
            kernel_size=x.size(-1) // self.params['cls_size'],
            stride=x.size(-1) // self.params['cls_size']
        )
        return pool(x)

    def _get_mask(self, mask):
        b, c, h, w = mask.size()
        pad = (abs(h % self.params['location']['size'] -
                   self.params['location']['size'])
               % self.params['location']['size'])
        pad2d = nn.ReplicationPad2d(
            (pad // 2, (pad + 1) // 2, pad // 2, (pad + 1) // 2)
        ).to(mask.device)
        mask = pad2d(mask)
        pool = nn.MaxPool2d(
            kernel_size=mask.size(-1) // self.params['location']['size'],
            stride=mask.size(-1) // self.params['location']['size']
        )
        mask = pool(mask)
        mask[mask > 0] = 1.0
        return mask

    def features(self, x):
        srm = self.srm_conv0(x)

        # === Stage-0 | 64×150×150 ===
        x0 = self.rgb_backbone.fea_part1_0(x)
        y0 = self.srm_backbone.fea_part1_0(srm)
        x0, y0 = self.cmc_layers[0](x0, y0)

        # === Stage-1 | 128×75×75 ===
        x1 = self.rgb_backbone.fea_part1_1(x0)
        y1 = self.srm_backbone.fea_part1_1(y0)
        x1, y1 = self.cmc_layers[1](x1, y1)

        # === Stage-2 | 256×38×38 ===
        x2 = self.rgb_backbone.fea_part1_2(x1)
        y2 = self.srm_backbone.fea_part1_2(y1)
        x2, y2 = self.cmc_layers[2](x2, y2)

        # === Stage-3 | 728×19×19 ===
        x3 = self.rgb_backbone.fea_part1_3(x2 + y2)
        y3 = self.srm_backbone.fea_part1_3(x2 + y2)
        y3 = self.lfe_layers[0](y3, x3)

        # === Stage-4 | 728×19×19 ===
        x4 = self.rgb_backbone.fea_part2_0(x3)
        y4 = self.srm_backbone.fea_part2_0(y3)
        y4 = self.lfe_layers[1](y4, x4)

        # === Stage-5 | 728×19×19 ===
        x5 = self.rgb_backbone.fea_part2_1(x4)
        y5 = self.srm_backbone.fea_part2_1(y4)
        y5 = self.lfe_layers[2](y5, x5)

        # === Stage-6 | 2048×10×10 (fc before GAP) ===
        x6 = self.rgb_backbone.fea_part3(x5)
        y6 = self.srm_backbone.fea_part3(y5)

        # print(x0.size(), x1.size(), x2.size(), x3.size(), x4.size(), x5.size())
        score_u = [layer(t) for layer, t in zip(self.score_layers,
                                               [x0, x1, x2, x3, x4, x5])]
        # print(score_u[0].size(), score_u[1].size(), score_u[2].size(), score_u[3].size(), score_u[4].size(), score_u[5].size())
        x4m = self.msff(score_u[4], score_u[5])
        x3m = self.msff(score_u[3], score_u[5])
        x2m = self.msff(score_u[2], score_u[5])
        x1m = self.msff(score_u[1], score_u[5])
        x0m = self.msff(score_u[0], score_u[5])
        seg_fea1 = torch.cat((x0m, x1m, x2m, x3m, x4m, x5), dim=1)
        # seg_fea2 = self.compute_up_features(score_u[0], score_u[1:])
        seg_fea2 = F.interpolate(x5, size=(x5.size(2), x5.size(3)), mode='bilinear', align_corners=False)

        y0m, y1m, y2m, y3m, y5m = map(self._pad_max_pool,
                                      [y0, y1, y2, y3, y5])
        mul_fea = torch.cat((y0m, y1m, y2m, y3m, y5m), dim=1)
        cls_fea = self.HBFusion(mul_fea, y6)

        return cls_fea, seg_fea1, seg_fea2
    
    def classifier(self, features):
        """Classify extracted features"""
        logits = self.cls_header(features[0])
        seg_logits1 = self.seg_header1(features[1])
        seg_logits2 = self.seg_header2(features[2])
        
        return logits, seg_logits1, seg_logits2
    
    def forward(self, data_dict: Dict, inference: bool = False) -> Dict:
        x = data_dict['image']
        mask_gt = data_dict.get('mask')     # mask
        logits, seg_logits1, seg_logits2 = self.classifier(self.features(x))
        seg_logits2 = F.interpolate(seg_logits2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        # print(logits.size(), seg_logits1.size(), seg_logits2.size())

        pred_dict = {
            'cls': logits,
            'prob': torch.softmax(logits, dim=1)[:, 1],
            'seg1': seg_logits1,
            'seg2': seg_logits2
        }

        if inference:
            self._update_inference_metrics(data_dict, pred_dict)

        if mask_gt is not None:
            if isinstance(mask_gt, list):
                dmask_gt = [self._get_mask(m) for m in mask_gt]
            else:
                dmask_gt = self._get_mask(mask_gt)
            pred_dict['dmask_gt'] = dmask_gt
            pred_dict['mask_gt'] = mask_gt

        return pred_dict

    def get_losses(self, data_dict: Dict, pred_dict: Dict) -> Dict:
        if isinstance(self.loss_func, nn.ModuleList):
            # Example for multi-loss configuration
            losses = {}
            losses['cls'] = self.loss_func[0](pred_dict['cls'], data_dict['label'])
            if 'mask_gt' in pred_dict and 'seg_loss' in self.config['loss_func']:
                losses['seg1'] = self.loss_func[1](pred_dict['seg1'], pred_dict['dmask_gt']) 
                losses['seg2'] = self.loss_func[1](pred_dict['seg2'], pred_dict['mask_gt'])
                losses['seg'] = losses['seg1'] + losses['seg2']
            
            losses['overall'] = losses['cls'] + self.config.get('seg_loss_weight', 1.0) * (losses['seg1'] + losses['seg2'])
            
            return losses
            
        return {'overall': self.loss_func(pred_dict['cls'], data_dict['label'])}

    @torch.no_grad()
    def get_train_metrics(self, data_dict: Dict, pred_dict: Dict) -> Dict:
        return calculate_metrics_for_train(
            data_dict['label'].detach(),
            pred_dict['cls'].detach()
        )

    @torch.no_grad()
    def get_test_metrics(self) -> Dict:
        y_pred = np.concatenate(self.probs)
        y_true = np.concatenate(self.labels)
        metrics_dict = {
            'auc': metrics.roc_auc_score(y_true, y_pred),
            'ap': metrics.average_precision_score(y_true, y_pred),
            'acc': self.correct / self.total if self.total else 0.0,
        }
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        metrics_dict['eer'] = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]

        if hasattr(self, 'seg_iou_total'):
            metrics_dict['miou'] = self.seg_iou_total / max(self.seg_iou_count, 1)

        self._reset_metrics()
        return metrics_dict

    def _reset_metrics(self):
        self.probs: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.correct: int = 0
        self.total: int = 0

        self.seg_iou_total = 0.0
        self.seg_iou_count = 0

    def _update_inference_metrics(self, data_dict: Dict, pred_dict: Dict):
        self.probs.append(pred_dict['prob'].detach().cpu().numpy().ravel())
        self.labels.append(data_dict['label'].detach().cpu().numpy().ravel())

        pred_label = pred_dict['cls'].argmax(dim=1)
        self.correct += (pred_label == data_dict['label']).sum().item()
        self.total += data_dict['label'].size(0)

        if 'mask_gt' in pred_dict:
            pred_seg = pred_dict['seg'].argmax(dim=1).float()
            gt_seg = pred_dict['mask_gt'].float()
            inter = ((pred_seg * gt_seg).sum(dim=(1, 2))).cpu()
            union = ((pred_seg + gt_seg) > 0).sum(dim=(1, 2)).cpu()
            self.seg_iou_total += (inter / (union + 1e-6)).sum().item()
            self.seg_iou_count += pred_seg.size(0)

    def compute_up_features(self, base_feature, other_features):
        B, C, H, W = base_feature.shape
        inner_product_maps = []

        for feat in other_features:
            feat_resized = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            
            inner_map = torch.sum(base_feature * feat_resized, dim=1, keepdim=True)
            inner_product_maps.append(inner_map)
            
        inner_product_maps.append(F.interpolate(other_features[-1], size=(H, W), mode='bilinear', align_corners=False))

        return torch.cat(inner_product_maps, dim=1)  # (B, 5 + 256, H, W)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
 
 
if __name__ == '__main__':
    # Example usage and testing
    import yaml
    from thop import profile, clever_format
    from PIL import Image
    from torchvision import transforms
    
    def test_model_performance():
        """Test model FLOPs and parameters"""
        with open('/data1/home/shuaichao/ZJU_IMcert_DDP/training/config/detector/dual_stream.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        geom = transforms.Compose([
            transforms.Resize((299, 299), interpolation=Image.BILINEAR),
        ])
        
        to_tensor_img = transforms.Compose([
            transforms.ToTensor(),                       # HWC→CHW, [0,1]
            transforms.Normalize([0.5]*3, [0.5]*3),      # 可按需要修改
        ])

        to_tensor_mask = transforms.ToTensor()      # 只有 0/1 像素
        
        model = DualStreamDetector(config)
        img = Image.open("/data2/ijcai/self/BELM/fake/0a0ae13458dd9474279875a629d559a2_001.png").convert("RGB")
        mask = Image.open("/data2/ijcai/self/BELM/mask/0a0ae13458dd9474279875a629d559a2_001.png").convert("L")  
        
        img  = geom(img)
        mask = geom(mask)

        img  = to_tensor_img(img).unsqueeze(0)
        mask = to_tensor_mask(mask).unsqueeze(0)
        
        print(img.size(), mask.size())
        
        dummy_data = {
            "image": img,
            "mask": mask,
            "label": torch.tensor([0])
        }
        
        out = model(dummy_data)
        print(out['mask_gt'])
        print(out['dmask_gt'])
    
        print(model.get_losses(dummy_data, out))
        
    #     # Profile model complexity
    #     flops, params = profile(model, inputs=(dummy_data,))
    #     flops, params = clever_format([flops, params], "%.3f")
        
    #     print(f"Model FLOPs: {flops}")
    #     print(f"Trainable Parameters: {params}")
    
    test_model_performance()