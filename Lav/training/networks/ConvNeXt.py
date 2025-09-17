import sys
import timm
import torch
import torch.nn as nn

sys.path.append('..')
from utils.registry import BACKBONE


@BACKBONE.register_module(module_name="ConvNeXt")
class ConvNeXt(timm.models.convnext.ConvNeXt):
    def __init__(self, conv_pretrain=False):
        super(ConvNeXt, self).__init__(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
        if conv_pretrain:
            print("Load Convnext pretrain.")
            model = timm.create_model('convnext_tiny', pretrained=True)
            self.load_state_dict(model.state_dict())
        original_first_layer = self.stem[0]
        new_first_layer = nn.Conv2d(6, original_first_layer.out_channels,
                                    kernel_size=original_first_layer.kernel_size, 
                                    stride=original_first_layer.stride,
                                    padding=original_first_layer.padding, bias=False)
        new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
        new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
        self.stem[0] = new_first_layer

    def features(self, x):
        x = self.stem(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        x = self.norm_pre(x)
        return x, out
    
    def classifier(self, features: torch.tensor):
        return None

    def forward(self, image, mask=None, *args, **kwargs):
        feature, out = self.features(image)
        return feature, out