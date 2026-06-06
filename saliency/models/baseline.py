"""Baseline saliency model: a DeepLabV3-ResNet50 encoder with a custom decoder.

Transfer learning from a segmentation network (as the dataset is small): the
DeepLabV3 backbone + ASPP produce stride-8 features, which a lightweight decoder
upsamples back to full resolution and reduces to a single saliency channel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SaliencyModel(nn.Module):
    """DeepLabV3-ResNet50 encoder + progressive-upsampling decoder."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        seg = torchvision.models.segmentation
        if pretrained:
            deeplab = seg.deeplabv3_resnet50(weights=seg.DeepLabV3_ResNet50_Weights.DEFAULT)
        else:
            # weights_backbone=None too, so nothing is downloaded.
            deeplab = seg.deeplabv3_resnet50(weights=None, weights_backbone=None)

        self.backbone = deeplab.backbone          # -> {'out': 2048 channels, stride 8}
        self.aspp = deeplab.classifier[0]         # ASPP: 2048 -> 256

        self.decoder = nn.Sequential(
            _conv_block(256, 128), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            _conv_block(128, 64), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            _conv_block(64, 32), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        features = self.backbone(x)["out"]
        x = self.aspp(features)
        x = self.decoder(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return torch.sigmoid(x)
