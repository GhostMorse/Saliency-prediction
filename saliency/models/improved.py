"""Improved saliency model: EfficientNet-B4 encoder + ASPP + attention + skips.

This is the model that addresses the baseline's limitations: a stronger
multi-scale encoder (EfficientNet-B4), an ASPP bottleneck, channel attention in
the decoder, and skip connections from the encoder for sharper localisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ASPPModule, ChannelAttention

# Output channels of the 5 EfficientNet-B4 feature stages (timm, features_only).
ENCODER_CHANNELS = [24, 32, 56, 160, 448]


class ImprovedSaliencyModel(nn.Module):
    """EfficientNet-B4 encoder-decoder with ASPP, channel attention and skips.

    :param use_skip_connections: Concatenate reduced encoder features in the decoder.
    :param use_attention: Apply channel attention after each decoder stage.
    :param dropout_rate: Dropout in the decoder blocks.
    :param pretrained: Load ImageNet weights for the backbone.
    """

    def __init__(self, use_skip_connections: bool = True, use_attention: bool = True,
                 dropout_rate: float = 0.1, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.use_skip_connections = use_skip_connections
        self.use_attention = use_attention

        self.backbone = timm.create_model("efficientnet_b4", pretrained=pretrained, features_only=True)
        self.aspp = ASPPModule(ENCODER_CHANNELS[-1], 256, [6, 12, 18])

        if use_skip_connections:
            self.skip_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ENCODER_CHANNELS[i], ENCODER_CHANNELS[i] // 4, 1, bias=False),
                    nn.BatchNorm2d(ENCODER_CHANNELS[i] // 4), nn.ReLU(inplace=True),
                )
                for i in range(len(ENCODER_CHANNELS) - 1)
            ])
            skip_reduced = [ch // 4 for ch in ENCODER_CHANNELS[:-1]]
        else:
            skip_reduced = [0, 0, 0, 0]

        if use_attention:
            self.attention_modules = nn.ModuleList(
                [ChannelAttention(128), ChannelAttention(64), ChannelAttention(32), ChannelAttention(16)]
            )

        self.dec1 = self._decoder_block(256, 128, dropout_rate)
        self.dec2 = self._decoder_block(128 + skip_reduced[2], 64, dropout_rate)
        self.dec3 = self._decoder_block(64 + skip_reduced[1], 32, dropout_rate)
        self.dec4 = self._decoder_block(32 + skip_reduced[0], 16, dropout_rate)

        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )

    @staticmethod
    def _decoder_block(in_channels: int, out_channels: int, dropout_rate: float) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        return nn.Sequential(*layers)

    @staticmethod
    def _minmax_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        shape = x.shape
        x = x.view(shape[0], -1)
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        x = (x - x_min) / (x_max - x_min + eps)
        return x.view(shape)

    def _fuse_skip(self, x: torch.Tensor, skip_features, level: int) -> torch.Tensor:
        if self.use_skip_connections and len(skip_features) > level:
            skip = self.skip_adapters[level](skip_features[level])
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        features = self.backbone(x)
        skip_features = features[:-1]  # 4 maps: [24, 32, 56, 160] channels
        x = self.aspp(features[-1])

        x = self.dec1(x)
        if self.use_attention:
            x = self.attention_modules[0](x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self._fuse_skip(x, skip_features, 2)

        x = self.dec2(x)
        if self.use_attention:
            x = self.attention_modules[1](x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self._fuse_skip(x, skip_features, 1)

        x = self.dec3(x)
        if self.use_attention:
            x = self.attention_modules[2](x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self._fuse_skip(x, skip_features, 0)

        x = self.dec4(x)
        if self.use_attention:
            x = self.attention_modules[3](x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)

        x = self.final_conv(x)
        return self._minmax_normalize(x)
