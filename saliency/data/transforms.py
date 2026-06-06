"""Image normalisation transforms (ImageNet statistics)."""

from __future__ import annotations

import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# uint8 HWC frame -> normalised float CHW tensor.
image_transform = T.Compose([T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

# Invert the normalisation (for visualisation).
denormalize = T.Normalize(
    mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1.0 / s for s in IMAGENET_STD],
)
