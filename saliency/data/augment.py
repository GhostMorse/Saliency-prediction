"""Training-time augmentations that preserve the saliency target geometry.

Spatial transforms (a horizontal flip) are applied jointly to the frame and its
saliency map; photometric transforms are applied to the frame only. Be careful
adding spatial transforms: they must keep the attention regions valid.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class AdvancedAugmentation:
    """Albumentations-based augmentation over a ``(frame_uint8, saliency_uint8)`` pair."""

    def __init__(self, enable: bool = True, horizontal_flip: bool = True,
                 brightness_contrast: bool = True, hue_saturation: bool = True, gaussian_noise: bool = True) -> None:
        self.enable = enable
        self.spatial = None
        self.pixel = None
        if not enable:
            return

        import albumentations as A

        spatial = [A.HorizontalFlip(p=0.5)] if horizontal_flip else []
        self.spatial = A.Compose(spatial) if spatial else None

        pixel = []
        if brightness_contrast:
            pixel.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
        if hue_saturation:
            pixel.append(A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3))
        if gaussian_noise:
            pixel.append(A.GaussNoise(p=0.2))
        self.pixel = A.Compose(pixel) if pixel else None

    def __call__(self, frame: np.ndarray, saliency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.enable:
            if self.spatial is not None:
                out = self.spatial(image=frame, mask=saliency)
                frame, saliency = out["image"], out["mask"]
            if self.pixel is not None:
                frame = self.pixel(image=frame)["image"]
        return frame, saliency


def mixup_data(frames: torch.Tensor, saliency: torch.Tensor, alpha: float = 0.2):
    """Convex combination of a batch with a shuffled copy of itself (mixup)."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(frames.size(0), device=frames.device)
    mixed_frames = lam * frames + (1 - lam) * frames[index]
    mixed_saliency = lam * saliency + (1 - lam) * saliency[index]
    return mixed_frames, mixed_saliency
