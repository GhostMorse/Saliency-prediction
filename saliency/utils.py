"""Image helpers and small utilities shared across the project."""

from __future__ import annotations

import random

import cv2
import numpy as np

# Frames and maps are processed at this fixed size (H, W).
TARGET_HEIGHT = 216
TARGET_WIDTH = 384

# Number of videos to score in quick development runs (full test set is larger).
NUM_TEST_VIDS = 10


def padding(frame, height: int = TARGET_HEIGHT, width: int = TARGET_WIDTH):
    """Zero-pad a frame / map to ``(height, width)``, centring the content.

    Only adds borders; if the input is already at least that large it is
    returned unchanged (the original code never downscales).
    """
    if isinstance(frame, np.ndarray) is False:
        frame = np.asarray(frame)

    h, w = frame.shape[:2]
    if h == height and w == width:
        return frame

    top = max(0, (height - h) // 2)
    bottom = max(0, height - h - top)
    left = max(0, (width - w) // 2)
    right = max(0, width - w - left)

    return cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


def padding_fixation(fixation, height: int = TARGET_HEIGHT, width: int = TARGET_WIDTH):
    """Zero-pad a (binary) fixation map to ``(height, width)``."""
    return padding(fixation, height=height, width=width)


def normalize_map(s_map):
    """Min-max normalise a saliency map to ``[0, 1]``."""
    s_map = s_map.astype(np.float32)
    return (s_map - np.min(s_map)) / (np.max(s_map) - np.min(s_map) + 1e-7)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def count_parameters(model) -> int:
    """Number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
