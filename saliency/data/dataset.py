"""Video saliency dataset (one frame at a time, read with OpenCV).

Expected layout::

    <input_dir>/<video>/source.mp4
    <gt_dir>/<video>/NNNN.png            (per-frame ground-truth saliency)
    <gt_dir>/<video>/fixations/NNNN.png  (per-frame binary fixations, optional)

Frames are 1-indexed in the PNG filenames (``0001.png`` is frame 0).
"""

from __future__ import annotations

import itertools
import os
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..utils import TARGET_HEIGHT, TARGET_WIDTH, padding, padding_fixation
from .augment import AdvancedAugmentation
from .transforms import image_transform


class SaliencyFrameDataset(Dataset):
    """Flat dataset of ``(video, frame)`` pairs across all videos in a split."""

    def __init__(self, input_dir: str, gt_dir: str, augment: bool = False, max_frames: int = 300) -> None:
        self.augment = AdvancedAugmentation(enable=augment)
        self.samples = []  # (video_path, gt_path, frame_idx)

        for folder in sorted(os.listdir(input_dir)):
            input_path = os.path.join(input_dir, folder)
            gt_path = os.path.join(gt_dir, folder)
            video_path = os.path.join(input_path, "source.mp4")
            if not (os.path.isdir(input_path) and os.path.isdir(gt_path) and os.path.exists(video_path)):
                continue
            cap = cv2.VideoCapture(video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for idx in range(min(num_frames, max_frames)):
                self.samples.append((video_path, gt_path, idx))

        if not self.samples:
            raise RuntimeError(f"No (video, frame) samples found under {input_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_frame(video_path: str, frame_idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return padding(frame, TARGET_HEIGHT, TARGET_WIDTH)

    @staticmethod
    def _read_map(directory: str, frame_idx: int, fixation: bool = False):
        path = os.path.join(directory, f"{str(frame_idx + 1).zfill(4)}.png")
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        return padding_fixation(gray) if fixation else padding(gray)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_path, gt_path, frame_idx = self.samples[idx]
        frame = self._read_frame(video_path, frame_idx)
        saliency = self._read_map(gt_path, frame_idx)
        if saliency is None:
            saliency = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)

        frame, saliency = self.augment(frame, saliency)

        frame_t = image_transform(np.ascontiguousarray(frame))
        saliency_t = torch.from_numpy(saliency.astype(np.float32) / 255.0).unsqueeze(0)

        fix = self._read_map(os.path.join(gt_path, "fixations"), frame_idx, fixation=True)
        fix_t = torch.from_numpy((fix > 0).astype(np.float32)).unsqueeze(0) if fix is not None else torch.zeros_like(saliency_t)

        return frame_t, saliency_t, fix_t


class InfiniteSampler(Sampler):
    """Yields an endless stream of shuffled indices (for step-based training)."""

    def __init__(self, data_source) -> None:
        self.size = len(data_source)

    def __iter__(self):
        def _gen():
            while True:
                yield from torch.randperm(self.size).tolist()

        return itertools.islice(_gen(), 0, None)

    def __len__(self) -> int:
        return self.size
