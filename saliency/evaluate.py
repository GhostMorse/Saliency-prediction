"""Run a trained model over a video and save per-frame saliency predictions."""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from .data.transforms import image_transform
from .utils import TARGET_HEIGHT, TARGET_WIDTH, padding


class SaliencyEvaluator:
    """Predict and save a saliency PNG for every frame of a video.

    :param model: A model in eval mode (already on ``device``).
    :param device: Torch device for inference.
    :param max_frames: Maximum number of frames to process per video.
    """

    def __init__(self, model, device: torch.device, max_frames: int = 300) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.max_frames = max_frames

    @classmethod
    def from_checkpoint(cls, model, checkpoint_path: str, device: torch.device, max_frames: int = 300):
        """Load a state-dict checkpoint into ``model`` and wrap it."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        return cls(model, device, max_frames)

    @torch.no_grad()
    def evaluate(self, video_path: str, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        for frame_num in range(1, self.max_frames + 1):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = padding(frame, TARGET_HEIGHT, TARGET_WIDTH)
            frame_t = image_transform(np.ascontiguousarray(frame)).unsqueeze(0).to(self.device)

            pred = self.model(frame_t)[0, 0].cpu().numpy()
            pred = (pred * 255).astype("uint8")
            cv2.imwrite(os.path.join(output_dir, f"{str(frame_num).zfill(4)}.png"), pred)
        cap.release()
