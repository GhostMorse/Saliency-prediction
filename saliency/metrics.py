"""Saliency evaluation metrics: SIM, CC, NSS, plus a dataset-level aggregator."""

from __future__ import annotations

import os
from typing import Dict, Optional

import cv2
import numpy as np

from .utils import normalize_map


def similarity(s_map: np.ndarray, gt: np.ndarray) -> float:
    """Histogram intersection (SIM) of two maps, each normalised to sum to 1."""
    s_map = s_map / (np.sum(s_map) + 1e-7)
    gt = gt / (np.sum(gt) + 1e-7)
    return float(np.sum(np.minimum(s_map, gt)))


def cc(s_map: np.ndarray, gt: np.ndarray) -> float:
    """Pearson correlation coefficient (CC) between two maps."""
    a = (s_map - np.mean(s_map)) / (np.std(s_map) + 1e-7)
    b = (gt - np.mean(gt)) / (np.std(gt) + 1e-7)
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-7))


def nss(s_map: np.ndarray, fixations: np.ndarray) -> float:
    """Normalized Scanpath Saliency (NSS): mean of the z-scored saliency at
    fixated locations. ``fixations`` is a binary (or non-zero) fixation map."""
    ys, xs = np.where(fixations > 0)
    if len(xs) == 0:
        return 0.0
    s_map_norm = (s_map - np.mean(s_map)) / (np.std(s_map) + 1e-7)
    return float(np.mean([s_map_norm[y, x] for y, x in zip(ys, xs)]))


def calculate_metrics(predictions_dir: str, gt_dir: str, num_videos: Optional[int] = None) -> Dict[str, float]:
    """Average SIM / CC / NSS over all predicted frames of all videos.

    For each video folder under ``predictions_dir`` it reads each ``NNNN.png``
    prediction and compares it with the ground-truth saliency at
    ``gt_dir/<video>/NNNN.png`` (SIM, CC) and the fixation map at
    ``gt_dir/<video>/fixations/NNNN.png`` (NSS). Ground-truth maps are resized
    to the prediction size if needed.
    """
    videos = sorted(os.listdir(predictions_dir))
    if num_videos is not None:
        videos = videos[:num_videos]

    sims, ccs, nsss = [], [], []
    for video in videos:
        pred_dir = os.path.join(predictions_dir, video)
        gt_video = os.path.join(gt_dir, video)
        if not os.path.isdir(pred_dir):
            continue

        for fname in sorted(os.listdir(pred_dir)):
            if not fname.endswith(".png"):
                continue
            pred = cv2.imread(os.path.join(pred_dir, fname), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(os.path.join(gt_video, fname), cv2.IMREAD_GRAYSCALE)
            if pred is None or gt is None:
                continue
            pred = pred.astype(np.float32)
            gt = gt.astype(np.float32)
            if gt.shape != pred.shape:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))

            sims.append(similarity(normalize_map(pred), normalize_map(gt)))
            ccs.append(cc(pred, gt))

            fix = cv2.imread(os.path.join(gt_video, "fixations", fname), cv2.IMREAD_GRAYSCALE)
            if fix is not None:
                if fix.shape != pred.shape:
                    fix = cv2.resize(fix, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                nsss.append(nss(pred, fix))

    return {
        "SIM": float(np.mean(sims)) if sims else 0.0,
        "CC": float(np.mean(ccs)) if ccs else 0.0,
        "NSS": float(np.mean(nsss)) if nsss else 0.0,
    }


def compute_validation_metrics(pred, target) -> Dict[str, float]:
    """Batched SIM / CC / KLD over ``(B, 1, H, W)`` tensors or arrays."""
    pred_np = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
    target_np = target.detach().cpu().numpy() if hasattr(target, "detach") else np.asarray(target)

    def kld(s_map, gt, eps=1e-7):
        s_map = s_map / (np.sum(s_map) + eps)
        gt = gt / (np.sum(gt) + eps)
        return float(np.sum(gt * np.log(gt / (s_map + eps) + eps)))

    sims, ccs, klds = [], [], []
    for i in range(pred_np.shape[0]):
        p, t = pred_np[i, 0], target_np[i, 0]
        sims.append(similarity(p, t))
        ccs.append(cc(p, t))
        klds.append(kld(p, t))
    return {"sim": float(np.mean(sims)), "cc": float(np.mean(ccs)), "kld": float(np.mean(klds))}
