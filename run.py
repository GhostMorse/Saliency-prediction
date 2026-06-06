"""Evaluation entry point: predict saliency for the test split and score it.

Mirrors the grading call ``python run.py <data_root>``: it runs the model over
every test video, writes predictions to ``outputs/<video>/NNNN.png``, and prints
the SIM / CC / NSS metrics.

    python run.py data/public_tests --model improved --checkpoint saliency.pth
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from saliency.evaluate import SaliencyEvaluator
from saliency.metrics import calculate_metrics
from saliency.models import ImprovedSaliencyModel, SaliencyModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict and score saliency on the test split.")
    p.add_argument("data_root", type=Path, help="Dataset root with 01_test_file_input / 01_test_file_gt")
    p.add_argument("--model", choices=["baseline", "improved"], default="baseline")
    p.add_argument("--checkpoint", type=Path, default=Path("saliency.pth"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--num-videos", type=int, default=None, help="Limit number of videos (default: all)")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    input_dir = args.data_root / "01_test_file_input" / "test"
    gt_dir = args.data_root / "01_test_file_gt" / "test"

    model = SaliencyModel(pretrained=False) if args.model == "baseline" else ImprovedSaliencyModel(pretrained=False)
    evaluator = SaliencyEvaluator.from_checkpoint(model, str(args.checkpoint), device)

    videos = sorted(os.listdir(input_dir))
    if args.num_videos is not None:
        videos = videos[: args.num_videos]
    for video in tqdm(videos, desc="predicting"):
        evaluator.evaluate(str(input_dir / video / "source.mp4"), str(args.output_dir / video))

    metrics = calculate_metrics(str(args.output_dir), str(gt_dir), num_videos=args.num_videos)
    print("\n=== Metrics ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
