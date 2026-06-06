"""Train a saliency model on the video dataset.

Expected data layout (pass the parent via ``--data-root``)::

    <root>/01_test_file_input/{train,test}/<video>/source.mp4
    <root>/01_test_file_gt/{train,test}/<video>/{NNNN.png, fixations/NNNN.png}

Examples
--------
    # DeepLabV3 baseline (KL-divergence loss)
    python train.py --model baseline --data-root data/public_tests --epochs 5

    # improved EfficientNet-B4 model (combined loss)
    python train.py --model improved --data-root data/public_tests --epochs 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from saliency.data import InfiniteSampler, SaliencyFrameDataset
from saliency.engine import EarlyStopping, WarmupCosineLR, train
from saliency.losses import CombinedSaliencyLoss, kld_loss
from saliency.models import ImprovedSaliencyModel, SaliencyModel
from saliency.utils import count_parameters, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a video saliency model.")
    p.add_argument("--model", choices=["baseline", "improved"], default="baseline")
    p.add_argument("--data-root", type=Path, default=Path("data/public_tests"))
    p.add_argument("--output", type=Path, default=Path("saliency.pth"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--use-mixup", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    input_root, gt_root = args.data_root / "01_test_file_input", args.data_root / "01_test_file_gt"
    train_ds = SaliencyFrameDataset(str(input_root / "train"), str(gt_root / "train"), augment=not args.no_augment)
    val_ds = SaliencyFrameDataset(str(input_root / "test"), str(gt_root / "test"), augment=False)
    print(f"Train frames: {len(train_ds)} | Val frames: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=InfiniteSampler(train_ds),
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    pretrained = not args.no_pretrained
    if args.model == "baseline":
        model = SaliencyModel(pretrained=pretrained).to(device)
        criterion = kld_loss
    else:
        model = ImprovedSaliencyModel(pretrained=pretrained).to(device)
        criterion = CombinedSaliencyLoss().to(device)
    print(f"Model: {args.model} | trainable params: {count_parameters(model) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineLR(optimizer, base_lr=args.lr, total_epochs=args.epochs)
    early_stopping = EarlyStopping(patience=10, mode="min")

    train(
        model, train_loader, val_loader, optimizer, criterion, device,
        num_epochs=args.epochs, num_steps=args.num_steps, scheduler=scheduler,
        early_stopping=early_stopping, accumulation_steps=args.accumulation_steps,
        use_mixup=args.use_mixup, save_path=str(args.output),
    )
    print(f"Done. Best checkpoint at {args.output}")


if __name__ == "__main__":
    main()
