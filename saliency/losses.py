"""Loss functions for saliency prediction."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def kld_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Kullback-Leibler divergence between predicted and ground-truth maps.

    Each map is normalised to a probability distribution (divided by its sum)
    before the divergence is computed and averaged over the batch.

    :param y_pred: ``(B, 1, H, W)`` predicted saliency in ``[0, 1]``.
    :param y_true: ``(B, 1, H, W)`` ground-truth saliency.
    """
    y_true = y_true / (eps + torch.sum(y_true, dim=(1, 2, 3), keepdim=True))
    y_pred = y_pred / (eps + torch.sum(y_pred, dim=(1, 2, 3), keepdim=True))
    loss = y_true * torch.log(eps + y_true / (eps + y_pred))
    return torch.mean(torch.sum(loss, dim=(1, 2, 3)))


class CombinedSaliencyLoss(nn.Module):
    """Weighted combination of MSE, (1 - SSIM), KL divergence and (1 - Pearson).

    ``forward`` returns ``(total_loss, components)`` where ``components`` maps
    each term name to its scalar value (for logging).
    """

    def __init__(self, weights: Dict[str, float] = None) -> None:
        super().__init__()
        self.weights = weights or {"mse": 1.0, "ssim": 0.5, "kl": 0.3, "pearson": 0.2}
        self.mse = nn.MSELoss()

    def ssim_loss(self, pred, target, window_size: int = 11):
        c1, c2 = 0.01**2, 0.03**2
        pad = window_size // 2
        mu_p = F.avg_pool2d(pred, window_size, 1, pad)
        mu_t = F.avg_pool2d(target, window_size, 1, pad)
        mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p * mu_t
        sigma_p = F.avg_pool2d(pred**2, window_size, 1, pad) - mu_p2
        sigma_t = F.avg_pool2d(target**2, window_size, 1, pad) - mu_t2
        sigma_pt = F.avg_pool2d(pred * target, window_size, 1, pad) - mu_pt
        ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / ((mu_p2 + mu_t2 + c1) * (sigma_p + sigma_t + c2))
        return 1 - ssim_map.mean()

    def kl_divergence(self, pred, target, eps: float = 1e-10):
        pred = (pred + eps) / (pred + eps).sum(dim=(2, 3), keepdim=True)
        target = (target + eps) / (target + eps).sum(dim=(2, 3), keepdim=True)
        return (target * torch.log(target / pred)).sum(dim=(2, 3)).mean()

    def pearson_loss(self, pred, target):
        p = pred.view(pred.size(0), -1)
        t = target.view(target.size(0), -1)
        p = p - p.mean(dim=1, keepdim=True)
        t = t - t.mean(dim=1, keepdim=True)
        corr = (p * t).sum(dim=1) / (torch.sqrt((p**2).sum(dim=1)) * torch.sqrt((t**2).sum(dim=1)) + 1e-8)
        return 1 - corr.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = pred.new_zeros(())
        components: Dict[str, float] = {}
        terms = {"mse": lambda: self.mse(pred, target), "ssim": lambda: self.ssim_loss(pred, target),
                 "kl": lambda: self.kl_divergence(pred, target), "pearson": lambda: self.pearson_loss(pred, target)}
        for name, fn in terms.items():
            if name in self.weights:
                value = fn()
                total = total + self.weights[name] * value
                components[name] = value.item()
        return total, components
