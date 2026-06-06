"""Tests for the saliency losses."""

import torch

from saliency.losses import CombinedSaliencyLoss, kld_loss


def test_kld_identical_is_near_zero():
    s = torch.rand(2, 1, 16, 16) + 1e-3
    assert kld_loss(s, s).item() < 1e-4


def test_kld_non_negative():
    pred = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16)
    assert kld_loss(pred, target).item() >= -1e-5


def test_combined_loss_returns_components():
    pred = torch.rand(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    total, components = CombinedSaliencyLoss()(pred, target)
    assert total.ndim == 0 and torch.isfinite(total)
    assert set(components) == {"mse", "ssim", "kl", "pearson"}


def test_combined_loss_respects_weight_subset():
    loss = CombinedSaliencyLoss(weights={"mse": 1.0})
    _, components = loss(torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16))
    assert set(components) == {"mse"}
