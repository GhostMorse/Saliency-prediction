"""Output-shape tests for both models (constructed without pretrained weights)."""

import torch

from saliency.models import ImprovedSaliencyModel, SaliencyModel


def test_baseline_output_shape_and_range():
    model = SaliencyModel(pretrained=False).eval()
    x = torch.rand(1, 3, 96, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 96, 128)
    assert out.min() >= 0.0 and out.max() <= 1.0  # sigmoid output


def test_improved_output_shape_and_range():
    model = ImprovedSaliencyModel(pretrained=False).eval()
    x = torch.rand(1, 3, 96, 160)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 96, 160)
    assert torch.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0  # min-max normalised


def test_improved_without_skips_or_attention():
    model = ImprovedSaliencyModel(use_skip_connections=False, use_attention=False, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 96, 160))
    assert out.shape == (1, 1, 96, 160)
