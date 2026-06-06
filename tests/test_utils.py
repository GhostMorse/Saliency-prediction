"""Tests for padding and saliency-map normalisation."""

import numpy as np

from saliency.utils import normalize_map, padding, padding_fixation


def test_padding_pads_small_frame_to_target():
    frame = np.ones((100, 200, 3), dtype=np.uint8)
    out = padding(frame, height=216, width=384)
    assert out.shape == (216, 384, 3)
    # Original content is centred; borders are zero.
    assert out.sum() == frame.sum()


def test_padding_leaves_exact_size_unchanged():
    frame = np.ones((216, 384), dtype=np.uint8)
    out = padding(frame, 216, 384)
    assert out.shape == (216, 384)


def test_padding_does_not_downscale_larger_input():
    frame = np.ones((300, 500), dtype=np.uint8)  # larger than target
    out = padding(frame, 216, 384)
    assert out.shape == (300, 500)  # returned unchanged, never cropped/scaled


def test_padding_fixation_matches_padding():
    fix = (np.random.rand(120, 150) > 0.99).astype(np.uint8)
    assert padding_fixation(fix).shape == (216, 384)


def test_normalize_map_to_unit_range():
    s = np.array([[0.0, 5.0], [10.0, 2.0]], dtype=np.float32)
    out = normalize_map(s)
    assert abs(out.min()) < 1e-5
    assert abs(out.max() - 1.0) < 1e-3
