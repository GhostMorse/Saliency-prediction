"""Tests for the SIM / CC / NSS saliency metrics."""

import numpy as np

from saliency.metrics import cc, nss, similarity


def test_similarity_identical_is_one():
    s = np.abs(np.random.rand(32, 48)).astype(np.float32)
    assert abs(similarity(s, s) - 1.0) < 1e-4


def test_similarity_in_unit_range():
    a = np.random.rand(16, 16)
    b = np.random.rand(16, 16)
    assert 0.0 <= similarity(a, b) <= 1.0


def test_cc_identical_is_one():
    s = np.random.rand(32, 32)
    assert abs(cc(s, s) - 1.0) < 1e-3


def test_cc_anticorrelated_is_negative():
    s = np.random.rand(32, 32)
    assert cc(s, -s) < 0


def test_nss_positive_when_fixations_hit_peaks():
    s = np.zeros((10, 10), dtype=np.float32)
    s[5, 5] = 10.0  # a salient peak
    fix = np.zeros((10, 10), dtype=np.uint8)
    fix[5, 5] = 1  # fixation on the peak
    assert nss(s, fix) > 0


def test_nss_zero_without_fixations():
    s = np.random.rand(8, 8)
    assert nss(s, np.zeros((8, 8), dtype=np.uint8)) == 0.0
