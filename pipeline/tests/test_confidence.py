"""Tests du calcul de confidence."""
from __future__ import annotations

import numpy as np

from pipeline.confidence import combine_confidence, low_confidence_mask


def test_combine_no_residuals() -> None:
    bbox = np.array([1.0, 0.5, 0.9], dtype=np.float32)
    out = combine_confidence(bbox, None)
    np.testing.assert_array_equal(out, bbox)


def test_combine_with_residuals() -> None:
    bbox = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    # Résidu 0 → score reproj = 1 ; bbox=1 → combine = 1
    res = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    out = combine_confidence(bbox, res)
    np.testing.assert_allclose(out, 1.0)


def test_combine_high_residual_lowers_score() -> None:
    bbox = np.array([1.0], dtype=np.float32)
    res_low  = np.array([0.0], dtype=np.float32)
    res_high = np.array([100.0], dtype=np.float32)
    out_low  = combine_confidence(bbox, res_low)
    out_high = combine_confidence(bbox, res_high)
    assert out_high[0] < out_low[0]


def test_combine_clips_to_unit_interval() -> None:
    bbox = np.array([1.5, -0.2], dtype=np.float32)  # hors range
    out = combine_confidence(bbox, None)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_low_confidence_mask() -> None:
    conf = np.array([0.3, 0.6, 0.4, 0.9, 0.5])
    mask = low_confidence_mask(conf, threshold=0.5)
    np.testing.assert_array_equal(mask, [True, False, True, False, True])
