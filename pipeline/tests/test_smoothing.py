"""Tests du pipeline de lissage complet sur une Animation."""
from __future__ import annotations

import numpy as np

from pipeline.animation_npz import make_empty
from pipeline.smoothing import (
    SmoothingParams, _axis_angle_to_quat, _quat_to_axis_angle, smooth_animation,
)


def test_axis_angle_quat_roundtrip() -> None:
    """Conversion axis-angle ↔ quaternion ne perd pas d'information."""
    rng = np.random.default_rng(42)
    aa = rng.standard_normal((100, 3)) * 0.5  # angles raisonnables
    q = _axis_angle_to_quat(aa)
    aa_back = _quat_to_axis_angle(q)
    # Précision : modulo 2π près, et signe d'axis-angle peut flipper
    np.testing.assert_array_almost_equal(aa_back, aa, decimal=5)


def test_zero_axis_angle_quat() -> None:
    """axis-angle nul → quaternion identité (0, 0, 0, 1)."""
    q = _axis_angle_to_quat(np.zeros((1, 3)))
    np.testing.assert_array_almost_equal(q, [[0, 0, 0, 1]])


def test_smooth_animation_preserves_shapes() -> None:
    anim = make_empty(num_frames=30, fps=30.0,
                      source_video="x.mp4", source_fps=30.0)
    rng = np.random.default_rng(0)
    anim.transl[:] = rng.standard_normal(anim.transl.shape) * 0.1
    anim.body_pose[:] = rng.standard_normal(anim.body_pose.shape) * 0.1

    params = SmoothingParams(fps=30.0)
    out = smooth_animation(anim, params)

    assert out.transl.shape == anim.transl.shape
    assert out.body_pose.shape == anim.body_pose.shape
    assert out.left_hand_pose.shape == anim.left_hand_pose.shape
    assert out.expression.shape == anim.expression.shape
    out.validate()


def test_smooth_animation_reduces_jitter() -> None:
    """Sur des données bruitées, le lissage réduit la variance frame-à-frame."""
    rng = np.random.default_rng(7)
    anim = make_empty(num_frames=60, fps=30.0,
                      source_video="x.mp4", source_fps=30.0)
    anim.transl[:] = rng.standard_normal(anim.transl.shape) * 0.5

    params = SmoothingParams(fps=30.0, transl_min_cutoff=0.5, transl_beta=0.0)
    out = smooth_animation(anim, params)

    # Variance frame-à-frame (différences successives)
    diff_in  = np.diff(anim.transl, axis=0)
    diff_out = np.diff(out.transl, axis=0)
    assert np.var(diff_out) < np.var(diff_in)


def test_smooth_animation_short_passthrough() -> None:
    """Avec T<2, smooth_animation retourne l'animation telle quelle."""
    anim = make_empty(num_frames=1, fps=30.0,
                      source_video="x.mp4", source_fps=30.0)
    out = smooth_animation(anim, SmoothingParams(fps=30.0))
    assert out is anim
