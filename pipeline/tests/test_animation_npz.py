"""Tests du format animation.npz."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.animation_npz import (
    NUM_BODY_JOINTS, NUM_HAND_JOINTS, Animation, make_empty,
)


def test_make_empty_validates() -> None:
    anim = make_empty(num_frames=10, fps=30.0,
                      source_video="x.mp4", source_fps=30.0)
    anim.validate()
    assert anim.num_frames == 10
    assert anim.fps == 30.0


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    anim = make_empty(num_frames=5, fps=24.0,
                      source_video="vid.mp4", source_fps=24.0)
    # Mute random data
    rng = np.random.default_rng(42)
    anim.transl = rng.standard_normal(anim.transl.shape).astype(np.float32)
    anim.body_pose = rng.standard_normal(anim.body_pose.shape).astype(np.float32) * 0.1
    anim.confidence_body = rng.uniform(0, 1, anim.confidence_body.shape).astype(np.float32)
    anim.confidence_lhand = rng.uniform(0, 1, anim.confidence_lhand.shape).astype(np.float32)
    anim.confidence_rhand = rng.uniform(0, 1, anim.confidence_rhand.shape).astype(np.float32)
    anim.confidence_face = rng.uniform(0, 1, anim.confidence_face.shape).astype(np.float32)

    out_path = tmp_path / "test.npz"
    anim.save(out_path)
    assert out_path.exists()

    loaded = Animation.load(out_path)
    np.testing.assert_array_equal(loaded.transl, anim.transl)
    np.testing.assert_array_equal(loaded.body_pose, anim.body_pose)
    assert loaded.fps == anim.fps
    assert loaded.source_video == anim.source_video


def test_validate_rejects_wrong_shape() -> None:
    anim = make_empty(num_frames=3, fps=30.0,
                      source_video="x.mp4", source_fps=30.0)
    anim.body_pose = np.zeros((3, 20, 3), dtype=np.float32)  # 20 ≠ 21
    with pytest.raises(AssertionError, match="body_pose"):
        anim.validate()


def test_validate_rejects_nan() -> None:
    anim = make_empty(num_frames=3, fps=30.0,
                      source_video="x.mp4", source_fps=30.0)
    anim.transl[0, 0] = np.nan
    with pytest.raises(AssertionError, match="NaN/Inf"):
        anim.validate()


def test_validate_rejects_invalid_confidence() -> None:
    anim = make_empty(num_frames=3, fps=30.0,
                      source_video="x.mp4", source_fps=30.0)
    anim.confidence_body[0] = 1.5  # > 1
    with pytest.raises(AssertionError, match="confidence_body"):
        anim.validate()


def test_validate_rejects_invalid_meta_json() -> None:
    anim = make_empty(num_frames=2, fps=30.0,
                      source_video="x.mp4", source_fps=30.0)
    anim.meta_json = "not json"
    with pytest.raises(AssertionError, match="meta_json"):
        anim.validate()


def test_with_meta_merges() -> None:
    anim = make_empty(num_frames=2, fps=30.0,
                      source_video="x.mp4", source_fps=30.0)
    anim.meta_json = json.dumps({"a": 1})
    new = anim.with_meta(b=2, c=3)
    meta = json.loads(new.meta_json)
    assert meta == {"a": 1, "b": 2, "c": 3}


def test_shape_constants() -> None:
    assert NUM_BODY_JOINTS == 21
    assert NUM_HAND_JOINTS == 15
