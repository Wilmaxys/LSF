"""Tests des helpers vidéo (sans dépendance à un fichier vidéo réel)."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.video_io import resample_indices


def test_resample_identity() -> None:
    """fps_in = fps_out → indices = 0..N-1."""
    idx = resample_indices(num_frames_in=30, fps_in=30, fps_out=30)
    np.testing.assert_array_equal(idx, np.arange(30))


def test_resample_half_speed() -> None:
    """60 fps → 30 fps : un index sur deux."""
    idx = resample_indices(num_frames_in=60, fps_in=60, fps_out=30)
    assert len(idx) == 30
    np.testing.assert_array_equal(idx, np.arange(30) * 2)


def test_resample_double_speed() -> None:
    """30 fps → 60 fps : on duplique."""
    idx = resample_indices(num_frames_in=30, fps_in=30, fps_out=60)
    assert len(idx) == 60


def test_resample_clamps_to_last_index() -> None:
    """Aucun index ne dépasse num_frames_in - 1."""
    idx = resample_indices(num_frames_in=10, fps_in=30, fps_out=120)
    assert idx.max() < 10


def test_resample_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        resample_indices(0, 30, 30)
    with pytest.raises(ValueError):
        resample_indices(30, 0, 30)
    with pytest.raises(ValueError):
        resample_indices(30, 30, 0)
