"""Tests du filtre One-Euro."""
from __future__ import annotations

import numpy as np

from pipeline.one_euro_filter import (
    OneEuroFilter, OneEuroFilterND, smooth_signal,
)


def test_first_value_is_passthrough() -> None:
    f = OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.0)
    assert abs(f.filter(3.14) - 3.14) < 1e-9


def test_constant_signal_remains_constant() -> None:
    f = OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.0)
    for _ in range(50):
        out = f.filter(5.0)
    assert abs(out - 5.0) < 1e-6


def test_step_response_is_smooth() -> None:
    """Un saut brutal doit être lissé (pas instantané)."""
    f = OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.0)
    f.filter(0.0)
    out = f.filter(10.0)
    # Avec min_cutoff=1Hz et dt=1/30s, alpha < 1 → out << 10
    assert 0 < out < 10


def test_higher_min_cutoff_is_more_reactive() -> None:
    f_slow = OneEuroFilter(freq=30, min_cutoff=0.1, beta=0.0)
    f_fast = OneEuroFilter(freq=30, min_cutoff=10.0, beta=0.0)
    f_slow.filter(0.0); f_fast.filter(0.0)
    out_slow = f_slow.filter(10.0)
    out_fast = f_fast.filter(10.0)
    assert out_fast > out_slow  # fast filter capte plus du signal


def test_nd_filter_independent_components() -> None:
    flt = OneEuroFilterND(n=3, freq=30, min_cutoff=1.0, beta=0.0)
    out = flt.filter(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_almost_equal(out, [1.0, 2.0, 3.0])
    out2 = flt.filter(np.array([2.0, 4.0, 6.0]))
    # Composantes indépendantes : la 2e doit être ~2× la 1ère, etc.
    assert out2[1] == 2 * out2[0]
    assert out2[2] == 3 * out2[0]


def test_smooth_signal_preserves_shape() -> None:
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((100, 5, 3)).astype(np.float32)
    out = smooth_signal(sig, freq=30, min_cutoff=1.0, beta=0.1)
    assert out.shape == sig.shape
    assert out.dtype == np.float32


def test_smooth_signal_reduces_variance() -> None:
    """Sur un signal bruité autour d'une constante, le lissage réduit la variance."""
    rng = np.random.default_rng(123)
    n = 200
    truth = np.zeros((n, 1))
    noise = rng.standard_normal((n, 1)) * 0.5
    sig = (truth + noise).astype(np.float32)
    out = smooth_signal(sig, freq=30, min_cutoff=0.5, beta=0.0)
    # Skip warm-up des premières frames
    assert np.var(out[20:]) < np.var(sig[20:])


def test_smooth_signal_short_input() -> None:
    """Avec moins de 2 frames, on retourne le signal tel quel."""
    sig = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = smooth_signal(sig, freq=30, min_cutoff=1.0, beta=0.1)
    np.testing.assert_array_equal(out, sig)
