"""Implémentation du filtre One-Euro pour signaux scalaires et vectoriels.

Référence : Casiez, Roussel, Vogel, "1€ Filter: A Simple Speed-based Low-pass
Filter for Noisy Input in Interactive Systems", CHI 2012.
https://gery.casiez.net/1euro/

API :
    OneEuroFilter         : filtre scalaire (un seul signal continu)
    OneEuroFilterND       : filtre vectoriel (N composantes indépendantes)
    smooth_signal         : helper qui filtre une série temporelle complète
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    """Facteur alpha d'un low-pass exponentiel pour un cutoff donné."""
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


def _exponential_smoothing(alpha: float, x: float, x_prev: float) -> float:
    return alpha * x + (1.0 - alpha) * x_prev


@dataclass
class OneEuroFilter:
    """Filtre One-Euro scalaire.

    Args:
        freq        : fréquence d'échantillonnage attendue (Hz). Sert à initialiser
                      le pas de temps si l'utilisateur ne fournit pas de timestamps.
        min_cutoff  : fréquence de coupure minimale (Hz). Plus bas = plus lisse.
        beta        : coefficient de réactivité à la vitesse. Plus haut = moins de lag.
        d_cutoff    : cutoff du filtre dérivée. Laisser à 1.0 par défaut.

    Procédure de tuning recommandée par les auteurs :
        1. Fixer beta=0. Avec entrée immobile, baisser min_cutoff jusqu'à
           éliminer le jitter perçu.
        2. Avec min_cutoff fixé, faire des mouvements rapides. Augmenter beta
           jusqu'à éliminer le retard.
    """
    freq: float = 30.0
    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def filter(self, x: float, t: float | None = None) -> float:
        """Filtre une nouvelle valeur. Retourne la valeur lissée.

        Args:
            x : valeur brute à l'instant t
            t : timestamp en secondes. Si None, utilise 1/freq comme pas.
        """
        if self._x_prev is None:
            self._x_prev = float(x)
            self._t_prev = 0.0 if t is None else float(t)
            return float(x)

        if t is None:
            t_e = 1.0 / self.freq
            new_t = self._t_prev + t_e
        else:
            t = float(t)
            t_e = max(t - self._t_prev, 1e-6)
            new_t = t

        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self._dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self._x_prev)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = new_t
        return x_hat


@dataclass
class OneEuroFilterND:
    """Filtre One-Euro N-dimensionnel : N filtres scalaires indépendants.

    Pour les rotations en quaternion, prévoir une étape de hemisphere
    consistency en amont (cf. smooth_quaternions dans smoothing.py) et une
    re-normalisation en aval.
    """
    n: int
    freq: float = 30.0
    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        self._filters: list[OneEuroFilter] = [
            OneEuroFilter(self.freq, self.min_cutoff, self.beta, self.d_cutoff)
            for _ in range(self.n)
        ]

    def reset(self) -> None:
        for f in self._filters:
            f.reset()

    def filter(self, x: np.ndarray, t: float | None = None) -> np.ndarray:
        """x : np.ndarray de shape (n,). Retourne le vecteur lissé."""
        assert x.shape == (self.n,), f"Attendu shape ({self.n},), reçu {x.shape}"
        out = np.empty(self.n, dtype=np.float64)
        for i, f in enumerate(self._filters):
            out[i] = f.filter(float(x[i]), t)
        return out


def smooth_signal(
    signal: np.ndarray,
    freq: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float = 1.0,
) -> np.ndarray:
    """Lisse une série temporelle multi-dimensionnelle frame par frame.

    Args:
        signal     : np.ndarray de shape (T, ...) — la première dim est le temps,
                     les dims suivantes sont vectorisées et filtrées indépendamment.
        freq       : fps du signal (Hz)
        min_cutoff : Hz
        beta       : coefficient de vitesse
        d_cutoff   : Hz pour le filtre de dérivée

    Returns:
        np.ndarray de même shape que signal, dtype float32.
    """
    if signal.size == 0 or signal.shape[0] < 2:
        return signal.astype(np.float32, copy=False)

    T = signal.shape[0]
    flat = signal.reshape(T, -1).astype(np.float64)
    n_dim = flat.shape[1]

    flt = OneEuroFilterND(n=n_dim, freq=freq, min_cutoff=min_cutoff,
                          beta=beta, d_cutoff=d_cutoff)
    out_flat = np.empty_like(flat)
    for t in range(T):
        out_flat[t] = flt.filter(flat[t], t=t / freq)

    return out_flat.reshape(signal.shape).astype(np.float32)
