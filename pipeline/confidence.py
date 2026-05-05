"""Calcul des arrays de confidence pour l'animation NPZ.

SMPLer-X ne fournit pas de score de confidence natif. On assemble une proxy
à partir de :
    - score de bbox du détecteur (mmdet) par région (corps / mains / visage)
    - 1 - distance(reprojection 2D, keypoints 2D détecteur) normalisée

Cf. docs/PIPELINE.md §3.2 et §10.5.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def combine_confidence(
    bbox_scores: np.ndarray,
    reprojection_residuals: np.ndarray | None,
    bbox_score_weight: float = 0.5,
    reprojection_weight: float = 0.5,
    residual_normalizer: float = 50.0,
) -> np.ndarray:
    """Combine bbox scores et résidus de reprojection en un score ∈ [0, 1].

    Args:
        bbox_scores            : (T,) scores ∈ [0, 1] du détecteur ;
                                 0 si la région n'a pas été détectée à cette frame.
        reprojection_residuals : (T,) erreur moyenne de reprojection en pixels,
                                 ou None si non disponible (alors le score est
                                 entièrement piloté par bbox_scores).
        bbox_score_weight      : poids du score bbox dans le score combiné
        reprojection_weight    : poids du résidu dans le score combiné
        residual_normalizer    : un résidu de N pixels donne un score reproj. = 0
                                 (clippé). 50 px est un défaut raisonnable pour
                                 vidéo 720p.

    Returns:
        np.ndarray (T,) float32 ∈ [0, 1].
    """
    bbox_scores = np.asarray(bbox_scores, dtype=np.float32)
    bbox_scores = np.clip(bbox_scores, 0.0, 1.0)

    if reprojection_residuals is None:
        return bbox_scores

    residuals = np.asarray(reprojection_residuals, dtype=np.float32)
    # Score reproj : 1 si parfait, 0 si >= residual_normalizer.
    reproj_score = np.clip(1.0 - residuals / residual_normalizer, 0.0, 1.0)

    total_weight = bbox_score_weight + reprojection_weight
    if total_weight <= 0:
        raise ValueError("La somme des poids confidence doit être > 0")

    combined = (bbox_score_weight * bbox_scores + reprojection_weight * reproj_score) / total_weight
    return np.clip(combined, 0.0, 1.0).astype(np.float32)


def low_confidence_mask(
    confidence: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Retourne un masque booléen (T,) marquant les frames sous le seuil."""
    return np.asarray(confidence) < threshold
