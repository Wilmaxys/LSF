"""Mapping FLAME → expressions VRM.

Cf. docs/PIPELINE.md §5.

Stratégie v1 (la seule implémentée) : mapping géométrique via les bones VRM
`jaw`, `leftEye`, `rightEye`. Pas de mapping sémantique des expressions PCA
FLAME vers les presets émotionnels VRM (`happy`, `sad`, etc.) ; ce mapping
n'est pas canonique et reste à concevoir si nécessaire en v2.

Les 17 presets standard VRM 1.0 :
    - Émotions  : happy, angry, sad, relaxed, surprised
    - Voyelles  : aa, ih, ou, ee, oh
    - Cligne    : blink, blinkLeft, blinkRight
    - Regard    : lookUp, lookDown, lookLeft, lookRight
    - Autre     : neutral
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Liste canonique des presets VRM 1.0. Source : VRMC_vrm-1.0/expressions.md.
VRM_PRESET_EXPRESSIONS: tuple[str, ...] = (
    # Émotions
    "happy", "angry", "sad", "relaxed", "surprised",
    # Voyelles (lip-sync)
    "aa", "ih", "ou", "ee", "oh",
    # Cligne
    "blink", "blinkLeft", "blinkRight",
    # Regard
    "lookUp", "lookDown", "lookLeft", "lookRight",
    # Neutre
    "neutral",
)


# Renames VRM 0.x → VRM 1.0 (pour les utilisateurs qui ont des VRM 0.x).
# VRM 0.x utilisait : joy, sorrow, fun, blink, blink_l, blink_r,
# a, i, u, e, o, lookup, lookdown, lookleft, lookright, neutral.
# (pas de surprised dans VRM 0.x)
VRM0_TO_VRM1_EXPRESSION_RENAMES: dict[str, str] = {
    "joy": "happy",
    "sorrow": "sad",
    "fun": "relaxed",
    "blink_l": "blinkLeft",
    "blink_r": "blinkRight",
    "a": "aa",
    "i": "ih",
    "u": "ou",
    "e": "ee",
    "o": "oh",
    "lookup": "lookUp",
    "lookdown": "lookDown",
    "lookleft": "lookLeft",
    "lookright": "lookRight",
}


@dataclass
class FaceMapping:
    """Mapping facial résolu pour un avatar VRM particulier.

    Chaque champ vaut None si l'élément correspondant n'existe pas dans le VRM,
    auquel cas le retargeting log un warning et passe.

    Attributs :
        jaw_bone   : nom du bone VRM qui reçoit jaw_pose, ou None
        leye_bone  : bone yeux gauche, ou None
        reye_bone  : bone yeux droit, ou None
        expressions_available : ensemble des noms d'expressions présentes dans le VRM
        warnings   : liste des éléments non-mappés (pour log à la lecture)
    """

    jaw_bone: str | None
    leye_bone: str | None
    reye_bone: str | None
    expressions_available: frozenset[str]
    warnings: list[str]


def build_face_mapping(
    available_bones: set[str],
    available_expressions: set[str],
) -> FaceMapping:
    """Construit le mapping facial pour un VRM donné.

    Args:
        available_bones      : ensemble des bones humanoïdes présents dans le VRM
                               (sortie de vrm_inspector.inspect().humanoid_bones.keys())
        available_expressions: ensemble des noms d'expressions présentes
                               (sortie de vrm_inspector.inspect().expressions)

    Returns:
        FaceMapping résolu.
    """
    warnings: list[str] = []

    jaw_bone = "jaw" if "jaw" in available_bones else None
    if jaw_bone is None:
        warnings.append("Bone 'jaw' absent du VRM — la lip-sync est désactivée.")

    leye_bone = "leftEye" if "leftEye" in available_bones else None
    reye_bone = "rightEye" if "rightEye" in available_bones else None
    if leye_bone is None or reye_bone is None:
        warnings.append(
            "Bones 'leftEye' et/ou 'rightEye' absents — le regard est figé."
        )

    # Normalise les noms VRM 0.x si nécessaire pour comparer aux presets standard.
    normalized_expressions: set[str] = set()
    for name in available_expressions:
        normalized_expressions.add(VRM0_TO_VRM1_EXPRESSION_RENAMES.get(name, name))

    missing_presets = set(VRM_PRESET_EXPRESSIONS) - normalized_expressions
    if missing_presets:
        # Pas un vrai blocker — la plupart des avatars n'ont qu'un sous-ensemble.
        logger.debug(
            "Presets VRM manquants (informatif) : %s",
            ", ".join(sorted(missing_presets)),
        )

    for w in warnings:
        logger.warning(w)

    return FaceMapping(
        jaw_bone=jaw_bone,
        leye_bone=leye_bone,
        reye_bone=reye_bone,
        expressions_available=frozenset(normalized_expressions),
        warnings=warnings,
    )
