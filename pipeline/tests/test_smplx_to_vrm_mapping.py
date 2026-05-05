"""Tests du mapping SMPL-X → VRM."""
from __future__ import annotations

import pytest

from pipeline.smplx_to_vrm_mapping import (
    BODY_JOINT_NAMES,
    HAND_JOINT_NAMES_PER_HAND,
    SMPLX_TO_VRM_BODY,
    VRM_REQUIRED_BONES,
    get_body_mapping,
    get_full_mapping,
    get_hand_mapping,
)


def test_body_joint_names_count() -> None:
    assert len(BODY_JOINT_NAMES) == 21


def test_hand_joint_names_count() -> None:
    assert len(HAND_JOINT_NAMES_PER_HAND) == 15


def test_required_bones_subset_of_mapping() -> None:
    """Tous les bones VRM obligatoires sont produits par le mapping."""
    full = get_full_mapping("1.0")
    targets = set(full.values())
    missing = VRM_REQUIRED_BONES - targets
    assert not missing, f"Bones VRM obligatoires non couverts par mapping : {missing}"


def test_thumb_naming_differs_vrm0_vs_vrm1() -> None:
    """VRM 1.0 a thumbMetacarpal, VRM 0.x a thumbProximal pour le 1er segment."""
    m1 = get_full_mapping("1.0")
    m0 = get_full_mapping("0.x")
    assert m1["left_thumb1"] == "leftThumbMetacarpal"
    assert m0["left_thumb1"] == "leftThumbProximal"
    assert m1["left_thumb3"] == "leftThumbDistal"
    assert m0["left_thumb3"] == "leftThumbDistal"  # même nom pour le dernier


def test_pinky_renamed_to_little() -> None:
    """SMPL-X pinky → VRM little."""
    m = get_full_mapping("1.0")
    assert m["left_pinky1"] == "leftLittleProximal"
    assert m["right_pinky3"] == "rightLittleDistal"


def test_invalid_version_raises() -> None:
    with pytest.raises(ValueError, match="Version VRM"):
        get_full_mapping("2.0")  # type: ignore[arg-type]


def test_body_mapping_returns_only_body() -> None:
    body_map = get_body_mapping("1.0")
    # 21 joints corps, mais certains sont optionnels (peuvent être None dans SMPLX_TO_VRM_BODY).
    # Ici tous sont mappés vers une cible non-None, donc len = 21.
    assert len(body_map) == 21
    assert all(name in BODY_JOINT_NAMES for name in body_map)


def test_hand_mapping_indexed() -> None:
    """get_hand_mapping retourne (index_dans_hand_pose, smplx_name, vrm_name)."""
    left = get_hand_mapping("left", "1.0")
    assert len(left) == 15
    indices = [i for i, _, _ in left]
    assert indices == list(range(15))
    smplx_names = [s for _, s, _ in left]
    assert all(name.startswith("left_") for name in smplx_names)


def test_pelvis_maps_to_hips() -> None:
    assert SMPLX_TO_VRM_BODY["pelvis"] == "hips"


def test_no_duplicate_targets_per_side() -> None:
    """Vérifie qu'aucun bone VRM n'est cible de plusieurs joints SMPL-X."""
    m = get_full_mapping("1.0")
    # On exclut les Nones (déjà filtrés) — vérification d'unicité
    targets = list(m.values())
    assert len(targets) == len(set(targets)), (
        f"Doublons : {[t for t in targets if targets.count(t) > 1]}"
    )
