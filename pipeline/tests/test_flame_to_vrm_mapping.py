"""Tests du mapping FLAME → expressions VRM."""
from __future__ import annotations

from pipeline.flame_to_vrm_mapping import (
    VRM0_TO_VRM1_EXPRESSION_RENAMES,
    VRM_PRESET_EXPRESSIONS,
    build_face_mapping,
)


def test_preset_count() -> None:
    """Presets standards VRM 1.0 : 5 émotions + 5 voyelles + 3 blink + 4 lookAt + neutral = 18."""
    assert len(VRM_PRESET_EXPRESSIONS) == 18
    assert "happy" in VRM_PRESET_EXPRESSIONS
    assert "surprised" in VRM_PRESET_EXPRESSIONS
    assert "neutral" in VRM_PRESET_EXPRESSIONS


def test_vrm0_renames() -> None:
    assert VRM0_TO_VRM1_EXPRESSION_RENAMES["joy"] == "happy"
    assert VRM0_TO_VRM1_EXPRESSION_RENAMES["sorrow"] == "sad"
    assert VRM0_TO_VRM1_EXPRESSION_RENAMES["a"] == "aa"


def test_full_humanoid_mapping() -> None:
    bones = {"hips", "spine", "head", "jaw", "leftEye", "rightEye"}
    expressions = {"happy", "aa", "blink"}
    fm = build_face_mapping(bones, expressions)
    assert fm.jaw_bone == "jaw"
    assert fm.leye_bone == "leftEye"
    assert fm.reye_bone == "rightEye"
    assert "happy" in fm.expressions_available
    assert fm.warnings == []


def test_missing_jaw_bone() -> None:
    bones = {"hips", "spine", "head"}  # pas de jaw
    fm = build_face_mapping(bones, set())
    assert fm.jaw_bone is None
    assert any("jaw" in w for w in fm.warnings)


def test_vrm0_expressions_normalized() -> None:
    """Les noms d'expression VRM 0.x sont normalisés en VRM 1.0."""
    expressions = {"joy", "sorrow", "a", "i", "blink"}
    fm = build_face_mapping({"hips", "spine", "head"}, expressions)
    assert "happy" in fm.expressions_available
    assert "sad" in fm.expressions_available
    assert "aa" in fm.expressions_available
    assert "ih" in fm.expressions_available


def test_missing_eyes_warns() -> None:
    bones = {"hips", "spine", "head", "jaw"}
    fm = build_face_mapping(bones, set())
    assert fm.leye_bone is None
    assert fm.reye_bone is None
    assert any("Eye" in w for w in fm.warnings)
