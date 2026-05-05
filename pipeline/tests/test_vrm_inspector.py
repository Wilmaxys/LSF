"""Tests de l'inspecteur VRM.

Comme on n'a pas de fichier VRM en fixture, on teste les fonctions auxiliaires
(parsers de matrice, classification d'extension) avec des entrées synthétiques.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.vrm_inspector import (
    _quat_to_rot_matrix,
    inspect,
    is_vrm_compatible,
)


def test_quat_to_rot_identity() -> None:
    R = _quat_to_rot_matrix(np.array([0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_array_almost_equal(R, np.eye(3))


def test_quat_to_rot_90deg_y() -> None:
    """Quaternion (0, sin45, 0, cos45) = rotation 90° autour de Y."""
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    R = _quat_to_rot_matrix(np.array([0, s, 0, c]))
    # Rotation 90° autour de Y : x → -z, z → x
    np.testing.assert_array_almost_equal(R @ [1, 0, 0], [0, 0, -1])
    np.testing.assert_array_almost_equal(R @ [0, 0, 1], [1, 0, 0])


def test_inspect_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        inspect(Path("/nonexistent.vrm"))


def test_inspect_requires_pygltflib(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Si pygltflib n'est pas dispo, ImportError clair."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "pygltflib":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pygltflib"):
        inspect("/some/path.vrm")


def test_is_vrm_compatible_calls_inspect(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """is_vrm_compatible vérifie la présence des bones obligatoires."""
    from pipeline import vrm_inspector
    from pipeline.vrm_inspector import VRMMetadata

    fake_meta = VRMMetadata(
        version="1.0",
        humanoid_bones={
            "hips": "Hips",
            "spine": "Spine",
            "head": "Head",
            # Manque tous les bones de bras/jambes !
        },
        expressions=[],
        rest_poses_local={},
        raw={},
        node_indices={},
    )
    monkeypatch.setattr(vrm_inspector, "inspect", lambda p: fake_meta)
    ok, problems = is_vrm_compatible("/dummy.vrm")
    assert not ok
    assert any("leftUpperArm" in p for p in problems)


def test_is_vrm_compatible_complete() -> None:
    """Avec tous les bones obligatoires : ok = True."""
    from pipeline import vrm_inspector
    from pipeline.vrm_inspector import VRMMetadata
    from pipeline.smplx_to_vrm_mapping import VRM_REQUIRED_BONES

    fake_meta = VRMMetadata(
        version="1.0",
        humanoid_bones={b: b.title() for b in VRM_REQUIRED_BONES},
        expressions=[],
        rest_poses_local={},
        raw={},
        node_indices={},
    )
    import unittest.mock as mock
    with mock.patch.object(vrm_inspector, "inspect", return_value=fake_meta):
        ok, problems = is_vrm_compatible("/dummy.vrm")
    assert ok
    assert problems == []
