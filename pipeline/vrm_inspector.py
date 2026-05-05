"""Inspection dynamique d'un fichier VRM.

Lit le squelette humanoïde et la liste d'expressions d'un VRM (0.x ou 1.0)
**sans** lancer Blender (utilise pygltflib). Suffisant pour :
    - vérifier qu'un VRM est compatible avec le pipeline (verify_env.py) ;
    - construire les mappings statiques au moment du retargeting.

L'extraction des matrices de rest-pose hiérarchiques cumulées (R_vrm_rest)
nécessite Blender (matrix_local des bones) — non implémenté dans ce module ;
voir pipeline/retarget.py.

Cf. docs/PIPELINE.md §6.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VRMMetadata:
    """Métadonnées extraites d'un fichier VRM.

    Attributs :
        version           : "0.x" ou "1.0"
        humanoid_bones    : dict {nom_bone_vrm: nom_node_glTF}.
                            Contient uniquement les bones effectivement mappés
                            dans le fichier — les bones VRM standards mais non
                            définis dans cet avatar n'apparaissent PAS ici.
        expressions       : liste triée des noms d'expressions disponibles
                            (presets et custom, normalisés en VRM 1.0 naming si VRM 0.x).
        rest_poses_local  : dict {nom_bone_vrm: matrice 4×4 locale (numpy.ndarray)}
                            ou {} si non calculé (mode pygltflib seul).
        raw               : dict brut de l'extension VRM/VRMC_vrm pour debug.
        node_indices      : dict {nom_bone_vrm: index du node glTF}.
    """

    version: Literal["0.x", "1.0"]
    humanoid_bones: dict[str, str]
    expressions: list[str]
    rest_poses_local: dict[str, np.ndarray]
    raw: dict[str, Any]
    node_indices: dict[str, int] = field(default_factory=dict)


# Presets reconnus par VRM 0.x — les autres clés `presetName` sont des customs.
_VRM0_PRESET_NAMES: frozenset[str] = frozenset({
    "neutral", "a", "i", "u", "e", "o",
    "blink", "blink_l", "blink_r",
    "joy", "angry", "sorrow", "fun",
    "lookup", "lookdown", "lookleft", "lookright",
})


def inspect(vrm_path: str | Path) -> VRMMetadata:
    """Inspecte un fichier .vrm et retourne ses métadonnées.

    Lit l'extension VRMC_vrm (VRM 1.0) ou VRM (VRM 0.x) du glTF. Les rest-poses
    locales retournées sont les transformations locales **directes** stockées
    dans chaque node glTF (rotation / translation / scale / matrix). Pour la
    rest-pose cumulée (concaténation hiérarchique), passer par retarget.py /
    Blender.

    Args:
        vrm_path : chemin vers le fichier .vrm

    Returns:
        VRMMetadata

    Raises:
        FileNotFoundError : fichier absent
        ValueError        : fichier non-VRM (ni VRMC_vrm ni VRM)
        ImportError       : pygltflib non installé
    """
    try:
        from pygltflib import GLTF2
    except ImportError as exc:
        raise ImportError(
            "pygltflib est requis pour inspecter un VRM. "
            "Installez-le avec : pip install pygltflib"
        ) from exc

    vrm_path = Path(vrm_path)
    if not vrm_path.exists():
        raise FileNotFoundError(f"VRM introuvable : {vrm_path}")

    # pygltflib choisit text/binary via l'extension : .glb → binaire, sinon JSON.
    # Les fichiers .vrm sont en réalité du GLB binaire (magic "glTF"). On détecte
    # via les premiers octets et on appelle directement le bon loader.
    with open(vrm_path, "rb") as f:
        magic = f.read(4)
    if magic == b"glTF":
        gltf = GLTF2().load_binary(str(vrm_path))
    else:
        gltf = GLTF2().load_json(str(vrm_path))
    extensions = gltf.extensions or {}

    # VRM 1.0 ?
    if "VRMC_vrm" in extensions:
        return _inspect_vrm1(gltf, extensions["VRMC_vrm"])

    # VRM 0.x ?
    if "VRM" in extensions:
        return _inspect_vrm0(gltf, extensions["VRM"])

    raise ValueError(
        f"Fichier {vrm_path} n'a ni l'extension 'VRMC_vrm' (VRM 1.0) "
        f"ni 'VRM' (VRM 0.x). Extensions trouvées : {list(extensions)}"
    )


def is_vrm_compatible(
    vrm_path: str | Path,
    *,
    require_jaw: bool = False,
    require_eyes: bool = False,
) -> tuple[bool, list[str]]:
    """Vérifie qu'un VRM contient au moins les bones obligatoires pour le pipeline.

    Args:
        vrm_path     : chemin vers le .vrm
        require_jaw  : si True, le bone `jaw` doit être présent (sinon warning seulement)
        require_eyes : si True, les bones `leftEye` et `rightEye` sont requis

    Returns:
        (compatible, problems) — compatible=False si l'un des bones obligatoires
        manque ; problems liste les bones manquants (vide si compatible).
    """
    from pipeline.smplx_to_vrm_mapping import VRM_REQUIRED_BONES

    metadata = inspect(vrm_path)
    available = set(metadata.humanoid_bones.keys())

    missing_required = VRM_REQUIRED_BONES - available
    problems: list[str] = []
    for bone in sorted(missing_required):
        problems.append(f"Bone obligatoire manquant : {bone}")

    if require_jaw and "jaw" not in available:
        problems.append("Bone 'jaw' requis mais absent")
    if require_eyes:
        if "leftEye" not in available:
            problems.append("Bone 'leftEye' requis mais absent")
        if "rightEye" not in available:
            problems.append("Bone 'rightEye' requis mais absent")

    return (len(problems) == 0, problems)


# ──────────────────────────────────────────────────────────────────────────────
# Implémentations privées
# ──────────────────────────────────────────────────────────────────────────────

def _inspect_vrm1(gltf, vrm_ext: dict) -> VRMMetadata:
    """Parse l'extension VRMC_vrm (VRM 1.0)."""
    humanoid = vrm_ext.get("humanoid", {})
    human_bones_dict = humanoid.get("humanBones", {})
    # Format VRM 1.0 : { "hips": {"node": 42}, "spine": {"node": 43}, ... }

    humanoid_bones: dict[str, str] = {}
    node_indices: dict[str, int] = {}
    rest_poses_local: dict[str, np.ndarray] = {}

    for bone_name, info in human_bones_dict.items():
        if not isinstance(info, dict) or "node" not in info:
            continue
        node_idx = int(info["node"])
        if node_idx < 0 or node_idx >= len(gltf.nodes):
            logger.warning("Node %d hors borne pour bone %s — ignoré", node_idx, bone_name)
            continue
        node = gltf.nodes[node_idx]
        humanoid_bones[bone_name] = node.name or f"node_{node_idx}"
        node_indices[bone_name] = node_idx
        rest_poses_local[bone_name] = _local_matrix(node)

    expressions_section = vrm_ext.get("expressions", {})
    presets = expressions_section.get("preset", {}) or {}
    customs = expressions_section.get("custom", {}) or {}
    expressions = sorted(set(presets.keys()) | set(customs.keys()))

    return VRMMetadata(
        version="1.0",
        humanoid_bones=humanoid_bones,
        expressions=expressions,
        rest_poses_local=rest_poses_local,
        raw=vrm_ext,
        node_indices=node_indices,
    )


def _inspect_vrm0(gltf, vrm_ext: dict) -> VRMMetadata:
    """Parse l'extension VRM (VRM 0.x).

    Format différent : humanBones est une LISTE [{bone, node}, ...] au lieu
    d'un dict. Expressions stockées sous blendShapeMaster.blendShapeGroups.
    Les noms d'expressions sont normalisés en VRM 1.0 naming.
    """
    from pipeline.flame_to_vrm_mapping import VRM0_TO_VRM1_EXPRESSION_RENAMES

    humanoid = vrm_ext.get("humanoid", {})
    human_bones_list = humanoid.get("humanBones", [])

    humanoid_bones: dict[str, str] = {}
    node_indices: dict[str, int] = {}
    rest_poses_local: dict[str, np.ndarray] = {}

    for entry in human_bones_list:
        if not isinstance(entry, dict):
            continue
        bone_name = entry.get("bone")
        node_idx = entry.get("node")
        if not bone_name or node_idx is None:
            continue
        node_idx = int(node_idx)
        if node_idx < 0 or node_idx >= len(gltf.nodes):
            continue
        node = gltf.nodes[node_idx]
        humanoid_bones[bone_name] = node.name or f"node_{node_idx}"
        node_indices[bone_name] = node_idx
        rest_poses_local[bone_name] = _local_matrix(node)

    blend_master = vrm_ext.get("blendShapeMaster", {})
    blend_groups = blend_master.get("blendShapeGroups", []) or []

    raw_expressions: list[str] = []
    for grp in blend_groups:
        if not isinstance(grp, dict):
            continue
        # Le name du groupe peut être custom ; le presetName est ce qui mappe sur les standards.
        preset = grp.get("presetName") or grp.get("name")
        if preset:
            raw_expressions.append(str(preset).lower())

    # Normalise vers le naming VRM 1.0 pour homogénéiser les comparaisons en aval.
    normalized: set[str] = set()
    for name in raw_expressions:
        normalized.add(VRM0_TO_VRM1_EXPRESSION_RENAMES.get(name, name))

    return VRMMetadata(
        version="0.x",
        humanoid_bones=humanoid_bones,
        expressions=sorted(normalized),
        rest_poses_local=rest_poses_local,
        raw=vrm_ext,
        node_indices=node_indices,
    )


def _local_matrix(node) -> np.ndarray:
    """Construit la matrice 4×4 de transformation locale d'un node glTF.

    Si node.matrix est défini, le retourne directement. Sinon, compose à partir
    de translation / rotation (quaternion) / scale.
    """
    if node.matrix is not None:
        # glTF stocke les matrices en column-major. numpy.reshape((4,4)) est row-major,
        # donc on transpose.
        return np.array(node.matrix, dtype=np.float64).reshape(4, 4).T

    T = np.array(node.translation or [0.0, 0.0, 0.0], dtype=np.float64)
    R_quat = np.array(node.rotation or [0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # (x, y, z, w)
    S = np.array(node.scale or [1.0, 1.0, 1.0], dtype=np.float64)

    R = _quat_to_rot_matrix(R_quat)
    M = np.eye(4)
    M[:3, :3] = R * S  # scale appliqué après rotation
    M[:3, 3] = T
    return M


def _quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convertit un quaternion (x, y, z, w) en matrice de rotation 3×3."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-10:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ], dtype=np.float64)
