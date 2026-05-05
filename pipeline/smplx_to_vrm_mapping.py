"""Mapping statique SMPL-X → bones humanoïdes VRM.

Mapping par NOM de bone humanoïde standard VRM (pas par index), pour fonctionner
sur n'importe quel VRM conforme aux specs VRM 0.x ou 1.0.

Référence : docs/PIPELINE.md §4.

L'ordre dans BODY_JOINT_NAMES correspond à l'ordre d'indexage de body_pose dans
animation.npz : body_pose[:, i, :] est la rotation du joint BODY_JOINT_NAMES[i].
"""
from __future__ import annotations

from typing import Literal


# Ordre canonique des 21 joints corps de SMPL-X dans body_pose (indices 1..21
# de smplx.JOINT_NAMES, pelvis exclu car traité par global_orient).
# Cf. docs/PIPELINE.md §4.1.
BODY_JOINT_NAMES: list[str] = [
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
assert len(BODY_JOINT_NAMES) == 21


# Ordre canonique des 15 joints par main de SMPL-X dans left_hand_pose / right_hand_pose
# (sans le préfixe left_/right_). Cf. smplx/joint_names.py.
HAND_JOINT_NAMES_PER_HAND: list[str] = [
    "index1", "index2", "index3",
    "middle1", "middle2", "middle3",
    "pinky1", "pinky2", "pinky3",
    "ring1", "ring2", "ring3",
    "thumb1", "thumb2", "thumb3",
]
assert len(HAND_JOINT_NAMES_PER_HAND) == 15


# Mapping corps : SMPL-X joint → bone VRM. None signifie "pas de cible standard"
# (le retargeting peut tenter de re-router vers le parent ou logger un warning).
SMPLX_TO_VRM_BODY: dict[str, str | None] = {
    "pelvis": "hips",
    "left_hip": "leftUpperLeg",
    "right_hip": "rightUpperLeg",
    "spine1": "spine",
    "left_knee": "leftLowerLeg",
    "right_knee": "rightLowerLeg",
    "spine2": "chest",          # optionnel — fallback : spine
    "left_ankle": "leftFoot",
    "right_ankle": "rightFoot",
    "spine3": "upperChest",     # optionnel — fallback : chest puis spine
    "left_foot": "leftToes",    # optionnel
    "right_foot": "rightToes",  # optionnel
    "neck": "neck",             # optionnel — fallback : head
    "left_collar": "leftShoulder",   # optionnel
    "right_collar": "rightShoulder", # optionnel
    "head": "head",
    "left_shoulder": "leftUpperArm",
    "right_shoulder": "rightUpperArm",
    "left_elbow": "leftLowerArm",
    "right_elbow": "rightLowerArm",
    "left_wrist": "leftHand",
    "right_wrist": "rightHand",
}


# Mapping face : ces joints sont stockés en dehors de body_pose dans animation.npz
# (jaw_pose, leye_pose, reye_pose).
SMPLX_FACE_TO_VRM: dict[str, str | None] = {
    "jaw": "jaw",                       # optionnel
    "left_eye_smplhf": "leftEye",       # optionnel
    "right_eye_smplhf": "rightEye",     # optionnel
}


# Mapping doigts pour VRM 1.0 : pouce = Metacarpal/Proximal/Distal,
# autres doigts = Proximal/Intermediate/Distal.
# SMPL-X « pinky » devient VRM « little ».
def _hand_mapping_vrm1(side: Literal["left", "right"]) -> dict[str, str]:
    return {
        f"{side}_thumb1":  f"{side}ThumbMetacarpal",
        f"{side}_thumb2":  f"{side}ThumbProximal",
        f"{side}_thumb3":  f"{side}ThumbDistal",
        f"{side}_index1":  f"{side}IndexProximal",
        f"{side}_index2":  f"{side}IndexIntermediate",
        f"{side}_index3":  f"{side}IndexDistal",
        f"{side}_middle1": f"{side}MiddleProximal",
        f"{side}_middle2": f"{side}MiddleIntermediate",
        f"{side}_middle3": f"{side}MiddleDistal",
        f"{side}_ring1":   f"{side}RingProximal",
        f"{side}_ring2":   f"{side}RingIntermediate",
        f"{side}_ring3":   f"{side}RingDistal",
        f"{side}_pinky1":  f"{side}LittleProximal",
        f"{side}_pinky2":  f"{side}LittleIntermediate",
        f"{side}_pinky3":  f"{side}LittleDistal",
    }


# Mapping doigts pour VRM 0.x : pouce = Proximal/Intermediate/Distal (sans Metacarpal).
def _hand_mapping_vrm0(side: Literal["left", "right"]) -> dict[str, str]:
    return {
        f"{side}_thumb1":  f"{side}ThumbProximal",
        f"{side}_thumb2":  f"{side}ThumbIntermediate",
        f"{side}_thumb3":  f"{side}ThumbDistal",
        f"{side}_index1":  f"{side}IndexProximal",
        f"{side}_index2":  f"{side}IndexIntermediate",
        f"{side}_index3":  f"{side}IndexDistal",
        f"{side}_middle1": f"{side}MiddleProximal",
        f"{side}_middle2": f"{side}MiddleIntermediate",
        f"{side}_middle3": f"{side}MiddleDistal",
        f"{side}_ring1":   f"{side}RingProximal",
        f"{side}_ring2":   f"{side}RingIntermediate",
        f"{side}_ring3":   f"{side}RingDistal",
        f"{side}_pinky1":  f"{side}LittleProximal",
        f"{side}_pinky2":  f"{side}LittleIntermediate",
        f"{side}_pinky3":  f"{side}LittleDistal",
    }


SMPLX_TO_VRM_LEFT_HAND_VRM1: dict[str, str] = _hand_mapping_vrm1("left")
SMPLX_TO_VRM_RIGHT_HAND_VRM1: dict[str, str] = _hand_mapping_vrm1("right")
SMPLX_TO_VRM_LEFT_HAND_VRM0: dict[str, str] = _hand_mapping_vrm0("left")
SMPLX_TO_VRM_RIGHT_HAND_VRM0: dict[str, str] = _hand_mapping_vrm0("right")


# Bones VRM 1.0 obligatoires (cf. spec humanoid.md).
VRM_REQUIRED_BONES: frozenset[str] = frozenset({
    "hips", "spine", "head",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot",
})


def get_full_mapping(vrm_version: Literal["0.x", "1.0"]) -> dict[str, str]:
    """Retourne le mapping complet SMPL-X → VRM en concaténant corps, face, mains.

    Tous les joints SMPL-X qui ont une cible non-nulle dans le mapping statique
    sont inclus. La présence effective du bone cible sur l'avatar VRM doit être
    vérifiée à part via vrm_inspector.

    Args:
        vrm_version : "0.x" ou "1.0" — détermine le naming des bones du pouce.

    Returns:
        dict {nom_smplx: nom_vrm} — uniquement les paires non-None.
    """
    mapping: dict[str, str] = {}
    for k, v in SMPLX_TO_VRM_BODY.items():
        if v is not None:
            mapping[k] = v
    for k, v in SMPLX_FACE_TO_VRM.items():
        if v is not None:
            mapping[k] = v
    if vrm_version == "1.0":
        mapping.update(SMPLX_TO_VRM_LEFT_HAND_VRM1)
        mapping.update(SMPLX_TO_VRM_RIGHT_HAND_VRM1)
    elif vrm_version == "0.x":
        mapping.update(SMPLX_TO_VRM_LEFT_HAND_VRM0)
        mapping.update(SMPLX_TO_VRM_RIGHT_HAND_VRM0)
    else:
        raise ValueError(f"Version VRM inconnue : {vrm_version!r} (attendu '0.x' ou '1.0')")
    return mapping


def get_body_mapping(vrm_version: Literal["0.x", "1.0"]) -> dict[str, str]:
    """Mapping pour les 21 joints du body_pose uniquement.

    L'index dans body_pose est donné par BODY_JOINT_NAMES.index(<nom>).
    """
    full = get_full_mapping(vrm_version)
    return {name: full[name] for name in BODY_JOINT_NAMES if name in full}


def get_hand_mapping(
    side: Literal["left", "right"],
    vrm_version: Literal["0.x", "1.0"],
) -> list[tuple[int, str, str]]:
    """Mapping ordonné pour une main : liste de (index_dans_hand_pose, smplx_name, vrm_name).

    Permet d'itérer left_hand_pose[:, i, :] et de savoir vers quel bone VRM
    l'envoyer.
    """
    full = get_full_mapping(vrm_version)
    out: list[tuple[int, str, str]] = []
    for i, joint_short in enumerate(HAND_JOINT_NAMES_PER_HAND):
        smplx_name = f"{side}_{joint_short}"
        if smplx_name in full:
            out.append((i, smplx_name, full[smplx_name]))
    return out
