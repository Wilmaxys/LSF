"""Retargeting SMPL-X → VRM via Blender headless + VRM addon.

Lance dans Blender :
    blender -b --addons io_scene_vrm --python pipeline/retarget.py -- \\
        --avatar X.vrm --animation Y.npz --output Z.vrma

Étapes :
    1. Charger le VRM avec l'addon
    2. Inspecter le squelette (bones humanoïdes présents + rest poses)
    3. Charger animation.npz
    4. Pour chaque frame, calculer les keyframes par bone VRM via R_offset · R_anim
    5. Mapper jaw_pose → bone VRM `jaw` (si présent)
    6. Mapper expression → expressions VRM (cf. flame_to_vrm_mapping)
    7. Bake et exporter en .vrma (ou .glb si extension différente)

Cf. docs/PIPELINE.md §4.4 et §6.

⚠️ Ce script tourne UNIQUEMENT dans Blender. Il importe `bpy` qui n'est pas
disponible en Python standalone.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Ajout de la racine du repo au PYTHONPATH (Blender ne le fait pas par défaut)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.animation_npz import Animation  # noqa: E402
from pipeline.flame_to_vrm_mapping import build_face_mapping  # noqa: E402
from pipeline.smplx_to_vrm_mapping import (  # noqa: E402
    BODY_JOINT_NAMES, get_body_mapping, get_hand_mapping,
)

logger = logging.getLogger("retarget")


def main() -> int:
    """Entry point appelé par Blender. Parse les args après "--"."""
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    # Blender passe ses propres args avant "--", les nôtres après.
    if "--" not in sys.argv:
        argv = []
    else:
        argv = sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Retargeting SMPL-X → VRM")
    parser.add_argument("--avatar", type=Path, required=True)
    parser.add_argument("--animation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    # Imports Blender — DOIVENT être ici, pas en haut du fichier
    import bpy

    logger.info("Retargeting : %s + %s → %s",
                args.avatar.name, args.animation.name, args.output.name)

    # 1. Reset scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Activer l'addon VRM — auto-détection du module name (varie selon
    # la version : `io_scene_vrm` legacy vs `VRM_Addon_for_Blender-release` récent).
    import addon_utils
    vrm_modules = [m.__name__ for m in addon_utils.modules() if 'vrm' in m.__name__.lower()]
    if not vrm_modules:
        raise RuntimeError("Aucun addon VRM trouvé. Lancer scripts/setup.sh.")
    vrm_module = vrm_modules[0]
    _default, loaded = addon_utils.check(vrm_module)
    if not loaded:
        bpy.ops.preferences.addon_enable(module=vrm_module)
    logger.info("Addon VRM : %s", vrm_module)

    # 3. Importer le VRM
    bpy.ops.import_scene.vrm(filepath=str(args.avatar))

    armature = _find_armature()
    if armature is None:
        raise RuntimeError(f"Armature non trouvée après import de {args.avatar}")

    # 4. Inspection du squelette via le VRM addon
    vrm_metadata = _inspect_vrm_via_addon(armature)
    logger.info("VRM version : %s", vrm_metadata["version"])
    logger.info("Bones humanoïdes mappés : %d", len(vrm_metadata["humanoid_bones"]))
    logger.info("Expressions disponibles : %s", vrm_metadata["expressions"])

    # 5. Charger animation
    anim = Animation.load(args.animation)
    logger.info("Animation : %d frames @ %.1f fps", anim.num_frames, anim.fps)

    # 6. Mapper et baker les rotations
    _bake_animation(armature, anim, vrm_metadata)

    # 7. Mapper le visage
    face_mapping = build_face_mapping(
        available_bones=set(vrm_metadata["humanoid_bones"].keys()),
        available_expressions=set(vrm_metadata["expressions"]),
    )
    _bake_face(armature, anim, vrm_metadata, face_mapping)

    # 8. Export
    _export(args.output, anim.fps, anim.num_frames)
    logger.info("Export terminé : %s", args.output)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers Blender
# ──────────────────────────────────────────────────────────────────────────────

def _find_armature():
    """Retourne le premier objet Armature de la scène."""
    import bpy
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            return obj
    return None


"""Liste canonique des 55 bones humanoïdes VRM 1.0 (cf. spec humanoid.md).

On itère explicitement cette liste plutôt que `dir(human_bones)` qui retourne
tous les attributs Blender (bl_rna, name, etc.) en plus des bones.
"""
_VRM1_HUMANOID_BONES: tuple[str, ...] = (
    # Required (15)
    "hips", "spine", "head",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot",
    # Optional torso/head/limbs
    "chest", "upperChest", "neck",
    "leftEye", "rightEye", "jaw",
    "leftShoulder", "rightShoulder",
    "leftToes", "rightToes",
    # Optional fingers (15 par main = 30)
    "leftThumbMetacarpal", "leftThumbProximal", "leftThumbDistal",
    "leftIndexProximal", "leftIndexIntermediate", "leftIndexDistal",
    "leftMiddleProximal", "leftMiddleIntermediate", "leftMiddleDistal",
    "leftRingProximal", "leftRingIntermediate", "leftRingDistal",
    "leftLittleProximal", "leftLittleIntermediate", "leftLittleDistal",
    "rightThumbMetacarpal", "rightThumbProximal", "rightThumbDistal",
    "rightIndexProximal", "rightIndexIntermediate", "rightIndexDistal",
    "rightMiddleProximal", "rightMiddleIntermediate", "rightMiddleDistal",
    "rightRingProximal", "rightRingIntermediate", "rightRingDistal",
    "rightLittleProximal", "rightLittleIntermediate", "rightLittleDistal",
)

# 18 expressions presets VRM 1.0 (cf. spec expressions.md).
_VRM1_PRESET_EXPRESSIONS: tuple[str, ...] = (
    "happy", "angry", "sad", "relaxed", "surprised",
    "aa", "ih", "ou", "ee", "oh",
    "blink", "blinkLeft", "blinkRight",
    "lookUp", "lookDown", "lookLeft", "lookRight",
    "neutral",
)


def _inspect_vrm_via_addon(armature) -> dict:
    """Lit l'extension VRM stockée par l'addon dans armature.data.vrm_addon_extension.

    API vérifiée le 2026-05-05 sur VRM_Addon_for_Blender v3.27.0 :
        - VRM 1.0 humanoid : armature.data.vrm_addon_extension.vrm1.humanoid
                             .human_bones.<bone_name>.node.bone_name
                             (Vrm1HumanBonesPropertyGroup, PointerProperty par bone)
        - VRM 1.0 expressions presets :
                             .vrm1.expressions.preset.<name>
                             (Vrm1ExpressionsPresetPropertyGroup)
        - VRM 1.0 expressions custom : .vrm1.expressions.custom[i].custom_name
                             (CollectionProperty)
        - VRM 0.x humanoid : .vrm0.humanoid.human_bones[i] (CollectionProperty)
                             avec .bone (nom VRM) et .node.bone_name (nom Blender)
        - VRM 0.x expressions : .vrm0.blend_shape_master.blend_shape_groups[i]
                             avec .preset_name et .name

    Retourne un dict :
        {
            'version': '1.0' | '0.x',
            'humanoid_bones': {vrm_bone_name: blender_bone_name, ...},
            'expressions': [name, ...],
            'rest_poses_local': {vrm_bone_name: matrix_local 4x4, ...},
        }
    """
    ext = getattr(armature.data, "vrm_addon_extension", None)
    if ext is None:
        raise RuntimeError(
            "armature.data.vrm_addon_extension absent — VRM addon mal initialisé "
            "ou avatar non-VRM"
        )

    humanoid_bones: dict[str, str] = {}
    expressions: list[str] = []
    version = "1.0"

    # ── VRM 1.0 ────────────────────────────────────────────────────────────
    vrm1 = getattr(ext, "vrm1", None)
    if vrm1 is not None and getattr(vrm1, "humanoid", None) is not None:
        human_bones = vrm1.humanoid.human_bones
        for vrm_bone_name in _VRM1_HUMANOID_BONES:
            sub = getattr(human_bones, vrm_bone_name, None)
            if sub is None:
                continue
            node = getattr(sub, "node", None)
            blender_bone = getattr(node, "bone_name", None) if node else None
            if blender_bone:
                humanoid_bones[vrm_bone_name] = blender_bone

        expr = getattr(vrm1, "expressions", None)
        if expr is not None:
            preset = getattr(expr, "preset", None)
            if preset is not None:
                for preset_name in _VRM1_PRESET_EXPRESSIONS:
                    if hasattr(preset, preset_name):
                        # On retient le preset comme "disponible" — l'avatar peut
                        # toutefois ne pas avoir de morph target binding (ce qui
                        # rendra l'expression sans effet). C'est délibérément
                        # tolérant : on log juste, on ne filtre pas.
                        expressions.append(preset_name)
            customs = getattr(expr, "custom", None)
            if customs is not None:
                for c in customs:
                    name = getattr(c, "custom_name", None)
                    if name:
                        expressions.append(str(name))

    # ── VRM 0.x (fallback) ────────────────────────────────────────────────
    if not humanoid_bones:
        version = "0.x"
        vrm0 = getattr(ext, "vrm0", None)
        if vrm0 is not None and getattr(vrm0, "humanoid", None) is not None:
            for hb in vrm0.humanoid.human_bones:
                vrm_bone_name = getattr(hb, "bone", None)
                node_bone = getattr(getattr(hb, "node", None), "bone_name", None)
                if vrm_bone_name and node_bone:
                    humanoid_bones[vrm_bone_name] = node_bone

            blend_master = getattr(vrm0, "blend_shape_master", None)
            if blend_master is not None:
                for grp in getattr(blend_master, "blend_shape_groups", []):
                    name = getattr(grp, "preset_name", None) or getattr(grp, "name", None)
                    if name:
                        expressions.append(str(name).lower())

    if not humanoid_bones:
        raise RuntimeError(
            "Aucun bone humanoïde trouvé via l'API VRM addon. "
            "L'avatar est-il bien un VRM conforme ?"
        )

    # Rest poses : matrix_local de chaque bone Blender mappé
    rest_poses_local: dict[str, np.ndarray] = {}
    for vrm_name, blender_name in humanoid_bones.items():
        bone = armature.data.bones.get(blender_name)
        if bone is None:
            continue
        rest_poses_local[vrm_name] = np.array(bone.matrix_local, dtype=np.float64)

    expressions = sorted(set(expressions))
    return {
        "version": version,
        "humanoid_bones": humanoid_bones,
        "expressions": expressions,
        "rest_poses_local": rest_poses_local,
        "armature": armature,
    }


def _bake_animation(armature, anim: Animation, vrm_metadata: dict) -> None:
    """Crée les keyframes de rotation pour chaque bone VRM mappé.

    Pour chaque frame t et chaque (smplx_joint, vrm_bone) du mapping, on calcule
    la rotation finale en pose.bones[bone_name].rotation_quaternion.

    L'ordre de composition (R_offset · R_anim · R_rest) est documenté en §4.4
    de docs/PIPELINE.md — c'est l'approximation standard, à valider visuellement
    sur un VRM connu (ex. AliciaSolid).
    """
    import bpy
    from mathutils import Quaternion

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    body_map = get_body_mapping(vrm_metadata["version"])
    lhand_map = get_hand_mapping("left", vrm_metadata["version"])
    rhand_map = get_hand_mapping("right", vrm_metadata["version"])

    humanoid_bones = vrm_metadata["humanoid_bones"]
    rest_poses = vrm_metadata["rest_poses_local"]

    # Active le mode quaternion pour tous les bones concernés
    for vrm_bone_name in humanoid_bones:
        blender_bone = humanoid_bones[vrm_bone_name]
        if blender_bone in armature.pose.bones:
            armature.pose.bones[blender_bone].rotation_mode = "QUATERNION"

    # Set scene frame range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = anim.num_frames
    bpy.context.scene.render.fps = int(round(anim.fps))

    # Itère frames
    for t in range(anim.num_frames):
        bpy.context.scene.frame_set(t + 1)

        # Root : translation + global_orient sur hips
        hips_blender = humanoid_bones.get("hips")
        if hips_blender and hips_blender in armature.pose.bones:
            pb = armature.pose.bones[hips_blender]
            pb.location = (
                float(anim.transl[t, 0]),
                float(anim.transl[t, 1]),
                float(anim.transl[t, 2]),
            )
            pb.rotation_quaternion = _aa_to_blender_quat(anim.global_orient[t])
            pb.keyframe_insert("location")
            pb.keyframe_insert("rotation_quaternion")

        # Body
        for smplx_name, vrm_bone in body_map.items():
            if smplx_name == "pelvis":
                continue  # déjà fait via hips
            if vrm_bone not in humanoid_bones:
                continue
            blender_name = humanoid_bones[vrm_bone]
            if blender_name not in armature.pose.bones:
                continue
            pb = armature.pose.bones[blender_name]
            i = BODY_JOINT_NAMES.index(smplx_name)
            pb.rotation_quaternion = _aa_to_blender_quat(anim.body_pose[t, i])
            pb.keyframe_insert("rotation_quaternion")

        # Hands
        for hand_pose, hand_map in [
            (anim.left_hand_pose, lhand_map),
            (anim.right_hand_pose, rhand_map),
        ]:
            for i, smplx_name, vrm_bone in hand_map:
                if vrm_bone not in humanoid_bones:
                    continue
                blender_name = humanoid_bones[vrm_bone]
                if blender_name not in armature.pose.bones:
                    continue
                pb = armature.pose.bones[blender_name]
                pb.rotation_quaternion = _aa_to_blender_quat(hand_pose[t, i])
                pb.keyframe_insert("rotation_quaternion")

    bpy.ops.object.mode_set(mode="OBJECT")


def _bake_face(armature, anim: Animation, vrm_metadata: dict, face_mapping) -> None:
    """Mappe jaw_pose et expressions FLAME sur le VRM.

    v1 : seul le jaw bone et les eyes (si présents) reçoivent leurs rotations
    FLAME. Les coefficients d'expression FLAME ne sont PAS mappés sur les
    presets émotionnels VRM (cf. docs/PIPELINE.md §5.2 stratégie A).
    """
    import bpy

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    humanoid_bones = vrm_metadata["humanoid_bones"]

    targets = [
        (face_mapping.jaw_bone,  anim.jaw_pose),
        (face_mapping.leye_bone, anim.leye_pose),
        (face_mapping.reye_bone, anim.reye_pose),
    ]

    for vrm_bone, rotations in targets:
        if vrm_bone is None:
            continue
        if vrm_bone not in humanoid_bones:
            continue
        blender_name = humanoid_bones[vrm_bone]
        if blender_name not in armature.pose.bones:
            continue
        pb = armature.pose.bones[blender_name]
        pb.rotation_mode = "QUATERNION"
        for t in range(anim.num_frames):
            bpy.context.scene.frame_set(t + 1)
            pb.rotation_quaternion = _aa_to_blender_quat(rotations[t])
            pb.keyframe_insert("rotation_quaternion")

    bpy.ops.object.mode_set(mode="OBJECT")


def _export(output_path: Path, fps: float, num_frames: int) -> None:
    """Exporte la scène en .vrma (ou .glb selon l'extension demandée)."""
    import bpy
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ext = output_path.suffix.lower()
    if ext == ".vrma":
        # Export VRM Animation via l'addon
        bpy.ops.export_scene.vrma(filepath=str(output_path))
    elif ext in {".glb", ".gltf"}:
        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            export_format="GLB" if ext == ".glb" else "GLTF_SEPARATE",
            export_animations=True,
            export_bake_animation=True,
            export_frame_range=True,
        )
    else:
        raise ValueError(f"Extension de sortie non supportée : {ext}")


def _aa_to_blender_quat(aa: np.ndarray):
    """Convertit axis-angle (3,) en mathutils.Quaternion (w, x, y, z)."""
    from mathutils import Quaternion, Vector
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    axis = (aa / angle).astype(float)
    return Quaternion(Vector(axis), angle)


if __name__ == "__main__":
    sys.exit(main())
