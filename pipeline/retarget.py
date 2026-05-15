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


def _normalize_vrm_bone_name(name: str) -> str:
    """Normalise un nom de bone VRM vers le camelCase VRM 1.0.

    Accepte plusieurs formats que l'API VRM expose selon le contexte :
        "LEFT_INDEX_PROXIMAL"             → "leftIndexProximal"
        "left_index_proximal"             → "leftIndexProximal"
        "HumanBoneName.LEFT_INDEX_DISTAL" → "leftIndexDistal"
        "leftIndexProximal"               → "leftIndexProximal"
    """
    if "." in name:
        name = name.rsplit(".", 1)[-1]
    if "_" in name:
        parts = name.lower().split("_")
        return parts[0] + "".join(p.capitalize() for p in parts[1:])
    return name


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

    # ── VRM 0.x (merge avec VRM 1.0) ─────────────────────────────────────
    # Beaucoup de VRM "1.0" exportés depuis VRoid contiennent EN PLUS les bones
    # en format 0.x (les "Duplicated VRM0 bone" warnings dans l'addon). Le mapping
    # VRM 1.0 peut être quasi-vide, mais le 0.x est complet. On merge les deux,
    # en privilégiant ce qui a déjà été trouvé en 1.0.
    # IMPORTANT : `hb.bone` en VRM 0.x peut être renvoyé en UPPERCASE_SNAKE_CASE
    # (ex. "LEFT_INDEX_PROXIMAL") par l'enum Python. On normalise vers le
    # camelCase VRM 1.0 ("leftIndexProximal") pour un lookup cohérent.
    vrm0 = getattr(ext, "vrm0", None)
    if vrm0 is not None and getattr(vrm0, "humanoid", None) is not None:
        n_before = len(humanoid_bones)
        for hb in vrm0.humanoid.human_bones:
            vrm_bone_name = getattr(hb, "bone", None)
            node_bone = getattr(getattr(hb, "node", None), "bone_name", None)
            if vrm_bone_name and node_bone:
                normalized = _normalize_vrm_bone_name(str(vrm_bone_name))
                if normalized not in humanoid_bones:
                    humanoid_bones[normalized] = node_bone
        if len(humanoid_bones) > n_before and n_before > 0:
            # VRM 1.0 partiel + VRM 0.x complétant → on garde version=1.0
            logger.info("Bones complétés depuis VRM 0.x : +%d", len(humanoid_bones) - n_before)
        elif n_before == 0:
            version = "0.x"

        blend_master = getattr(vrm0, "blend_shape_master", None)
        if blend_master is not None and not expressions:
            for grp in getattr(blend_master, "blend_shape_groups", []):
                name = getattr(grp, "preset_name", None) or getattr(grp, "name", None)
                if name:
                    expressions.append(str(name).lower())

    if not humanoid_bones:
        raise RuntimeError(
            "Aucun bone humanoïde trouvé via l'API VRM addon. "
            "L'avatar est-il bien un VRM conforme ?"
        )

    # Diagnostic : si des bones de doigts manquent, ça expliquera des animations
    # cassées sur les phalanges (le mapping côté retargeting échoue silencieusement)
    finger_roles = {"leftIndexProximal", "leftIndexIntermediate", "leftIndexDistal"}
    missing_fingers = finger_roles - humanoid_bones.keys()
    if missing_fingers:
        logger.warning("Rôles VRM doigts manquants : %s", sorted(missing_fingers))
        logger.warning("Clés humanoid_bones disponibles (50 premières) : %s",
                       sorted(humanoid_bones.keys())[:50])

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


def _convert_animation_to_amass_npz(anim: Animation, output_path: str,
                                     gender: str = "neutral") -> None:
    """Convertit notre animation.npz au format AMASS attendu par le SMPL-X Blender Add-on.

    Format AMASS requis par bpy.ops.object.smplx_add_animation :
        trans (T, 3), gender (str), mocap_frame_rate (int),
        betas (10,), poses (T, 165) flat axis-angle SMPL-X 55 joints.

    Layout `poses` (offsets en composantes axis-angle) :
        [0:3]    global_orient
        [3:66]   body_pose         (21 joints × 3)
        [66:69]  jaw
        [69:72]  left_eye
        [72:75]  right_eye
        [75:120] left_hand_pose   (15 joints × 3)
        [120:165] right_hand_pose (15 joints × 3)
    """
    import numpy as np

    T = anim.num_frames
    poses = np.zeros((T, 165), dtype=np.float32)
    # global_orient = 0 : SMPLer-X sort cette rotation dans le repère caméra,
    # pas dans le canonique SMPL-X que l'addon AMASS attend. Combiné à la
    # conversion X+90° de l'addon, ça fait basculer le personnage au sol au
    # fil des frames. Pour la LSF on veut un avatar debout face caméra qui
    # signe — les rotations articulaires (body_pose, hand_pose, ...) sont
    # locales joint→parent donc invariantes au repère, on les garde.
    # poses[:, 0:3] reste à zéro.
    poses[:, 3:66] = anim.body_pose.reshape(T, -1)
    poses[:, 66:69] = anim.jaw_pose
    poses[:, 69:72] = anim.leye_pose
    poses[:, 72:75] = anim.reye_pose
    poses[:, 75:120] = anim.left_hand_pose.reshape(T, -1)
    poses[:, 120:165] = anim.right_hand_pose.reshape(T, -1)

    # Pour la LSF on veut que le signeur reste en place sur l'avatar. SMPLer-X
    # produit `transl` en repère caméra (souvent décalé de plusieurs mètres),
    # ce qui téléporterait le personnage. On force trans=0 — Rokoko ne baque
    # alors que les rotations.
    trans = np.zeros((T, 3), dtype=np.float32)

    np.savez(
        output_path,
        trans=trans,
        gender=np.array(gender),
        mocap_frame_rate=np.int32(round(anim.fps)),
        betas=anim.betas.astype(np.float32)[:10],
        poses=poses,
    )


def _override_finger_rotations(src_armature, tgt_armature, vrm_metadata: dict,
                                num_frames: int) -> None:
    """Override les rotations des bones doigts via rest-pose conjugation.

    Pourquoi : Rokoko foire les doigts (orphelins sans source dans son bone_list
    → rotations écrasées). Le diag a confirmé que l'armature SMPL-X contient
    exactement les bonnes rotations locales (diff = 0.0000).

    Le précédent essai (set pose_bone.matrix world-space) alignait les frames
    complets des bones, mais ratait à cause du **bone roll** : SMPL-X et VRoid
    n'ont pas la même orientation locale (axe X vs Y le long du doigt). Résultat :
    fingers à 90° du sens attendu.

    Formule correcte (conjugaison par offset de rest pose) :
        tgt_local_pose = correction @ src_local_pose @ correction⁻¹
        correction = inv(tgt_rest_parent_relative) @ src_rest_parent_relative

    Ça transforme une rotation locale du repère SMPL-X vers le repère VRM tout
    en préservant la sémantique "le doigt se plie de N° autour de son axe naturel".
    """
    import bpy
    from mathutils import Quaternion

    humanoid_bones = vrm_metadata["humanoid_bones"]
    src_pose_bones = src_armature.pose.bones
    tgt_pose_bones = tgt_armature.pose.bones

    finger_pairs: list[tuple[str, str]] = []
    for smplx_name, vrm_role in _SMPLX_TO_VRM_BONE_NAME.items():
        if not any(k in smplx_name for k in ("index", "middle", "ring", "pinky", "thumb")):
            continue
        if smplx_name not in src_pose_bones:
            continue
        target_name = humanoid_bones.get(vrm_role)
        if target_name is None or target_name not in tgt_pose_bones:
            continue
        finger_pairs.append((smplx_name, target_name))

    if not finger_pairs:
        logger.warning("Override doigts : aucune paire trouvée")
        return

    # Pre-calcule l'offset de rest pose par paire (parent-relatif).
    corrections: dict[tuple[str, str], Quaternion] = {}
    for src_name, tgt_name in finger_pairs:
        src_bone = src_armature.data.bones[src_name]
        tgt_bone = tgt_armature.data.bones[tgt_name]
        # matrix_local est en armature space (chaînée à travers les parents).
        # Pour avoir le rest "relatif au parent" on inverse le parent.
        if src_bone.parent:
            src_local_rest = src_bone.parent.matrix_local.inverted() @ src_bone.matrix_local
        else:
            src_local_rest = src_bone.matrix_local.copy()
        if tgt_bone.parent:
            tgt_local_rest = tgt_bone.parent.matrix_local.inverted() @ tgt_bone.matrix_local
        else:
            tgt_local_rest = tgt_bone.matrix_local.copy()
        src_rest_q = src_local_rest.to_3x3().to_quaternion()
        tgt_rest_q = tgt_local_rest.to_3x3().to_quaternion()
        corrections[(src_name, tgt_name)] = tgt_rest_q.inverted() @ src_rest_q

    logger.info("Override doigts (rest-pose conjugation) : %d paires, %d frames",
                len(finger_pairs), num_frames)

    for _, tgt_name in finger_pairs:
        tgt_pose_bones[tgt_name].rotation_mode = "QUATERNION"

    saved_frame = bpy.context.scene.frame_current
    for f in range(num_frames):
        bpy.context.scene.frame_set(f + 1)
        bpy.context.view_layer.update()
        for src_name, tgt_name in finger_pairs:
            src_pb = src_pose_bones[src_name]
            tgt_pb = tgt_pose_bones[tgt_name]
            src_q = src_pb.rotation_quaternion
            corr = corrections[(src_name, tgt_name)]
            tgt_q = corr @ src_q @ corr.inverted()
            tgt_pb.rotation_quaternion = tgt_q
            tgt_pb.keyframe_insert(data_path="rotation_quaternion", frame=f + 1)
    bpy.context.scene.frame_set(saved_frame)
    logger.info("Override doigts terminé")


def _diag_smplx_finger_pose(smplx_armature, anim: Animation, frame: int = 10) -> None:
    """Compare la rotation des bones doigts de l'armature SMPL-X dans Blender
    à la valeur correspondante dans `anim.left_hand_pose[frame]`.

    Si les quaternions matchent → le SMPL-X Add-on a bien chargé nos hand_pose,
    le bug est ailleurs (Rokoko ou export VRM).
    Si les quaternions divergent → l'addon mal-interprète nos données (layout
    AMASS, PCA, axis convention…).
    """
    import bpy
    from mathutils import Quaternion

    finger_order = ["index1", "index2", "index3",
                    "middle1", "middle2", "middle3",
                    "pinky1", "pinky2", "pinky3",
                    "ring1", "ring2", "ring3",
                    "thumb1", "thumb2", "thumb3"]

    bpy.context.scene.frame_set(frame + 1)  # frames Blender 1-indexed
    pose_bones = smplx_armature.pose.bones
    t = min(frame, anim.num_frames - 1)

    def aa_to_quat(aa: np.ndarray) -> tuple:
        angle = float(np.linalg.norm(aa))
        if angle < 1e-7:
            return (1.0, 0.0, 0.0, 0.0)
        axis = aa / angle
        c, s = float(np.cos(angle / 2)), float(np.sin(angle / 2))
        return (c, float(axis[0]) * s, float(axis[1]) * s, float(axis[2]) * s)

    logger.info("=== Diag SMPL-X finger pose @ frame=%d ===", frame)
    max_diff = 0.0
    worst_bone = None
    for side_smplx, pose_array in (("left", anim.left_hand_pose),
                                    ("right", anim.right_hand_pose)):
        for joint_idx, joint_suffix in enumerate(finger_order):
            bone_name = f"{side_smplx}_{joint_suffix}"
            if bone_name not in pose_bones:
                continue
            pb = pose_bones[bone_name]
            actual_q = (pb.rotation_quaternion.w, pb.rotation_quaternion.x,
                        pb.rotation_quaternion.y, pb.rotation_quaternion.z)
            expected_q = aa_to_quat(pose_array[t, joint_idx])
            # Distance L2 entre les 2 quaternions (ou leur inverse, ambigus de signe)
            diff = min(
                sum((a - b) ** 2 for a, b in zip(actual_q, expected_q)) ** 0.5,
                sum((a + b) ** 2 for a, b in zip(actual_q, expected_q)) ** 0.5,
            )
            if diff > max_diff:
                max_diff = diff
                worst_bone = bone_name
            # Log les 3 premiers en détail
            if joint_idx < 3 and side_smplx == "left":
                logger.info(
                    "  %s expected (w,x,y,z)=%s actual=%s diff=%.4f",
                    bone_name,
                    tuple(round(v, 3) for v in expected_q),
                    tuple(round(v, 3) for v in actual_q),
                    diff,
                )

    logger.info("Max diff sur 30 bones doigts : %.4f (pire : %s)",
                max_diff, worst_bone)
    if max_diff < 0.01:
        logger.info("→ SMPL-X armature contient exactement nos poses. Bug downstream.")
    elif max_diff < 0.1:
        logger.info("→ Petites divergences, l'addon ajoute une rest pose offset peut-être.")
    else:
        logger.warning("→ Gros écart, l'addon n'applique PAS nos hand_pose tel quel.")


def _bake_rotations_manual(src_armature, tgt_armature, vrm_metadata: dict,
                            num_frames: int) -> None:
    """Bake direct des rotations locales source SMPL-X → target VRM.

    Hypothèse : SMPL-X et VRM partagent la convention humanoid T-pose
    canonique. Les rotations locales (joint relatif à son parent dans la rest
    pose) sont donc directement transférables d'une armature à l'autre.

    Pour chaque frame :
        - Avance la scène (le SMPL-X Add-on a baked les keyframes sur l'armature
          source, donc pose_bone.rotation_quaternion = pose à cette frame)
        - Pour chaque (src_name, tgt_name) du mapping _SMPLX_TO_VRM_BONE_NAME :
            copie src_pb.rotation_quaternion → tgt_pb.rotation_quaternion
            insère un keyframe sur le target.
    """
    import bpy

    humanoid_bones = vrm_metadata["humanoid_bones"]
    src_pose_bones = src_armature.pose.bones
    tgt_pose_bones = tgt_armature.pose.bones

    pairs: list[tuple[str, str]] = []
    for smplx_name, vrm_role in _SMPLX_TO_VRM_BONE_NAME.items():
        if smplx_name not in src_pose_bones:
            continue
        target_name = humanoid_bones.get(vrm_role)
        if target_name is None or target_name not in tgt_pose_bones:
            continue
        pairs.append((smplx_name, target_name))

    logger.info("Bake manuel : %d paires sur %d frames", len(pairs), num_frames)

    for _, tgt_name in pairs:
        tgt_pose_bones[tgt_name].rotation_mode = "QUATERNION"

    # Crée l'action sur le target si elle n'existe pas
    if tgt_armature.animation_data is None:
        tgt_armature.animation_data_create()
    if tgt_armature.animation_data.action is None:
        action = bpy.data.actions.new(name=f"{tgt_armature.name}_retargeted")
        tgt_armature.animation_data.action = action

    for f in range(num_frames):
        bpy.context.scene.frame_set(f + 1)
        for src_name, tgt_name in pairs:
            src_pb = src_pose_bones[src_name]
            tgt_pb = tgt_pose_bones[tgt_name]
            q = src_pb.rotation_quaternion
            tgt_pb.rotation_quaternion = (q.w, q.x, q.y, q.z)
            tgt_pb.keyframe_insert(data_path="rotation_quaternion", frame=f + 1)

    logger.info("Bake manuel terminé : %d keyframes (%d frames × %d bones)",
                num_frames * len(pairs), num_frames, len(pairs))


def _bake_animation(armature, anim: Animation, vrm_metadata: dict) -> None:
    """Retargete l'animation SMPL-X sur le VRM via SMPL-X Add-on + Rokoko.

    Étapes :
    1. Convertit anim → NPZ AMASS dans un fichier temp.
    2. SMPL-X Blender Add-on importe l'animation sur une armature source.
    3. Rokoko Studio Live retargete les rotations vers l'armature VRM.
    4. Diagnostic : compte les keyframes par bone target pour vérifier que
       l'animation est bien baked sur toute la durée et pas juste frame 0.
    5. Cleanup : retire l'armature source.
    """
    import bpy, addon_utils
    import tempfile
    from pathlib import Path as _Path

    addon_utils.enable("bl_ext.user_default.smplx_blender_addon")
    addon_utils.enable("rokoko_studio_live_blender")

    # 2. Sauvegarde notre anim au format AMASS dans un temp file
    tmp_dir = _Path(tempfile.mkdtemp(prefix="lsf_amass_"))
    amass_path = tmp_dir / "animation_amass.npz"
    _convert_animation_to_amass_npz(anim, str(amass_path))
    logger.info("Animation convertie au format AMASS : %s", amass_path)

    # 3. Importe via l'opérateur SMPL-X Add-on (crée armature + mesh + anim baked)
    target_framerate = int(round(anim.fps))
    # Note : le mode_set ferme bien le contexte si on est en autre chose
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    # anim_format="SMPL-X" : l'addon n'applique aucune rotation supplémentaire.
    # On a déjà neutralisé global_orient (poses[:,0:3]=0) et trans (zéros)
    # parce qu'ils viennent du repère caméra SMPLer-X et donneraient des
    # rotations/positions parasites. Les body_pose, hand_pose etc. sont des
    # rotations locales joint→parent, valides telles quelles.
    # Avec "AMASS", l'addon ajouterait sa propre conversion X+90° qui tilt le
    # personnage à l'horizontale.
    bpy.ops.object.smplx_add_animation(
        filepath=str(amass_path),
        anim_format="SMPL-X",
        target_framerate=target_framerate,
    )

    # 4. Trouve l'armature SMPL-X qui vient d'être créée
    # L'addon crée un mesh + son parent armature. On cherche la dernière armature ajoutée.
    smplx_armature = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and obj.name != armature.name:
            # Heuristique : l'armature SMPL-X a un nom comme "SMPLX_neutral_..."
            if "smplx" in obj.name.lower() or "smpl" in obj.name.lower():
                smplx_armature = obj
                break
    if smplx_armature is None:
        raise RuntimeError("Armature SMPL-X non trouvée après smplx_add_animation")
    logger.info("Armature source SMPL-X : %s", smplx_armature.name)
    logger.info("Armature target VRM    : %s", armature.name)

    # Diagnostic : compare la rotation des bones de doigts dans l'armature
    # SMPL-X (post smplx_add_animation) à ce que notre NPZ leur prescrit. Si
    # écart → l'addon mal-interprète nos hand_pose (PCA, layout AMASS différent,
    # etc.). Si égalité → la pose est correcte côté SMPL-X et le bug est
    # downstream (Rokoko ou export VRM).
    _diag_smplx_finger_pose(smplx_armature, anim, frame=10)

    # 5. Configure Rokoko et bake le retargeting.
    bpy.context.scene.rsl_retargeting_armature_source = smplx_armature
    bpy.context.scene.rsl_retargeting_armature_target = armature
    _populate_rokoko_bone_mapping(smplx_armature, armature, vrm_metadata)
    logger.info("Rokoko bone list construite — lancement retarget…")

    # Étend la scene frame range pour s'assurer que Rokoko itère bien sur
    # toutes les frames de la source action, pas juste sur [1, 250] par défaut.
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = anim.num_frames

    bpy.ops.rsl.retarget_animation()
    logger.info("Rokoko retargeting terminé")

    # Rokoko foire les doigts (génère des orphelins sans source pour certaines
    # phalanges → rotations effacées au lieu d'être transférées). Diag confirme
    # que l'armature SMPL-X a les bonnes rotations, donc on les copie nous-mêmes
    # depuis SMPL-X vers VRM, frame par frame, juste pour les bones de doigts.
    _override_finger_rotations(smplx_armature, armature, vrm_metadata,
                                anim.num_frames)

    # Diagnostic : compte les keyframes par bone du VRM. Si N=1 partout → Rokoko
    # n'a baked que frame 0. Si N=num_frames → bake complet.
    if armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        bone_kf_counts: dict[str, int] = {}
        for fc in action.fcurves:
            if "pose.bones[" in fc.data_path:
                bone_name = fc.data_path.split('"')[1]
                bone_kf_counts.setdefault(bone_name, 0)
                bone_kf_counts[bone_name] = max(bone_kf_counts[bone_name],
                                                 len(fc.keyframe_points))
        if bone_kf_counts:
            kf_values = list(bone_kf_counts.values())
            logger.info(
                "Keyframes par bone : min=%d max=%d moyenne=%.1f (sur %d bones, attendu %d)",
                min(kf_values), max(kf_values), sum(kf_values) / len(kf_values),
                len(bone_kf_counts), anim.num_frames,
            )
            # Liste les bones avec très peu de keyframes (suspect)
            poor = sorted([(n, c) for n, c in bone_kf_counts.items() if c < anim.num_frames // 2])
            if poor:
                logger.warning("Bones sous-baked (< %d kf) : %s",
                               anim.num_frames // 2, poor[:10])

    # 6. Bloque la position du hips à sa rest pose.
    # Rokoko copie la location world-space du pelvis source vers le hips du VRM.
    # Source pelvis = (0,0,0) (on a forcé trans=0), donc le hips VRM tombe à
    # l'origine du monde et le perso s'enterre dans le sol. On supprime les
    # fcurves de location sur le hips pour qu'il reste à sa rest position.
    # On garde la rotation_quaternion du hips (utile pour le body sway).
    hips_bone_name = vrm_metadata["humanoid_bones"].get("hips")
    if hips_bone_name and armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        loc_path = f'pose.bones["{hips_bone_name}"].location'
        to_remove = [fc for fc in action.fcurves if fc.data_path == loc_path]
        for fc in to_remove:
            action.fcurves.remove(fc)
        logger.info("Hips location bloqué : %d fcurves supprimées", len(to_remove))
        # Reset la pose location au cas où une valeur résiduelle traîne
        if hips_bone_name in armature.pose.bones:
            armature.pose.bones[hips_bone_name].location = (0.0, 0.0, 0.0)

    # 7. Cleanup armature source + son mesh (on veut export VRM seulement)
    for obj in list(bpy.data.objects):
        if obj == smplx_armature or (obj.parent == smplx_armature):
            bpy.data.objects.remove(obj, do_unlink=True)


# Mapping SMPL-X canonical joint name → VRM standard humanoid bone name.
# Le SMPL-X Blender Add-on nomme les bones avec les conventions SMPL-X
# (snake_case). Le VRM utilise camelCase. On bridge les deux ici pour Rokoko.
_SMPLX_TO_VRM_BONE_NAME: dict[str, str] = {
    "pelvis": "hips",
    "left_hip": "leftUpperLeg",
    "right_hip": "rightUpperLeg",
    "spine1": "spine",
    "left_knee": "leftLowerLeg",
    "right_knee": "rightLowerLeg",
    "spine2": "chest",
    "left_ankle": "leftFoot",
    "right_ankle": "rightFoot",
    "spine3": "upperChest",
    "left_foot": "leftToes",
    "right_foot": "rightToes",
    "neck": "neck",
    "left_collar": "leftShoulder",
    "right_collar": "rightShoulder",
    "head": "head",
    "left_shoulder": "leftUpperArm",
    "right_shoulder": "rightUpperArm",
    "left_elbow": "leftLowerArm",
    "right_elbow": "rightLowerArm",
    "left_wrist": "leftHand",
    "right_wrist": "rightHand",
    "jaw": "jaw",
    "left_eye_smplhf": "leftEye",
    "right_eye_smplhf": "rightEye",
    # Doigts gauche
    "left_index1": "leftIndexProximal",
    "left_index2": "leftIndexIntermediate",
    "left_index3": "leftIndexDistal",
    "left_middle1": "leftMiddleProximal",
    "left_middle2": "leftMiddleIntermediate",
    "left_middle3": "leftMiddleDistal",
    "left_ring1": "leftRingProximal",
    "left_ring2": "leftRingIntermediate",
    "left_ring3": "leftRingDistal",
    "left_pinky1": "leftLittleProximal",
    "left_pinky2": "leftLittleIntermediate",
    "left_pinky3": "leftLittleDistal",
    "left_thumb1": "leftThumbProximal",
    "left_thumb2": "leftThumbIntermediate",
    "left_thumb3": "leftThumbDistal",
    # Doigts droite (mêmes suffixes)
    "right_index1": "rightIndexProximal",
    "right_index2": "rightIndexIntermediate",
    "right_index3": "rightIndexDistal",
    "right_middle1": "rightMiddleProximal",
    "right_middle2": "rightMiddleIntermediate",
    "right_middle3": "rightMiddleDistal",
    "right_ring1": "rightRingProximal",
    "right_ring2": "rightRingIntermediate",
    "right_ring3": "rightRingDistal",
    "right_pinky1": "rightLittleProximal",
    "right_pinky2": "rightLittleIntermediate",
    "right_pinky3": "rightLittleDistal",
    "right_thumb1": "rightThumbProximal",
    "right_thumb2": "rightThumbIntermediate",
    "right_thumb3": "rightThumbDistal",
}


def _populate_rokoko_bone_mapping(source_arm, target_arm, vrm_metadata: dict) -> None:
    """Construit `bpy.context.scene.rsl_retargeting_bone_list` from scratch
    avec un mapping explicite SMPL-X (source) → VRM bone Blender name (target).

    On clear la liste auto-générée par `build_bone_list()` (qui peut avoir
    matché certains bones aléatoirement et créer des conflits target) puis on
    rebuild avec notre table _SMPLX_TO_VRM_BONE_NAME + humanoid_bones.
    """
    import bpy

    humanoid_bones = vrm_metadata["humanoid_bones"]  # vrm_role → blender_bone_name
    bone_list = bpy.context.scene.rsl_retargeting_bone_list

    source_bone_names = {b.name for b in source_arm.data.bones}
    target_bone_names = {b.name for b in target_arm.data.bones}

    logger.info("Source SMPL-X bones (%d) : ex. %s",
                len(source_bone_names),
                sorted(source_bone_names)[:5])

    # Clear robuste : `.clear()` peut être muet selon la version Blender, on
    # supprime explicitement chaque item.
    while len(bone_list) > 0:
        bone_list.remove(0)

    # Sonde une fois les attributs réels (varient selon les versions Rokoko)
    probe = bone_list.add()
    item_attrs = [a for a in dir(probe) if not a.startswith("_") and not callable(getattr(probe, a, None))]
    bone_list.remove(0)
    logger.info("Rokoko bone_list item attrs : %s", item_attrs)

    n_mapped = 0
    n_target_missing = 0
    n_source_missing = 0
    used_targets: set[str] = set()
    missing_roles: list[str] = []

    for smplx_name, vrm_role in _SMPLX_TO_VRM_BONE_NAME.items():
        if smplx_name not in source_bone_names:
            n_source_missing += 1
            continue
        target_name = humanoid_bones.get(vrm_role)
        if target_name is None or target_name not in target_bone_names:
            n_target_missing += 1
            missing_roles.append(vrm_role)
            continue
        if target_name in used_targets:
            continue

        item = bone_list.add()
        # `name` est ce que Rokoko utilise comme label affiché et comme clé
        # dans le bone_list. `bone_name_key` semble être la "clé canonique"
        # (équivalent au nom standard VRM 1.0 quand connu). `bone_name_source`
        # est le nom du bone côté source armature. Set les trois cohérents.
        if hasattr(item, "name"):
            item.name = smplx_name
        if hasattr(item, "bone_name_key"):
            item.bone_name_key = smplx_name
        if hasattr(item, "bone_name_source"):
            item.bone_name_source = smplx_name
        if hasattr(item, "bone_name_target"):
            item.bone_name_target = target_name
        # Force is_custom=True : nos noms SMPL-X (snake_case) ne sont pas
        # standards Rokoko, l'auto-detect risque de les écraser sinon.
        if hasattr(item, "is_custom"):
            item.is_custom = True
        used_targets.add(target_name)
        n_mapped += 1

    logger.info(
        "Rokoko mapping : %d mappés / %d source manquant / %d target manquant",
        n_mapped, n_source_missing, n_target_missing,
    )
    if missing_roles:
        logger.warning("Rôles VRM non-mappés : %s", missing_roles[:20])

    # Dump complet du bone_list juste avant retarget_animation : on saura
    # exactement ce que Rokoko voit.
    logger.info("Dump bone_list (%d items) avant retarget :", len(bone_list))
    for i, item in enumerate(bone_list):
        logger.info(
            "  [%d] name=%r key=%r source=%r target=%r custom=%s",
            i,
            getattr(item, "name", ""),
            getattr(item, "bone_name_key", ""),
            getattr(item, "bone_name_source", ""),
            getattr(item, "bone_name_target", ""),
            getattr(item, "is_custom", None),
        )


def _bake_animation_OLD_LOOKAT(armature, anim: Animation, vrm_metadata: dict) -> None:
    """Ancien retargeting via FK + look-at. Conservé pour référence/fallback.

    Limitations connues :
    - Twist autour du bone perdu (paumes/doigts dans orientation aléatoire)
    - Pas de gestion fine des rest pose différences SMPL-X / VRM
    """
    import bpy
    import numpy as np
    from mathutils import Vector

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")

    humanoid_bones = vrm_metadata["humanoid_bones"]

    # Charge le squelette SMPL-X (rest joints + parents)
    rest_joints, parents = _load_smplx_rest_skeleton()

    # Quaternion mode pour tous les bones humanoïdes
    for vrm_bone_name in humanoid_bones:
        blender_bone = humanoid_bones[vrm_bone_name]
        if blender_bone in armature.pose.bones:
            armature.pose.bones[blender_bone].rotation_mode = "QUATERNION"

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = anim.num_frames
    bpy.context.scene.render.fps = int(round(anim.fps))

    # Bones ordonnés topologiquement (parents avant enfants).
    ordered_bones: list[str] = [
        "hips", "spine", "chest", "upperChest", "neck",
        "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
        "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
        "leftUpperLeg", "leftLowerLeg", "leftFoot",
        "rightUpperLeg", "rightLowerLeg", "rightFoot",
        # Mains : 5 doigts × 3 phalanges × 2 mains
        "leftThumbProximal", "leftThumbIntermediate", "leftThumbDistal",
        "leftIndexProximal", "leftIndexIntermediate", "leftIndexDistal",
        "leftMiddleProximal", "leftMiddleIntermediate", "leftMiddleDistal",
        "leftRingProximal", "leftRingIntermediate", "leftRingDistal",
        "leftLittleProximal", "leftLittleIntermediate", "leftLittleDistal",
        "rightThumbProximal", "rightThumbIntermediate", "rightThumbDistal",
        "rightIndexProximal", "rightIndexIntermediate", "rightIndexDistal",
        "rightMiddleProximal", "rightMiddleIntermediate", "rightMiddleDistal",
        "rightRingProximal", "rightRingIntermediate", "rightRingDistal",
        "rightLittleProximal", "rightLittleIntermediate", "rightLittleDistal",
    ]

    from mathutils import Matrix

    for t in range(anim.num_frames):
        bpy.context.scene.frame_set(t + 1)

        # FK SMPL-X complet (55 joints : body + face + 2 mains)
        joints_smplx, rot_world_smplx = _smplx_fk_full(rest_joints, parents, anim, t)

        # Conversion frame SMPL-X → Blender. Empirique (cumulant Y-down de SMPL-X
        # et forward direction VRM) : (x, y, z) → (x, z, -y).
        joints_blender = np.column_stack([
            joints_smplx[:, 0],
            joints_smplx[:, 2],
            -joints_smplx[:, 1],
        ])

        # On track nos propres pose matrices (en armature space) au lieu de
        # se reposer sur pb.matrix qui n'est pas reliably mis à jour pour les
        # enfants quand on modifie le parent dans la même frame.
        pose_matrix_cache: dict[str, Matrix] = {}

        for vrm_bone in ordered_bones:
            seg = _VRM_BONE_SEGMENTS.get(vrm_bone)
            if seg is None or vrm_bone not in humanoid_bones:
                continue
            blender_name = humanoid_bones[vrm_bone]
            if blender_name not in armature.pose.bones:
                continue
            pb = armature.pose.bones[blender_name]

            start_idx, end_idx = seg
            target_dir_world = Vector(joints_blender[end_idx] - joints_blender[start_idx])
            if target_dir_world.length < 1e-6:
                continue
            target_dir_world.normalize()

            # Parent pose dans armature space (notre cache, pas Blender)
            parent_pb = pb.parent
            if parent_pb is not None and parent_pb.name in pose_matrix_cache:
                parent_pose = pose_matrix_cache[parent_pb.name]
                parent_rest = parent_pb.bone.matrix_local
                rest_local_relative = parent_rest.inverted() @ pb.bone.matrix_local
            else:
                parent_pose = Matrix.Identity(4)
                rest_local_relative = pb.bone.matrix_local

            # Pose sans rotation locale = parent_pose @ rest_local_relative
            current_world = parent_pose @ rest_local_relative

            # Convertir target_dir (armature space) en frame du bone (post parent + rest)
            current_world_3x3_inv = current_world.to_3x3().inverted()
            target_in_bone_frame = (current_world_3x3_inv @ target_dir_world).normalized()

            # Look-at simple : aligne axe Y du bone sur target_dir.
            # Le twist (palms, orientation des phalanges) reste libre — limitation du look-at.
            local_rot_q = Vector((0.0, 1.0, 0.0)).rotation_difference(target_in_bone_frame)
            pb.rotation_quaternion = local_rot_q
            pb.keyframe_insert("rotation_quaternion")

            # Met à jour le cache pour les enfants
            final_world = current_world @ local_rot_q.to_matrix().to_4x4()
            pose_matrix_cache[blender_name] = final_world

    bpy.ops.object.mode_set(mode="OBJECT")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers FK SMPL-X + look-at retargeting
# ──────────────────────────────────────────────────────────────────────────────

# Indices canoniques des joints SMPL-X (55 au total).
# Source : smplx.JOINT_NAMES.
_SMPLX_JOINT_IDX = {
    # Body (0..21)
    "pelvis": 0,
    "left_hip": 1, "right_hip": 2, "spine1": 3,
    "left_knee": 4, "right_knee": 5, "spine2": 6,
    "left_ankle": 7, "right_ankle": 8, "spine3": 9,
    "left_foot": 10, "right_foot": 11, "neck": 12,
    "left_collar": 13, "right_collar": 14, "head": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
    # Face (22..24)
    "jaw": 22, "left_eye_smplhf": 23, "right_eye_smplhf": 24,
    # Left hand (25..39)
    "left_index1": 25, "left_index2": 26, "left_index3": 27,
    "left_middle1": 28, "left_middle2": 29, "left_middle3": 30,
    "left_pinky1": 31, "left_pinky2": 32, "left_pinky3": 33,
    "left_ring1": 34, "left_ring2": 35, "left_ring3": 36,
    "left_thumb1": 37, "left_thumb2": 38, "left_thumb3": 39,
    # Right hand (40..54)
    "right_index1": 40, "right_index2": 41, "right_index3": 42,
    "right_middle1": 43, "right_middle2": 44, "right_middle3": 45,
    "right_pinky1": 46, "right_pinky2": 47, "right_pinky3": 48,
    "right_ring1": 49, "right_ring2": 50, "right_ring3": 51,
    "right_thumb1": 52, "right_thumb2": 53, "right_thumb3": 54,
}


def _seg(start: str, end: str) -> tuple[int, int]:
    return (_SMPLX_JOINT_IDX[start], _SMPLX_JOINT_IDX[end])


# Mapping VRM bone → (start_joint_idx, end_joint_idx) en indices SMPL-X.
# Le bone VRM est orienté du premier vers le second joint.
_VRM_BONE_SEGMENTS: dict[str, tuple[int, int]] = {
    # Body
    "hips":          _seg("pelvis", "spine1"),
    "spine":         _seg("spine1", "spine2"),
    "chest":         _seg("spine2", "spine3"),
    "upperChest":    _seg("spine3", "neck"),
    "neck":          _seg("neck", "head"),
    "leftShoulder":  _seg("spine3", "left_shoulder"),
    "leftUpperArm":  _seg("left_shoulder", "left_elbow"),
    "leftLowerArm":  _seg("left_elbow", "left_wrist"),
    "leftHand":      _seg("left_wrist", "left_middle1"),
    "rightShoulder": _seg("spine3", "right_shoulder"),
    "rightUpperArm": _seg("right_shoulder", "right_elbow"),
    "rightLowerArm": _seg("right_elbow", "right_wrist"),
    "rightHand":     _seg("right_wrist", "right_middle1"),
    "leftUpperLeg":  _seg("left_hip", "left_knee"),
    "leftLowerLeg":  _seg("left_knee", "left_ankle"),
    "leftFoot":      _seg("left_ankle", "left_foot"),
    "rightUpperLeg": _seg("right_hip", "right_knee"),
    "rightLowerLeg": _seg("right_knee", "right_ankle"),
    "rightFoot":     _seg("right_ankle", "right_foot"),
    # Left hand (15 bones)
    "leftThumbProximal":      _seg("left_thumb1", "left_thumb2"),
    "leftThumbIntermediate":  _seg("left_thumb2", "left_thumb3"),
    "leftThumbDistal":        _seg("left_thumb2", "left_thumb3"),
    "leftIndexProximal":      _seg("left_index1", "left_index2"),
    "leftIndexIntermediate":  _seg("left_index2", "left_index3"),
    "leftIndexDistal":        _seg("left_index2", "left_index3"),
    "leftMiddleProximal":     _seg("left_middle1", "left_middle2"),
    "leftMiddleIntermediate": _seg("left_middle2", "left_middle3"),
    "leftMiddleDistal":       _seg("left_middle2", "left_middle3"),
    "leftRingProximal":       _seg("left_ring1", "left_ring2"),
    "leftRingIntermediate":   _seg("left_ring2", "left_ring3"),
    "leftRingDistal":         _seg("left_ring2", "left_ring3"),
    "leftLittleProximal":     _seg("left_pinky1", "left_pinky2"),
    "leftLittleIntermediate": _seg("left_pinky2", "left_pinky3"),
    "leftLittleDistal":       _seg("left_pinky2", "left_pinky3"),
    # Right hand (15 bones) — symétrique
    "rightThumbProximal":      _seg("right_thumb1", "right_thumb2"),
    "rightThumbIntermediate":  _seg("right_thumb2", "right_thumb3"),
    "rightThumbDistal":        _seg("right_thumb2", "right_thumb3"),
    "rightIndexProximal":      _seg("right_index1", "right_index2"),
    "rightIndexIntermediate":  _seg("right_index2", "right_index3"),
    "rightIndexDistal":        _seg("right_index2", "right_index3"),
    "rightMiddleProximal":     _seg("right_middle1", "right_middle2"),
    "rightMiddleIntermediate": _seg("right_middle2", "right_middle3"),
    "rightMiddleDistal":       _seg("right_middle2", "right_middle3"),
    "rightRingProximal":       _seg("right_ring1", "right_ring2"),
    "rightRingIntermediate":   _seg("right_ring2", "right_ring3"),
    "rightRingDistal":         _seg("right_ring2", "right_ring3"),
    "rightLittleProximal":     _seg("right_pinky1", "right_pinky2"),
    "rightLittleIntermediate": _seg("right_pinky2", "right_pinky3"),
    "rightLittleDistal":       _seg("right_pinky2", "right_pinky3"),
}


def _load_smplx_rest_skeleton():
    """Charge tous les rest joints SMPL-X (55) + table des parents.

    Returns:
        rest_joints (55, 3) : positions des 55 joints en pose canonique (frame SMPL-X)
        parents (55,)       : indice du parent pour chaque joint (-1 ou très grand pour racine)
    """
    import numpy as np
    npz_path = REPO_ROOT / "pipeline" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"SMPLX_NEUTRAL.npz introuvable : {npz_path}. Lancer scripts/download_weights.sh"
        )
    data = np.load(npz_path, allow_pickle=True)
    v_template = data["v_template"]
    J_regressor = data["J_regressor"]
    if hasattr(J_regressor, "toarray"):
        J_regressor = J_regressor.toarray()
    rest_joints = J_regressor @ v_template
    kintree = data["kintree_table"]
    parents = kintree[0].astype(np.int64)
    return rest_joints.astype(np.float64), parents


def _aa_to_rot_mat(aa):
    """Rodrigues : axis-angle (3,) → matrice de rotation (3, 3)."""
    import numpy as np
    aa = np.asarray(aa, dtype=np.float64)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3)
    k = aa / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _smplx_fk_full(rest_joints, parents, anim, t: int):
    """Forward kinematic SMPL-X complet (55 joints : body + face + hands).

    Args:
        rest_joints (55, 3) : positions rest canoniques
        parents (55,)        : indices parents
        anim                 : Animation à la frame t (utilise global_orient, body_pose,
                               jaw_pose, leye_pose, reye_pose, left/right_hand_pose)
        t                    : indice de frame

    Returns:
        joints_world (55, 3) : positions monde après application des rotations
        rot_world (55, 3, 3) : rotation monde de chaque joint (pour extraction du twist)
    """
    import numpy as np

    n = 55
    rotations = np.zeros((n, 3, 3), dtype=np.float64)
    # 0 = pelvis = global_orient
    rotations[0] = _aa_to_rot_mat(anim.global_orient[t])
    # 1..21 = body_pose
    for i in range(21):
        rotations[i + 1] = _aa_to_rot_mat(anim.body_pose[t, i])
    # 22 = jaw, 23 = left eye, 24 = right eye
    rotations[22] = _aa_to_rot_mat(anim.jaw_pose[t])
    rotations[23] = _aa_to_rot_mat(anim.leye_pose[t])
    rotations[24] = _aa_to_rot_mat(anim.reye_pose[t])
    # 25..39 = left hand (15 joints)
    for i in range(15):
        rotations[25 + i] = _aa_to_rot_mat(anim.left_hand_pose[t, i])
    # 40..54 = right hand (15 joints)
    for i in range(15):
        rotations[40 + i] = _aa_to_rot_mat(anim.right_hand_pose[t, i])

    joints_world = np.zeros((n, 3), dtype=np.float64)
    rot_world = np.zeros((n, 3, 3), dtype=np.float64)
    joints_world[0] = rest_joints[0]
    rot_world[0] = rotations[0]
    for i in range(1, n):
        p = int(parents[i])
        if p < 0 or p >= n:
            joints_world[i] = rest_joints[i]
            rot_world[i] = rotations[i]
            continue
        rel = rest_joints[i] - rest_joints[p]
        joints_world[i] = joints_world[p] + rot_world[p] @ rel
        rot_world[i] = rot_world[p] @ rotations[i]
    return joints_world, rot_world


# Mapping VRM bone → indice SMPL-X joint (joint à la tête du bone, dont la rotation
# définit l'orientation/twist du bone enfant qui en sort).
_VRM_BONE_TO_SMPLX_JOINT_HEAD: dict[str, int] = {
    "hips": 0,            # pelvis
    "spine": 3,           # spine1
    "chest": 6,           # spine2
    "upperChest": 9,      # spine3
    "neck": 12,
    "leftShoulder": 13,   # left_collar
    "leftUpperArm": 16,   # left_shoulder
    "leftLowerArm": 18,   # left_elbow
    "leftHand": 20,       # left_wrist
    "rightShoulder": 14,
    "rightUpperArm": 17,
    "rightLowerArm": 19,
    "rightHand": 21,
    "leftUpperLeg": 1,
    "leftLowerLeg": 4,
    "leftFoot": 7,
    "rightUpperLeg": 2,
    "rightLowerLeg": 5,
    "rightFoot": 8,
    # Doigts : joint à la tête de la phalange
    "leftThumbProximal": 37, "leftThumbIntermediate": 38, "leftThumbDistal": 39,
    "leftIndexProximal": 25, "leftIndexIntermediate": 26, "leftIndexDistal": 27,
    "leftMiddleProximal": 28, "leftMiddleIntermediate": 29, "leftMiddleDistal": 30,
    "leftRingProximal": 34, "leftRingIntermediate": 35, "leftRingDistal": 36,
    "leftLittleProximal": 31, "leftLittleIntermediate": 32, "leftLittleDistal": 33,
    "rightThumbProximal": 52, "rightThumbIntermediate": 53, "rightThumbDistal": 54,
    "rightIndexProximal": 40, "rightIndexIntermediate": 41, "rightIndexDistal": 42,
    "rightMiddleProximal": 43, "rightMiddleIntermediate": 44, "rightMiddleDistal": 45,
    "rightRingProximal": 49, "rightRingIntermediate": 50, "rightRingDistal": 51,
    "rightLittleProximal": 46, "rightLittleIntermediate": 47, "rightLittleDistal": 48,
}


# Matrice de changement de base SMPL-X → Blender (cohérent avec joints_blender).
# SMPL-X (x, y, z) → Blender (x, z, -y).
import numpy as _np
_SMPLX_TO_BLENDER_BASIS = _np.array([
    [1.0, 0.0, 0.0],   # X stays
    [0.0, 0.0, 1.0],   # SMPL-X Y → Blender Z
    [0.0, -1.0, 0.0],  # SMPL-X Z → Blender -Y
], dtype=_np.float64)


def _extract_twist_around_y(R_world_smplx, bone_rest_world_3x3):
    """Extrait l'angle de twist (rotation autour de l'axe Y du bone) depuis la
    rotation monde SMPL-X.

    Args:
        R_world_smplx (3, 3) : rotation monde du joint en frame SMPL-X
        bone_rest_world_3x3 (mathutils.Matrix 3x3) : matrice de rest du bone en frame Blender (armature space)

    Returns:
        twist_quat (mathutils.Quaternion) : rotation pure autour de Y dans le frame local du bone
    """
    import numpy as np
    from mathutils import Matrix, Quaternion

    # Conversion SMPL-X → Blender
    B = _SMPLX_TO_BLENDER_BASIS
    R_blender = B @ R_world_smplx @ B.T

    # Conversion en bone local frame : R_local = rest^-1 @ R_world @ rest
    rest_np = np.array([[bone_rest_world_3x3[i][j] for j in range(3)] for i in range(3)],
                        dtype=np.float64)
    R_local_np = rest_np.T @ R_blender @ rest_np

    # Convertir en quaternion via mathutils
    R_local_mat = Matrix([list(row) for row in R_local_np])
    q_local = R_local_mat.to_quaternion()

    # Twist = composante de q autour de Y local (= (0, 1, 0))
    twist = Quaternion((q_local.w, 0.0, q_local.y, 0.0))
    if twist.magnitude < 1e-8:
        return Quaternion()  # identité
    twist.normalize()
    return twist


def _retarget_bone_lookat(pb, target_dir_armature) -> None:
    """Oriente un pose bone pour pointer dans target_dir (en armature space).

    Calcule la rotation qui aligne la direction Y locale du bone (= direction
    head→tail) sur target_dir, puis set pb.matrix pour appliquer.
    Blender backsolve rotation_quaternion automatiquement.

    Le twist autour de la direction n'est pas contrôlé (limitation du look-at).
    """
    from mathutils import Vector

    rest_mat = pb.bone.matrix_local  # 4x4 in armature space
    rest_rot = rest_mat.to_3x3()
    # Direction du bone au repos en armature space : axe Y du bone local
    rest_dir = (rest_rot @ Vector((0.0, 1.0, 0.0))).normalized()

    target = Vector(target_dir_armature).normalized()
    align_quat = rest_dir.rotation_difference(target)

    # Nouvelle pose matrix en armature space : align_quat ∘ rest_rot
    new_rot_4x4 = align_quat.to_matrix().to_4x4() @ rest_rot.to_4x4()
    new_mat = new_rot_4x4.copy()
    new_mat.translation = rest_mat.translation
    pb.matrix = new_mat


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
    """Convertit axis-angle (3,) SMPL-X (Y-up) en mathutils.Quaternion Blender (Z-up).

    SMPL-X : Y up, Z forward (vers la caméra)
    Blender + VRM : Z up
    Le rest pose VRM importé est tourné de 180° autour de X par rapport à la
    base SMPL-X (le corps est upside-down). Pour corriger en basis change :
    une rotation autour de (ax, ay, az) devient une rotation autour de
    (ax, -ay, -az) — angle inchangé.

    NOTE : ça fixe l'orientation globale (corps à l'endroit). Mais les rotations
    de bones individuels (coudes, genoux…) restent dépendantes du delta entre
    rest pose SMPL-X (A-pose) et VRM (T-pose) — pas encore compensé ici.
    """
    from mathutils import Quaternion, Vector
    aa_blender = np.array([aa[0], -aa[1], -aa[2]], dtype=float)
    angle = float(np.linalg.norm(aa_blender))
    if angle < 1e-8:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    axis = (aa_blender / angle).astype(float)
    return Quaternion(Vector(axis), angle)


if __name__ == "__main__":
    sys.exit(main())
