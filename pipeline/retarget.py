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

    # ── VRM 0.x (merge avec VRM 1.0) ─────────────────────────────────────
    # Beaucoup de VRM "1.0" exportés depuis VRoid contiennent EN PLUS les bones
    # en format 0.x (les "Duplicated VRM0 bone" warnings dans l'addon). Le mapping
    # VRM 1.0 peut être quasi-vide, mais le 0.x est complet. On merge les deux,
    # en privilégiant ce qui a déjà été trouvé en 1.0.
    vrm0 = getattr(ext, "vrm0", None)
    if vrm0 is not None and getattr(vrm0, "humanoid", None) is not None:
        n_before = len(humanoid_bones)
        for hb in vrm0.humanoid.human_bones:
            vrm_bone_name = getattr(hb, "bone", None)
            node_bone = getattr(getattr(hb, "node", None), "bone_name", None)
            if vrm_bone_name and node_bone and vrm_bone_name not in humanoid_bones:
                humanoid_bones[vrm_bone_name] = node_bone
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
    """Crée les keyframes de rotation pour chaque bone VRM via look-at.

    Approche :
    1. Forward kinematic SMPL-X en pur numpy → positions monde des 22 joints
       body à chaque frame.
    2. Pour chaque bone VRM, look-at vers la position du joint enfant SMPL-X.
       Indépendant des conventions axis-angle SMPL-X/Blender (qui sont la cause
       du retargeting tordu de l'approche précédente).

    Limitations actuelles :
    - Les rotations des doigts ne sont pas retargetées (rest pose conservée).
      Faire ça nécessiterait la FK des 15 phalanges par main — TODO.
    - Le twist (rotation autour de l'axe long du bone) est perdu. Acceptable
      pour la LSF (corps + bras), à corriger pour le visage si nécessaire.
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

    # Bones ordonnés topologiquement (parents avant enfants) pour que les
    # transformations en cascade soient correctes.
    ordered_bones: list[str] = [
        "hips", "spine", "chest", "upperChest", "neck",
        "leftShoulder", "leftUpperArm", "leftLowerArm",
        "rightShoulder", "rightUpperArm", "rightLowerArm",
        "leftUpperLeg", "leftLowerLeg", "leftFoot",
        "rightUpperLeg", "rightLowerLeg", "rightFoot",
    ]

    from mathutils import Matrix

    for t in range(anim.num_frames):
        bpy.context.scene.frame_set(t + 1)

        # FK SMPL-X pour cette frame (en frame SMPL-X : Y up, X right)
        joints_smplx = _smplx_fk_body(
            rest_joints, parents,
            anim.global_orient[t], anim.body_pose[t],
        )

        # SMPL-X (X right, Y up, Z forward) → Blender (X right, Y forward, Z up).
        # Empiriquement (testé visuellement) : (x, y, z) → (x, -z, y).
        joints_blender = np.column_stack([
            joints_smplx[:, 0],
            -joints_smplx[:, 2],
            joints_smplx[:, 1],
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

            # Rotation locale qui aligne axe Y du bone (= direction) sur target
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

# Indices canoniques des joints SMPL-X body (0..21).
# Source : smplx.JOINT_NAMES.
_SMPLX_JOINT_IDX = {
    "pelvis": 0,
    "left_hip": 1, "right_hip": 2, "spine1": 3,
    "left_knee": 4, "right_knee": 5, "spine2": 6,
    "left_ankle": 7, "right_ankle": 8, "spine3": 9,
    "left_foot": 10, "right_foot": 11, "neck": 12,
    "left_collar": 13, "right_collar": 14, "head": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
}

# Mapping VRM bone → (start_joint_idx, end_joint_idx) en indices SMPL-X.
# Le bone VRM est orienté du premier vers le second joint.
_VRM_BONE_SEGMENTS: dict[str, tuple[int, int]] = {
    "hips":          (0, 3),    # pelvis → spine1 (oriente le buste)
    "spine":         (3, 6),    # spine1 → spine2
    "chest":         (6, 9),    # spine2 → spine3
    "upperChest":    (9, 12),   # spine3 → neck
    "neck":          (12, 15),  # neck → head
    "leftShoulder":  (9, 16),   # spine3 → left_shoulder
    "leftUpperArm":  (16, 18),  # left_shoulder → left_elbow
    "leftLowerArm":  (18, 20),  # left_elbow → left_wrist
    "rightShoulder": (9, 17),
    "rightUpperArm": (17, 19),
    "rightLowerArm": (19, 21),
    "leftUpperLeg":  (1, 4),    # left_hip → left_knee
    "leftLowerLeg":  (4, 7),    # left_knee → left_ankle
    "leftFoot":      (7, 10),   # left_ankle → left_foot
    "rightUpperLeg": (2, 5),
    "rightLowerLeg": (5, 8),
    "rightFoot":     (8, 11),
}


def _load_smplx_rest_skeleton():
    """Charge rest joints SMPL-X + table des parents depuis SMPLX_NEUTRAL.npz.

    Returns:
        rest_joints (22, 3) : positions des 22 body joints en pose canonique (Y-up)
        parents (22,)       : indice du parent pour chaque joint (-1 pour racine)
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
    # Le regressor peut être en sparse selon le format
    if hasattr(J_regressor, "toarray"):
        J_regressor = J_regressor.toarray()
    rest_joints = J_regressor @ v_template
    kintree = data["kintree_table"]
    parents = kintree[0].astype(np.int64)
    return rest_joints[:22].astype(np.float64), parents[:22]


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


def _smplx_fk_body(rest_joints, parents, global_orient, body_pose):
    """Forward kinematic SMPL-X (22 body joints).

    Args:
        rest_joints (22, 3) : positions rest des 22 body joints
        parents (22,)        : indices parents
        global_orient (3,)   : axis-angle racine
        body_pose (21, 3)    : axis-angles des 21 joints non-root

    Returns:
        joints_world (22, 3) : positions monde après application des rotations
    """
    import numpy as np

    # Rotations locales de chaque joint (22)
    rotations = np.zeros((22, 3, 3), dtype=np.float64)
    rotations[0] = _aa_to_rot_mat(global_orient)
    for i in range(21):
        rotations[i + 1] = _aa_to_rot_mat(body_pose[i])

    # FK : pour chaque joint, position = parent_world_pos + parent_world_rot @ rel
    joints_world = np.zeros((22, 3), dtype=np.float64)
    rot_world = np.zeros((22, 3, 3), dtype=np.float64)
    joints_world[0] = rest_joints[0]
    rot_world[0] = rotations[0]
    for i in range(1, 22):
        p = int(parents[i])
        rel = rest_joints[i] - rest_joints[p]
        joints_world[i] = joints_world[p] + rot_world[p] @ rel
        rot_world[i] = rot_world[p] @ rotations[i]
    return joints_world


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
