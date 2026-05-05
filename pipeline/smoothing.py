"""Lissage temporel d'une Animation SMPL-X.

Lisse :
    - transl, expression, betas        → filtrage scalaire direct (espace euclidien)
    - rotations axis-angle             → conversion en quaternion + filtrage + reconversion
    - confidence_*                     → laissé tel quel (déjà ∈ [0, 1])

Cf. docs/PIPELINE.md §8.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from pipeline.animation_npz import Animation
from pipeline.one_euro_filter import smooth_signal

logger = logging.getLogger(__name__)


@dataclass
class SmoothingParams:
    """Paramètres One-Euro par groupe d'articulations.

    Defaults issus de la littérature mocap. Surchargeables via config.yaml.
    """
    fps: float = 30.0

    transl_min_cutoff: float = 0.5
    transl_beta: float = 0.05

    body_min_cutoff: float = 1.0
    body_beta: float = 0.1

    hands_min_cutoff: float = 1.5
    hands_beta: float = 0.2

    face_min_cutoff: float = 1.5
    face_beta: float = 0.1

    @classmethod
    def from_config(cls, fps: float, smoothing_cfg: dict) -> "SmoothingParams":
        return cls(
            fps=fps,
            transl_min_cutoff=smoothing_cfg.get("transl", {}).get("min_cutoff", 0.5),
            transl_beta=smoothing_cfg.get("transl", {}).get("beta", 0.05),
            body_min_cutoff=smoothing_cfg.get("body", {}).get("min_cutoff", 1.0),
            body_beta=smoothing_cfg.get("body", {}).get("beta", 0.1),
            hands_min_cutoff=smoothing_cfg.get("hands", {}).get("min_cutoff", 1.5),
            hands_beta=smoothing_cfg.get("hands", {}).get("beta", 0.2),
            face_min_cutoff=smoothing_cfg.get("face", {}).get("min_cutoff", 1.5),
            face_beta=smoothing_cfg.get("face", {}).get("beta", 0.1),
        )


def smooth_animation(anim: Animation, params: SmoothingParams) -> Animation:
    """Retourne une nouvelle Animation lissée selon `params`.

    L'animation d'entrée n'est pas modifiée.

    Args:
        anim   : animation à lisser
        params : paramètres One-Euro par groupe

    Returns:
        Animation avec mêmes shapes / dtypes mais valeurs lissées.
    """
    if anim.num_frames < 2:
        logger.info("Animation trop courte (T=%d) — pas de lissage", anim.num_frames)
        return anim

    T = anim.num_frames
    fps = params.fps
    logger.info("Lissage One-Euro de %d frames @ %.1f fps", T, fps)

    # Translation et expression : espace euclidien, filtrage scalaire direct.
    transl_smooth = smooth_signal(
        anim.transl, fps, params.transl_min_cutoff, params.transl_beta,
    )
    expression_smooth = smooth_signal(
        anim.expression, fps, params.face_min_cutoff, params.face_beta,
    )

    # Rotations : on passe par les quaternions (cf. §8.2).
    global_orient_smooth = _smooth_axis_angle(
        anim.global_orient.reshape(T, 1, 3), fps,
        params.body_min_cutoff, params.body_beta,
    ).reshape(T, 3)

    body_pose_smooth = _smooth_axis_angle(
        anim.body_pose, fps, params.body_min_cutoff, params.body_beta,
    )

    left_hand_smooth = _smooth_axis_angle(
        anim.left_hand_pose, fps, params.hands_min_cutoff, params.hands_beta,
    )
    right_hand_smooth = _smooth_axis_angle(
        anim.right_hand_pose, fps, params.hands_min_cutoff, params.hands_beta,
    )

    jaw_smooth = _smooth_axis_angle(
        anim.jaw_pose.reshape(T, 1, 3), fps,
        params.face_min_cutoff, params.face_beta,
    ).reshape(T, 3)
    leye_smooth = _smooth_axis_angle(
        anim.leye_pose.reshape(T, 1, 3), fps,
        params.face_min_cutoff, params.face_beta,
    ).reshape(T, 3)
    reye_smooth = _smooth_axis_angle(
        anim.reye_pose.reshape(T, 1, 3), fps,
        params.face_min_cutoff, params.face_beta,
    ).reshape(T, 3)

    return Animation(
        fps=anim.fps,
        transl=transl_smooth,
        global_orient=global_orient_smooth,
        body_pose=body_pose_smooth,
        left_hand_pose=left_hand_smooth,
        right_hand_pose=right_hand_smooth,
        jaw_pose=jaw_smooth,
        leye_pose=leye_smooth,
        reye_pose=reye_smooth,
        expression=expression_smooth,
        betas=anim.betas,
        confidence_body=anim.confidence_body,
        confidence_lhand=anim.confidence_lhand,
        confidence_rhand=anim.confidence_rhand,
        confidence_face=anim.confidence_face,
        frame_indices=anim.frame_indices,
        source_video=anim.source_video,
        source_fps=anim.source_fps,
        meta_json=anim.meta_json,
    )


def _smooth_axis_angle(
    rotations: np.ndarray,
    fps: float,
    min_cutoff: float,
    beta: float,
) -> np.ndarray:
    """Lisse un tableau de rotations axis-angle de shape (T, J, 3) via quaternions.

    Étapes :
        1. axis-angle → quaternion par frame
        2. hemisphere consistency : pour chaque j, négativer q_t si dot(q_t, q_{t-1}) < 0
        3. filtrer One-Euro composante par composante (4 filtres)
        4. re-normaliser à norme 1
        5. quaternion → axis-angle
    """
    T, J, three = rotations.shape
    assert three == 3

    # 1. axis-angle → quaternion (T, J, 4) ordre (x, y, z, w)
    quats = _axis_angle_to_quat(rotations.reshape(-1, 3)).reshape(T, J, 4)

    # 2. hemisphere consistency frame par frame
    for t in range(1, T):
        d = (quats[t] * quats[t - 1]).sum(axis=-1)  # (J,)
        flip = d < 0
        quats[t, flip] *= -1.0

    # 3. filtrer chaque composante de chaque joint indépendamment
    quats_smooth = smooth_signal(quats, fps, min_cutoff, beta)

    # 4. re-normaliser
    norms = np.linalg.norm(quats_smooth, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    quats_smooth = quats_smooth / norms

    # 5. quaternion → axis-angle
    return _quat_to_axis_angle(quats_smooth.reshape(-1, 4)).reshape(T, J, 3).astype(np.float32)


def _axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle (rad) → (N, 4) quaternion (x, y, z, w)."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)  # (N, 1)
    safe_angle = np.maximum(angle, 1e-8)
    axis = aa / safe_angle
    half = angle / 2.0
    sin_half = np.sin(half)
    cos_half = np.cos(half)
    q = np.concatenate([axis * sin_half, cos_half], axis=-1)
    # Si angle = 0 : retourner (0, 0, 0, 1)
    zero = (angle.squeeze(-1) < 1e-8)
    q[zero] = np.array([0.0, 0.0, 0.0, 1.0])
    return q


def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """(N, 4) quaternion (x, y, z, w) → (N, 3) axis-angle (rad)."""
    # Force w >= 0 pour minimiser l'amplitude de rotation représentée
    q = q.copy()
    flip = q[..., 3] < 0
    q[flip] *= -1.0

    w = np.clip(q[..., 3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))

    axis = np.zeros((q.shape[0], 3), dtype=q.dtype)
    nonzero = sin_half > 1e-8
    axis[nonzero] = q[nonzero, :3] / sin_half[nonzero, None]

    return axis * angle[..., None]
