"""Format de l'animation intermédiaire animation.npz.

Toutes les rotations sont en axis-angle (Rodrigues), float32, radians.
Le format complet est documenté dans docs/PIPELINE.md §3.

Usage :
    from pipeline.animation_npz import Animation
    anim = Animation.load("animation.npz")
    anim.validate()
    anim.save("animation.npz")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Constantes de référence (cf. docs/PIPELINE.md §3 et §4.1).
NUM_BODY_JOINTS = 21
NUM_HAND_JOINTS = 15
NUM_BETAS_DEFAULT = 10
NUM_EXPRESSION_COEFFS_DEFAULT = 50


@dataclass
class Animation:
    """Animation SMPL-X consolidée, prête pour le retargeting VRM.

    Tous les arrays time-series ont T frames en première dimension.

    Attributs :
        fps                  : cadence après ré-échantillonnage (Hz)
        transl               : (T, 3) translation racine, mètres
        global_orient        : (T, 3) rotation racine, axis-angle
        body_pose            : (T, 21, 3) rotations articulaires corps
        left_hand_pose       : (T, 15, 3) rotations doigts main gauche
        right_hand_pose      : (T, 15, 3) rotations doigts main droite
        jaw_pose             : (T, 3) rotation mâchoire FLAME
        leye_pose            : (T, 3) rotation œil gauche FLAME
        reye_pose            : (T, 3) rotation œil droit FLAME
        expression           : (T, num_expression_coeffs) coefficients PCA FLAME
        betas                : (num_betas,) forme corps, constante sur la séquence
        confidence_body      : (T,) confiance corps ∈ [0, 1]
        confidence_lhand     : (T,) confiance main gauche ∈ [0, 1]
        confidence_rhand     : (T,) confiance main droite ∈ [0, 1]
        confidence_face      : (T,) confiance visage ∈ [0, 1]
        frame_indices        : (T,) indice de frame dans la vidéo originale
        source_video         : nom de la vidéo source (basename)
        source_fps           : fps de la vidéo source
        meta_json            : JSON sérialisé (versions des modèles, params, dates)
    """

    fps: float
    transl: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    left_hand_pose: np.ndarray
    right_hand_pose: np.ndarray
    jaw_pose: np.ndarray
    leye_pose: np.ndarray
    reye_pose: np.ndarray
    expression: np.ndarray
    betas: np.ndarray
    confidence_body: np.ndarray
    confidence_lhand: np.ndarray
    confidence_rhand: np.ndarray
    confidence_face: np.ndarray
    frame_indices: np.ndarray
    source_video: str
    source_fps: float
    meta_json: str = field(default="{}")

    @property
    def num_frames(self) -> int:
        return int(self.transl.shape[0])

    @property
    def num_expression_coeffs(self) -> int:
        return int(self.expression.shape[1])

    @property
    def num_betas(self) -> int:
        return int(self.betas.shape[0])

    def save(self, path: str | Path) -> None:
        """Écrit le NPZ. Lance validate() avant écriture."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            fps=np.float32(self.fps),
            transl=self.transl.astype(np.float32),
            global_orient=self.global_orient.astype(np.float32),
            body_pose=self.body_pose.astype(np.float32),
            left_hand_pose=self.left_hand_pose.astype(np.float32),
            right_hand_pose=self.right_hand_pose.astype(np.float32),
            jaw_pose=self.jaw_pose.astype(np.float32),
            leye_pose=self.leye_pose.astype(np.float32),
            reye_pose=self.reye_pose.astype(np.float32),
            expression=self.expression.astype(np.float32),
            betas=self.betas.astype(np.float32),
            confidence_body=self.confidence_body.astype(np.float32),
            confidence_lhand=self.confidence_lhand.astype(np.float32),
            confidence_rhand=self.confidence_rhand.astype(np.float32),
            confidence_face=self.confidence_face.astype(np.float32),
            frame_indices=self.frame_indices.astype(np.int32),
            source_video=np.array(self.source_video),
            source_fps=np.float32(self.source_fps),
            meta_json=np.array(self.meta_json),
        )
        logger.info("Animation sauvegardée : %s (%d frames @ %.1f fps)",
                    path, self.num_frames, self.fps)

    @classmethod
    def load(cls, path: str | Path) -> "Animation":
        """Charge un NPZ et lance validate()."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Animation NPZ introuvable : {path}")
        with np.load(path, allow_pickle=False) as data:
            anim = cls(
                fps=float(data["fps"]),
                transl=data["transl"],
                global_orient=data["global_orient"],
                body_pose=data["body_pose"],
                left_hand_pose=data["left_hand_pose"],
                right_hand_pose=data["right_hand_pose"],
                jaw_pose=data["jaw_pose"],
                leye_pose=data["leye_pose"],
                reye_pose=data["reye_pose"],
                expression=data["expression"],
                betas=data["betas"],
                confidence_body=data["confidence_body"],
                confidence_lhand=data["confidence_lhand"],
                confidence_rhand=data["confidence_rhand"],
                confidence_face=data["confidence_face"],
                frame_indices=data["frame_indices"],
                source_video=str(data["source_video"]),
                source_fps=float(data["source_fps"]),
                meta_json=str(data["meta_json"]),
            )
        anim.validate()
        return anim

    def validate(self) -> None:
        """Asserte que toutes les shapes / dtypes / contraintes sont respectées.

        Lève AssertionError au premier problème.
        """
        T = self.num_frames
        assert T > 0, f"Animation vide (T={T})"
        assert self.fps > 0, f"fps invalide : {self.fps}"
        assert self.source_fps > 0, f"source_fps invalide : {self.source_fps}"

        _check_shape(self.transl, (T, 3), "transl")
        _check_shape(self.global_orient, (T, 3), "global_orient")
        _check_shape(self.body_pose, (T, NUM_BODY_JOINTS, 3), "body_pose")
        _check_shape(self.left_hand_pose, (T, NUM_HAND_JOINTS, 3), "left_hand_pose")
        _check_shape(self.right_hand_pose, (T, NUM_HAND_JOINTS, 3), "right_hand_pose")
        _check_shape(self.jaw_pose, (T, 3), "jaw_pose")
        _check_shape(self.leye_pose, (T, 3), "leye_pose")
        _check_shape(self.reye_pose, (T, 3), "reye_pose")

        # expression et betas ont des dim variables, on vérifie juste le rang.
        assert self.expression.ndim == 2 and self.expression.shape[0] == T, (
            f"expression doit être (T, n_exp), reçu {self.expression.shape}"
        )
        assert self.betas.ndim == 1, f"betas doit être (n_betas,), reçu {self.betas.shape}"

        for name in ("confidence_body", "confidence_lhand",
                     "confidence_rhand", "confidence_face"):
            arr = getattr(self, name)
            _check_shape(arr, (T,), name)
            assert (arr >= 0).all() and (arr <= 1).all(), (
                f"{name} doit être dans [0, 1], min={arr.min()}, max={arr.max()}"
            )

        _check_shape(self.frame_indices, (T,), "frame_indices")
        assert self.frame_indices.dtype.kind == "i", (
            f"frame_indices doit être entier, dtype={self.frame_indices.dtype}"
        )

        # Pas de NaN/Inf dans les rotations / translations.
        for name in ("transl", "global_orient", "body_pose",
                     "left_hand_pose", "right_hand_pose",
                     "jaw_pose", "leye_pose", "reye_pose",
                     "expression", "betas"):
            arr = getattr(self, name)
            assert np.isfinite(arr).all(), f"{name} contient des NaN/Inf"

        # meta_json doit être du JSON valide (ne lève pas si vide).
        try:
            json.loads(self.meta_json)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"meta_json n'est pas du JSON valide : {exc}") from exc

    def with_meta(self, **extra: Any) -> "Animation":
        """Retourne une copie avec meta_json enrichi (les clés dupliquées sont écrasées)."""
        meta = json.loads(self.meta_json) if self.meta_json else {}
        meta.update(extra)
        new_anim = Animation(**{f.name: getattr(self, f.name) for f in fields(self)})
        new_anim.meta_json = json.dumps(meta, default=str)
        return new_anim


def _check_shape(arr: np.ndarray, expected: tuple, name: str) -> None:
    assert isinstance(arr, np.ndarray), f"{name} doit être numpy.ndarray, reçu {type(arr)}"
    assert arr.shape == expected, (
        f"{name} shape={arr.shape}, attendu={expected}"
    )


def make_empty(
    num_frames: int,
    fps: float,
    source_video: str,
    source_fps: float,
    num_expression_coeffs: int = NUM_EXPRESSION_COEFFS_DEFAULT,
    num_betas: int = NUM_BETAS_DEFAULT,
) -> Animation:
    """Construit une Animation entièrement nulle aux bonnes shapes.

    Utilisé pour les tests et comme base à remplir incrémentalement par les
    différentes étapes du pipeline.
    """
    T = num_frames
    return Animation(
        fps=fps,
        transl=np.zeros((T, 3), np.float32),
        global_orient=np.zeros((T, 3), np.float32),
        body_pose=np.zeros((T, NUM_BODY_JOINTS, 3), np.float32),
        left_hand_pose=np.zeros((T, NUM_HAND_JOINTS, 3), np.float32),
        right_hand_pose=np.zeros((T, NUM_HAND_JOINTS, 3), np.float32),
        jaw_pose=np.zeros((T, 3), np.float32),
        leye_pose=np.zeros((T, 3), np.float32),
        reye_pose=np.zeros((T, 3), np.float32),
        expression=np.zeros((T, num_expression_coeffs), np.float32),
        betas=np.zeros((num_betas,), np.float32),
        confidence_body=np.zeros((T,), np.float32),
        confidence_lhand=np.zeros((T,), np.float32),
        confidence_rhand=np.zeros((T,), np.float32),
        confidence_face=np.zeros((T,), np.float32),
        frame_indices=np.arange(T, dtype=np.int32),
        source_video=source_video,
        source_fps=source_fps,
        meta_json="{}",
    )
