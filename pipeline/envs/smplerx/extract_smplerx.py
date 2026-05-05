"""Extraction SMPLer-X — étape 1 du pipeline.

Lit une vidéo, détecte la personne avec mmdet, régresse les paramètres SMPL-X
avec SMPLer-X, et écrit un NPZ partiel (corps + mains + visage estimés
par SMPLer-X) au format animation.npz.

L'étape suivante (HaMeR) raffine les mains. EMOCA raffine le visage.

Cet env utilise Python 3.8 + PyTorch 1.12 + cu113. Cf. docs/PIPELINE.md §1.1.

Usage standalone :
    python extract_smplerx.py --video X.mp4 --output smplerx.npz [--config '{}']
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ajoute la racine du repo au PYTHONPATH pour importer pipeline.*
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.animation_npz import (  # noqa: E402
    NUM_BODY_JOINTS, NUM_HAND_JOINTS, Animation, make_empty,
)
from pipeline.confidence import combine_confidence  # noqa: E402
from pipeline.video_io import probe_video, resample_indices  # noqa: E402

logger = logging.getLogger("smplerx")


def run_smplerx(
    video_path: Path,
    output_npz: Path,
    config: dict,
) -> None:
    """Lance SMPLer-X sur une vidéo et écrit l'animation partielle.

    Args:
        video_path : chemin vers la vidéo source
        output_npz : chemin vers le NPZ à écrire
        config     : dict pipeline.* du config.yaml (output_fps, smplerx_model,
                     person_selection, num_betas, num_expression_coeffs)
    """
    logger.info("SMPLer-X : %s → %s", video_path, output_npz)

    info = probe_video(video_path)
    fps_out = float(config.get("output_fps", 30))
    indices = resample_indices(info.num_frames, info.fps, fps_out)
    T = len(indices)
    logger.info("  Source : %.2f fps, %d frames → cible %.1f fps, %d frames",
                info.fps, info.num_frames, fps_out, T)

    if T == 0:
        raise RuntimeError("Vidéo vide après ré-échantillonnage")

    # ── Imports ML lazy (échouent si SMPLer-X non installé — c'est attendu) ──
    smplerx_model = _load_smplerx(config)
    detector = _load_detector(config)

    num_expr = int(config.get("num_expression_coeffs", 50))
    num_betas = int(config.get("num_betas", 10))

    anim = make_empty(
        num_frames=T,
        fps=fps_out,
        source_video=video_path.name,
        source_fps=info.fps,
        num_expression_coeffs=num_expr,
        num_betas=num_betas,
    )
    bbox_scores = np.zeros(T, dtype=np.float32)
    reproj_residuals = np.zeros(T, dtype=np.float32)
    betas_accum: list[np.ndarray] = []

    # ── Boucle principale : itère les frames cibles ──────────────────────
    from tqdm import tqdm
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        for t, src_idx in enumerate(tqdm(indices, desc="SMPLer-X", unit="frame")):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_idx))
            ok, frame = cap.read()
            if not ok:
                logger.warning("Impossible de lire la frame %d — frame nulle", src_idx)
                continue

            # 1. Détection bbox personne
            bbox, det_score = _detect_person(detector, frame, config)
            if bbox is None:
                # Pas de personne détectée — on laisse les params à 0 et confidence=0
                bbox_scores[t] = 0.0
                continue
            bbox_scores[t] = det_score

            # 2. Régression SMPL-X
            params, residual = _infer_smplerx(smplerx_model, frame, bbox)
            reproj_residuals[t] = residual

            anim.transl[t]          = params["transl"]
            anim.global_orient[t]   = params["global_orient"]
            anim.body_pose[t]       = params["body_pose"].reshape(NUM_BODY_JOINTS, 3)
            anim.left_hand_pose[t]  = params["left_hand_pose"].reshape(NUM_HAND_JOINTS, 3)
            anim.right_hand_pose[t] = params["right_hand_pose"].reshape(NUM_HAND_JOINTS, 3)
            anim.jaw_pose[t]        = params["jaw_pose"]
            anim.leye_pose[t]       = params["leye_pose"]
            anim.reye_pose[t]       = params["reye_pose"]
            # SMPLer-X sort une dim 10 d'expression — on padde si num_expr > 10
            expr = params["expression"]
            if expr.shape[0] >= num_expr:
                anim.expression[t] = expr[:num_expr]
            else:
                anim.expression[t, :expr.shape[0]] = expr
            betas_accum.append(params["betas"][:num_betas])

            anim.frame_indices[t] = int(src_idx)

    finally:
        cap.release()

    # Forme corps : médiane sur la séquence (plus robuste à un détrompage ponctuel).
    if betas_accum:
        anim.betas = np.median(np.stack(betas_accum, axis=0), axis=0).astype(np.float32)

    # Confidence par région — pour SMPLer-X tout vient des mêmes bbox/résidu
    body_conf = combine_confidence(bbox_scores, reproj_residuals)
    anim.confidence_body  = body_conf
    anim.confidence_lhand = body_conf  # affiné par HaMeR à l'étape 2
    anim.confidence_rhand = body_conf
    anim.confidence_face  = body_conf  # affiné par EMOCA à l'étape 3

    anim = anim.with_meta(
        stage="smplerx",
        smplerx_model=config.get("smplerx_model", "h32_correct"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    anim.save(output_npz)


# ──────────────────────────────────────────────────────────────────────────────
# Chargement modèles (à implémenter avec l'API réelle de SMPLer-X)
# ──────────────────────────────────────────────────────────────────────────────

def _load_smplerx(config: dict):
    """Charge SMPLer-X via le wrapper smplerx_wrapper.load_model().

    Retourne une instance Demoer (le helper du repo SMPLer-X qui expose .model).
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible — SMPLer-X requiert un GPU NVIDIA")

    model_name = config.get("smplerx_model", "h32_correct")
    weights_path = REPO_ROOT / "pipeline" / "models" / "smplerx" / f"smpler_x_{model_name}.pth.tar"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Poids SMPLer-X manquants : {weights_path}. "
            "Lancer scripts/download_weights.sh"
        )

    from smplerx_wrapper import load_model  # type: ignore[import-not-found]
    return load_model(weights_path, model_name)


def _load_detector(config: dict):
    """Charge le détecteur mmdet Faster R-CNN R50."""
    weights_path = REPO_ROOT / "pipeline" / "models" / "mmdet" / "faster_rcnn_r50_fpn_1x_coco.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids mmdet manquants : {weights_path}")

    from mmdet.apis import init_detector  # type: ignore[import-not-found]
    config_file = REPO_ROOT / "pipeline" / "envs" / "smplerx" / "mmdet_config.py"
    return init_detector(str(config_file), str(weights_path), device="cuda:0")


def _detect_person(
    detector,
    frame_bgr: np.ndarray,
    config: dict,
) -> tuple[np.ndarray | None, float]:
    """Détecte la personne dans une frame.

    Reproduit la convention de SMPLer-X : utilise process_mmdet_results de
    common/utils/inference_utils.py qui filtre la classe 0 (person) et
    retourne une liste [bbox_list, score_list].

    Args:
        detector  : modèle mmdet (init_detector)
        frame_bgr : (H, W, 3) uint8 BGR
        config    : dict pipeline.* — utilise person_selection

    Returns:
        (bbox xyxy en pixels, score) ou (None, 0.0) si aucune détection valide.
    """
    from mmdet.apis import inference_detector  # type: ignore[import-not-found]
    sys.path.insert(0, str(REPO_ROOT / "pipeline" / "envs" / "smplerx" / "repo" / "common"))
    from utils.inference_utils import process_mmdet_results  # type: ignore[import-not-found]

    result = inference_detector(detector, frame_bgr)
    # process_mmdet_results filtre cat_id=0 (person dans COCO) et retourne
    # une liste de bboxes [x1, y1, x2, y2, score] en multi-person mode.
    person_results = process_mmdet_results(result, cat_id=0, multi_person=True)
    if not person_results or len(person_results) == 0:
        return None, 0.0

    dets = np.array(person_results, dtype=np.float32)
    if dets.ndim == 1:
        dets = dets[None, :]
    if dets.shape[0] == 0:
        return None, 0.0

    selection = config.get("person_selection", "largest")
    if selection == "largest":
        sizes = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        idx = int(np.argmax(sizes))
    elif selection == "center":
        H, W = frame_bgr.shape[:2]
        cx = (dets[:, 0] + dets[:, 2]) / 2
        cy = (dets[:, 1] + dets[:, 3]) / 2
        d = (cx - W / 2) ** 2 + (cy - H / 2) ** 2
        idx = int(np.argmin(d))
    else:
        idx = 0

    return dets[idx, :4], float(dets[idx, 4])


def _infer_smplerx(
    demoer,
    frame_bgr: np.ndarray,
    bbox: np.ndarray,
) -> tuple[dict[str, np.ndarray], float]:
    """Régression SMPL-X via le wrapper smplerx_wrapper.infer().

    Args:
        demoer    : Demoer chargé par _load_smplerx()
        frame_bgr : (H, W, 3) uint8 BGR
        bbox      : (4,) xyxy en pixels

    Returns:
        (params_dict, reprojection_residual_pixels)
    """
    from smplerx_wrapper import infer  # type: ignore[import-not-found]
    return infer(demoer, frame_bgr, bbox)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extraction SMPLer-X")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=str, default="{}",
                        help="JSON sérialisé de la section pipeline.* du config")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    config = json.loads(args.config)
    run_smplerx(args.video, args.output, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
