"""Extraction EMOCA v2 — étape 3 du pipeline.

Lit la vidéo et le NPZ produit par HaMeR, raffine les paramètres FLAME du
visage avec EMOCA v2, et écrit un nouveau NPZ avec :
    - corps + mains inchangés
    - jaw_pose remplacé par jaw FLAME (axis-angle, derniers 3 de posecode)
    - expression remplacée par expcode FLAME (50 coefficients PCA)
    - confidence_face mise à jour selon le score FAN

API vérifiée le 2026-05-05 sur le commit `e0be0dbc2d32629ae384ae10c0b7974948c994fd`
de https://github.com/radekd91/emoca par lecture directe du code source :
    - gdl_apps/EMOCA/utils/load.py (load_model)
    - gdl_apps/EMOCA/utils/io.py    (test, save_codes — clés du dict vals)
    - gdl/datasets/ImageTestDataset.py (preprocessing : crop FAN, /255, 224×224)
    - gdl/utils/FaceDetector.py (FAN wrapper)

Cf. docs/PIPELINE.md §7.

Usage standalone :
    python extract_emoca.py --video X.mp4 --input hamer.npz --output emoca.npz
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.animation_npz import Animation  # noqa: E402
from pipeline.confidence import combine_confidence  # noqa: E402

logger = logging.getLogger("emoca")


def run_emoca(
    video_path: Path,
    input_npz: Path,
    output_npz: Path,
    config: dict,
) -> None:
    """Lance EMOCA et raffine le visage dans l'animation."""
    logger.info("EMOCA : %s (raffinement visage)", input_npz)

    anim = Animation.load(input_npz)
    num_expr = int(config.get("num_expression_coeffs", 50))

    if anim.expression.shape[1] != num_expr:
        logger.warning(
            "Animation a %d coefficients d'expression, config demande %d — redimensionnement",
            anim.expression.shape[1], num_expr,
        )

    emoca = _load_emoca()
    fan = _load_face_detector()

    import cv2
    import torch
    from tqdm import tqdm

    face_scores = np.zeros(anim.num_frames, dtype=np.float32)
    new_expressions = np.zeros((anim.num_frames, num_expr), dtype=np.float32)

    cap = cv2.VideoCapture(str(video_path))
    try:
        for t in tqdm(range(anim.num_frames), desc="EMOCA", unit="frame"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(anim.frame_indices[t]))
            ok, frame_bgr = cap.read()
            if not ok:
                continue

            face_bbox, face_score = _detect_face(fan, frame_bgr)
            if face_bbox is None:
                face_scores[t] = 0.0
                continue
            face_scores[t] = face_score

            # Pré-process : crop similarity transform à 224×224 + /255 + (3, H, W)
            img_tensor = _preprocess_face(frame_bgr, face_bbox)

            # Inference (cf. gdl_apps/EMOCA/utils/io.py::test)
            with torch.no_grad():
                vals = emoca.encode({"image": img_tensor.cuda()}, training=False)

            posecode = vals["posecode"].detach().cpu().numpy().squeeze(0)  # (6,)
            expcode = vals["expcode"].detach().cpu().numpy().squeeze(0)    # (50,)

            # posecode : [0:3] = global rotation, [3:6] = jaw_pose
            anim.jaw_pose[t] = posecode[3:6].astype(np.float32)

            if expcode.shape[0] >= num_expr:
                new_expressions[t] = expcode[:num_expr]
            else:
                new_expressions[t, :expcode.shape[0]] = expcode
    finally:
        cap.release()

    anim.expression = new_expressions
    anim.confidence_face = combine_confidence(face_scores, None)

    anim = anim.with_meta(
        stage="emoca",
        timestamp_emoca=datetime.now(timezone.utc).isoformat(),
        emoca_variant="EMOCA_v2_lr_mse_20",
    )
    anim.save(output_npz)


# ──────────────────────────────────────────────────────────────────────────────
# Chargement modèles
# ──────────────────────────────────────────────────────────────────────────────

def _load_emoca():
    """Charge EMOCA v2 via load_model().

    Signature exacte (cf. gdl_apps/EMOCA/utils/load.py::load_model) :
        load_model(path_to_models, run_name, stage)
        - path_to_models : dossier contenant le variant (ex: assets/EMOCA/models)
        - run_name       : nom du variant (ex: 'EMOCA_v2_lr_mse_20')
        - stage          : 'detail' ou 'coarse'
    Retourne (deca: DecaModule, conf: OmegaConf).
    """
    repo = REPO_ROOT / "pipeline" / "envs" / "emoca" / "repo"
    if not repo.exists():
        raise FileNotFoundError(f"Repo EMOCA non cloné : {repo}")
    sys.path.insert(0, str(repo))

    from gdl_apps.EMOCA.utils.load import load_model  # type: ignore[import-not-found]

    path_to_models = REPO_ROOT / "pipeline" / "models" / "emoca" / "assets" / "EMOCA" / "models"
    model_name = "EMOCA_v2_lr_mse_20"
    if not (path_to_models / model_name).exists():
        raise FileNotFoundError(
            f"Variant EMOCA manquant : {path_to_models / model_name}. "
            "Lancer scripts/download_weights.sh"
        )

    emoca, _conf = load_model(str(path_to_models), model_name, "detail")
    emoca.cuda().eval()
    logger.info("EMOCA chargé : %s/detail", model_name)
    return emoca


def _load_face_detector():
    """Charge FAN (face_alignment) — détecteur visage par défaut d'EMOCA."""
    repo = REPO_ROOT / "pipeline" / "envs" / "emoca" / "repo"
    sys.path.insert(0, str(repo))
    from gdl.utils.FaceDetector import FAN  # type: ignore[import-not-found]
    return FAN()


def _detect_face(fan, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Détecte le visage le plus grand dans la frame.

    FAN.run(image_rgb) retourne (bboxes, bbox_type) où bbox = [left, top, right, bottom].

    Retourne (bbox xyxy en pixels, score) ou (None, 0).
    """
    rgb = frame_bgr[..., ::-1].copy()
    try:
        bboxes, _bbox_type = fan.run(rgb)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Échec détection FAN : %s", exc)
        return None, 0.0

    if bboxes is None or len(bboxes) == 0:
        return None, 0.0

    arr = np.asarray(bboxes, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]

    # Plus grosse bbox
    sizes = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
    idx = int(np.argmax(sizes))
    bbox = arr[idx, :4]

    # FAN ne retourne pas de score natif — on utilise la taille relative comme proxy
    H, W = frame_bgr.shape[:2]
    bbox_size = float(sizes[idx])
    img_size = float(H * W)
    score = float(np.clip(bbox_size / (img_size * 0.05), 0.1, 1.0))
    return bbox, score


def _preprocess_face(frame_bgr: np.ndarray, bbox: np.ndarray) -> "torch.Tensor":  # type: ignore[name-defined]
    """Pré-process exact qu'EMOCA attend en entrée.

    Reproduit ImageTestDataset.__getitem__ : crop similarity transform avec
    scale=1.25 autour de la bbox FAN, resize à 224×224, /255, transpose (3, H, W).

    Note : EMOCA n'applique PAS de normalisation imagenet — juste /255.
    """
    import torch
    from skimage.transform import estimate_transform, warp

    # bbox = [left, top, right, bottom]
    left, top, right, bottom = [float(v) for v in bbox]
    old_size = (right - left + bottom - top) / 2
    center = np.array([
        right - (right - left) / 2.0,
        bottom - (bottom - top) / 2.0,
    ])

    scale = 1.25
    size = int(old_size * scale)
    src_pts = np.array([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2],
    ])

    resolution = 224
    dst_pts = np.array([
        [0, 0], [0, resolution - 1], [resolution - 1, 0],
    ])
    tform = estimate_transform("similarity", src_pts, dst_pts)

    # BGR → RGB et normalisation /255
    rgb = frame_bgr[..., ::-1].astype(np.float32) / 255.0
    dst_image = warp(rgb, tform.inverse, output_shape=(resolution, resolution))
    dst_image = dst_image.transpose(2, 0, 1)  # (3, H, W)
    return torch.from_numpy(dst_image).float().unsqueeze(0)  # (1, 3, 224, 224)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extraction EMOCA (visage)")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=str, default="{}")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    config = json.loads(args.config)
    run_emoca(args.video, args.input, args.output, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
