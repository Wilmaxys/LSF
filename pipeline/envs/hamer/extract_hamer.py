"""Extraction HaMeR — étape 2 du pipeline.

Lit la vidéo et le NPZ produit par SMPLer-X, raffine les paramètres MANO
des deux mains avec HaMeR, et écrit un nouveau NPZ avec :
    - corps inchangé
    - left_hand_pose / right_hand_pose remplacés (axis-angle 15 doigts × 3)
    - confidence_lhand / confidence_rhand mis à jour

Pipeline réel HaMeR (cf. .research/hamer/demo.py) :
    1. ViTDet détecte les bodies (classe COCO 0)
    2. ViTPose extrait 133 keypoints whole-body par personne
    3. Les keypoints [-42:-21] et [-21:] donnent les mains gauche/droite
    4. Les bboxes mains sont calculées depuis les keypoints (conf > 0.5)
    5. ViTDetDataset prepare les crops (256×256, flip horizontal pour main gauche)
    6. HaMeR model produit pred_mano_params (rotmat) + cams
    7. Conversion rotmat → axis-angle ; flip Y/Z pour main gauche

Cf. docs/PIPELINE.md §7.

Usage standalone :
    python extract_hamer.py --video X.mp4 --input smplerx.npz --output hamer.npz
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

from pipeline.animation_npz import NUM_HAND_JOINTS, Animation  # noqa: E402
from pipeline.confidence import combine_confidence  # noqa: E402

logger = logging.getLogger("hamer")


def run_hamer(
    video_path: Path,
    input_npz: Path,
    output_npz: Path,
    config: dict,
) -> None:
    """Lance HaMeR et raffine les mains dans l'animation."""
    logger.info("HaMeR : %s (raffinement mains)", input_npz)

    anim = Animation.load(input_npz)
    detector = _load_body_detector()
    vitpose = _load_vitpose()
    hamer_model, hamer_cfg = _load_hamer()

    import cv2
    import torch
    from tqdm import tqdm

    # Imports HaMeR (depuis le repo cloné)
    repo = REPO_ROOT / "pipeline" / "envs" / "hamer" / "repo"
    sys.path.insert(0, str(repo))
    from hamer.datasets.vitdet_dataset import ViTDetDataset  # type: ignore[import-not-found]
    from hamer.utils import recursive_to  # type: ignore[import-not-found]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(str(video_path))
    try:
        for t in tqdm(range(anim.num_frames), desc="HaMeR", unit="frame"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(anim.frame_indices[t]))
            ok, frame_bgr = cap.read()
            if not ok:
                continue

            # 1. Détection bodies par ViTDet
            det_out = detector(frame_bgr)
            det_instances = det_out["instances"]
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores = det_instances.scores[valid_idx].cpu().numpy()
            if len(pred_bboxes) == 0:
                continue

            # 2. ViTPose : 133 keypoints whole-body par personne
            img_rgb = frame_bgr[:, :, ::-1]
            vitposes_out = vitpose.predict_pose(
                img_rgb,
                [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
            )

            # 3. Extraire bboxes mains depuis les keypoints
            hand_bboxes: list[list[float]] = []
            is_right_flags: list[int] = []
            hand_scores: list[float] = []
            for vp in vitposes_out:
                left_kp = vp["keypoints"][-42:-21]
                right_kp = vp["keypoints"][-21:]
                for kp, is_right in [(left_kp, 0), (right_kp, 1)]:
                    valid = kp[:, 2] > 0.5
                    if int(valid.sum()) <= 3:
                        continue
                    bbox = [
                        float(kp[valid, 0].min()), float(kp[valid, 1].min()),
                        float(kp[valid, 0].max()), float(kp[valid, 1].max()),
                    ]
                    hand_bboxes.append(bbox)
                    is_right_flags.append(is_right)
                    hand_scores.append(float(kp[valid, 2].mean()))

            if not hand_bboxes:
                continue

            boxes_np = np.array(hand_bboxes, dtype=np.float32)
            right_np = np.array(is_right_flags, dtype=np.int32)

            # 4. Préparer les crops via ViTDetDataset (gère le flip pour main gauche)
            dataset = ViTDetDataset(
                hamer_cfg, frame_bgr, boxes_np, right_np, rescale_factor=2.0,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=len(hand_bboxes), shuffle=False, num_workers=0,
            )

            # 5. Inférence
            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = hamer_model(batch)

                # Récupère MANO params (rotmat) et batch right flag
                hand_pose_rotmat = out["pred_mano_params"]["hand_pose"].cpu().numpy()
                # Shape : (B, 15, 3, 3)
                # global_orient et betas non utilisés ici (le wrist est géré par body_pose
                # dans SMPL-X via SMPLer-X ; on garde uniquement les 15 finger joints)

                right_flags = batch["right"].cpu().numpy()  # (B,)
                # mean keypoint conf comme score par main
                # (l'index dans hand_scores correspond au batch order si on a 1 batch)
                for i in range(hand_pose_rotmat.shape[0]):
                    rotmat = hand_pose_rotmat[i]  # (15, 3, 3)
                    is_right = bool(right_flags[i])
                    score = hand_scores[i] if i < len(hand_scores) else 0.5

                    # rotmat → axis-angle
                    pose_aa = np.stack([_rotmat_to_aa(R) for R in rotmat], axis=0)

                    # Pour main gauche, l'input a été flippé par ViTDetDataset.
                    # Le résultat est dans le frame "right hand model" ; pour le
                    # ramener au frame réel de la main gauche dans l'image source,
                    # on flippe Y et Z des axis-angles (cf. hamer/datasets/utils.py
                    # fliplr_params).
                    if not is_right:
                        pose_aa[:, 1] *= -1
                        pose_aa[:, 2] *= -1

                    fingers_15 = pose_aa.astype(np.float32)
                    if is_right:
                        anim.right_hand_pose[t] = fingers_15
                        anim.confidence_rhand[t] = score
                    else:
                        anim.left_hand_pose[t] = fingers_15
                        anim.confidence_lhand[t] = score
    finally:
        cap.release()

    # Réharmonise les confidences (clip [0, 1])
    anim.confidence_lhand = combine_confidence(anim.confidence_lhand, None)
    anim.confidence_rhand = combine_confidence(anim.confidence_rhand, None)

    anim = anim.with_meta(
        stage="hamer",
        timestamp_hamer=datetime.now(timezone.utc).isoformat(),
    )
    anim.save(output_npz)


# ──────────────────────────────────────────────────────────────────────────────
# Chargement modèles
# ──────────────────────────────────────────────────────────────────────────────

def _load_body_detector():
    """Charge ViTDet (cascade_mask_rcnn_vitdet_h_75ep) via DefaultPredictor_Lazy.

    Reproduit le code de demo.py pour le branch body_detector='vitdet'.
    """
    repo = REPO_ROOT / "pipeline" / "envs" / "hamer" / "repo"
    sys.path.insert(0, str(repo))
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy  # type: ignore[import-not-found]
    from detectron2.config import LazyConfig  # type: ignore[import-not-found]
    import hamer  # type: ignore[import-not-found]

    cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    detectron2_cfg = LazyConfig.load(str(cfg_path))

    weights = REPO_ROOT / "pipeline" / "models" / "hamer" / "model_final_f05665.pkl"
    if not weights.exists():
        raise FileNotFoundError(f"Poids ViTDet manquants : {weights}")
    detectron2_cfg.train.init_checkpoint = str(weights)

    # Lower thresholds comme dans demo.py
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25

    return DefaultPredictor_Lazy(detectron2_cfg)


def _load_vitpose():
    """Charge ViTPose (whole-body keypoints).

    Le repo HaMeR fournit `vitpose_model.py` à la racine qui wrappe la lib ViTPose.
    Le MODEL_DICT interne référence des paths relatifs (`third-party/ViTPose/...`)
    qui sont résolus au cwd. On chdir temporairement à la racine du repo HaMeR
    pour le init, puis on restaure.
    """
    import os
    repo = REPO_ROOT / "pipeline" / "envs" / "hamer" / "repo"
    sys.path.insert(0, str(repo))
    from vitpose_model import ViTPoseModel  # type: ignore[import-not-found]
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    old_cwd = os.getcwd()
    try:
        os.chdir(str(repo))
        return ViTPoseModel(device)
    finally:
        os.chdir(old_cwd)


def _load_hamer():
    """Charge le modèle HaMeR + sa config."""
    repo = REPO_ROOT / "pipeline" / "envs" / "hamer" / "repo"
    sys.path.insert(0, str(repo))
    from hamer.models import load_hamer  # type: ignore[import-not-found]
    import torch

    weights = REPO_ROOT / "pipeline" / "models" / "hamer" / "_DATA" / "data" / "checkpoints" / "hamer.ckpt"
    if not weights.exists():
        # Fallback : chemin alternatif si demo_data.tar.gz a été extrait différemment
        weights_alt = REPO_ROOT / "pipeline" / "models" / "hamer" / "checkpoint.ckpt"
        if weights_alt.exists():
            weights = weights_alt
        else:
            raise FileNotFoundError(
                f"Poids HaMeR manquants. Cherché : {weights} et {weights_alt}. "
                "Lancer scripts/download_weights.sh"
            )

    model, model_cfg = load_hamer(str(weights))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, model_cfg


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """Matrice 3×3 → axis-angle (3,)."""
    angle = np.arccos(np.clip((R.trace() - 1) / 2, -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2 * np.sin(angle))
    return (axis * angle).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extraction HaMeR (mains)")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=str, default="{}")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    config = json.loads(args.config)
    run_hamer(args.video, args.input, args.output, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
