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
    debug_overlay_dir: Path | None = None,
) -> None:
    """Lance HaMeR et raffine les mains dans l'animation.

    Si `debug_overlay_dir` est fourni, écrit une PNG par frame avec la mesh
    MANO superposée sur l'image source. Permet de vérifier visuellement ce
    que HaMeR détecte (sans interférence du retargeting downstream).
    """
    logger.info("HaMeR : %s (raffinement mains)", input_npz)
    if debug_overlay_dir is not None:
        debug_overlay_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Debug overlay HaMeR : écriture dans %s", debug_overlay_dir)

    # HaMeR + ViTPose utilisent des chemins relatifs `./_DATA/...` partout
    # (config, weights, MANO mean params, etc.). On chdir à la racine du repo
    # HaMeR pour toute la durée du run, puis on restaure.
    import os
    repo = REPO_ROOT / "pipeline" / "envs" / "hamer" / "repo"
    _orig_cwd = os.getcwd()
    os.chdir(str(repo))
    try:
        _run_hamer_impl(video_path, input_npz, output_npz, config, debug_overlay_dir)
    finally:
        os.chdir(_orig_cwd)


def _run_hamer_impl(
    video_path: Path,
    input_npz: Path,
    output_npz: Path,
    config: dict,
    debug_overlay_dir: Path | None = None,
) -> None:
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

    # Renderer optionnel pour le debug overlay : mesh MANO sur frame source.
    renderer = None
    cam_crop_to_full = None
    scaled_focal_length: float = 0.0
    if debug_overlay_dir is not None:
        try:
            from hamer.utils.renderer import Renderer, cam_crop_to_full as _ccf  # type: ignore[import-not-found]
            renderer = Renderer(hamer_cfg, faces=hamer_model.mano.faces)
            cam_crop_to_full = _ccf
            scaled_focal_length = hamer_cfg.EXTRA.FOCAL_LENGTH / hamer_cfg.MODEL.IMAGE_SIZE
            logger.info("Renderer HaMeR initialisé (focal=%.1f)", scaled_focal_length)
        except Exception as e:
            logger.warning("Impossible d'initialiser le renderer HaMeR : %s", e)
            debug_overlay_dir = None

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

                # Debug overlay : rend la mesh MANO prédite sur la frame source
                if renderer is not None and debug_overlay_dir is not None:
                    try:
                        verts_b = out["pred_vertices"].detach().cpu().numpy()  # (B, 778, 3)
                        # Pour la main gauche, HaMeR flippe X dans le crop → on
                        # désinverse pour aligner avec l'image source.
                        rights = batch["right"].cpu().numpy()
                        for i in range(verts_b.shape[0]):
                            verts_b[i, :, 0] = (2 * rights[i] - 1) * verts_b[i, :, 0]

                        # Camera : crop → frame entière
                        box_center = batch["box_center"].cpu().numpy()
                        box_size = batch["box_size"].cpu().numpy()
                        H, W = frame_bgr.shape[:2]
                        img_size_t = torch.tensor([[W, H]] * verts_b.shape[0], dtype=torch.float32)
                        focal_full = scaled_focal_length * max(W, H)
                        cam_full = cam_crop_to_full(
                            out["pred_cam"].detach().cpu(),
                            torch.tensor(box_center, dtype=torch.float32),
                            torch.tensor(box_size, dtype=torch.float32),
                            img_size_t,
                            focal_full,
                        ).numpy()  # (B, 3)

                        rgba = renderer.render_rgba_multiple(
                            [verts_b[i] for i in range(verts_b.shape[0])],
                            cam_t=[cam_full[i] for i in range(cam_full.shape[0])],
                            render_res=[W, H],
                            is_right=[bool(rights[i]) for i in range(rights.shape[0])],
                            focal_length=focal_full,
                        )  # (H, W, 4) float ∈ [0,1]

                        # Composite : source BGR + mesh RGBA
                        src = frame_bgr.astype(np.float32) / 255.0  # BGR
                        src_rgb = src[:, :, ::-1]  # → RGB
                        alpha = rgba[:, :, 3:4]
                        composite = src_rgb * (1 - alpha) + rgba[:, :, :3] * alpha
                        composite_bgr = (composite[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)
                        out_path = debug_overlay_dir / f"frame_{t:04d}.png"
                        cv2.imwrite(str(out_path), composite_bgr)
                    except Exception as e:
                        if t == 0:
                            logger.warning("Échec rendu overlay frame 0 : %s", e)
    finally:
        cap.release()

    # Réharmonise les confidences (clip [0, 1])
    anim.confidence_lhand = combine_confidence(anim.confidence_lhand, None)
    anim.confidence_rhand = combine_confidence(anim.confidence_rhand, None)

    # Frames sans détection HaMeR : SMPLer-X y a souvent halluciné une pose
    # chelou (mains hors cadre dans la source). On force la rest pose HaMeR
    # (axis-angle = 0 = main plate dans la convention "flat-rest" de HaMeR).
    LOW_CONF = 0.2
    low_l = anim.confidence_lhand < LOW_CONF
    low_r = anim.confidence_rhand < LOW_CONF
    anim.left_hand_pose[low_l] = 0.0
    anim.right_hand_pose[low_r] = 0.0
    logger.info(
        "Mains réinitialisées (conf < %.2f) : %d frames G / %d frames D sur %d",
        LOW_CONF, int(low_l.sum()), int(low_r.sum()), len(anim.confidence_lhand),
    )

    # NOTE : on avait testé une compensation `-= hands_mean` ici en supposant
    # un décalage de convention MANO entre HaMeR (flat-rest) et SMPL-X Add-on
    # (mean-rest). Empiriquement les doigts partent en hyperextension : c'est
    # que HaMeR ET SMPL-X Add-on utilisent la même convention (probablement
    # flat-rest des deux côtés). Aucune compensation ne doit être appliquée.
    # Les imperfections de pose des doigts restantes viennent du tracker
    # HaMeR lui-même, pas d'un bug de convention.

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
    Les paths relatifs (`third-party/ViTPose/...`, `_DATA/...`) sont résolus au
    cwd → c'est `run_hamer()` qui chdir à la racine du repo HaMeR pour tout le run.
    """
    repo = REPO_ROOT / "pipeline" / "envs" / "hamer" / "repo"
    sys.path.insert(0, str(repo))
    from vitpose_model import ViTPoseModel  # type: ignore[import-not-found]
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ViTPoseModel(device)


def _load_hamer():
    """Charge le modèle HaMeR + sa config."""
    repo = REPO_ROOT / "pipeline" / "envs" / "hamer" / "repo"
    sys.path.insert(0, str(repo))
    from hamer.models import load_hamer  # type: ignore[import-not-found]
    import torch

    # Cherche hamer.ckpt dans plusieurs chemins possibles selon le layout de
    # l'archive demo_data (release-dépendant : data/checkpoints/ vs hamer_ckpts/checkpoints/).
    hamer_models_dir = REPO_ROOT / "pipeline" / "models" / "hamer"
    candidates = [
        hamer_models_dir / "_DATA" / "data" / "checkpoints" / "hamer.ckpt",
        hamer_models_dir / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt",
        hamer_models_dir / "checkpoint.ckpt",
    ]
    weights = next((p for p in candidates if p.exists()), None)
    if weights is None:
        # Recherche large en dernier recours
        found = list(hamer_models_dir.rglob("hamer.ckpt"))
        if found:
            weights = found[0]
        else:
            raise FileNotFoundError(
                "Poids HaMeR manquants. Cherché :\n  - "
                + "\n  - ".join(str(c) for c in candidates)
                + "\nLancer scripts/download_weights.sh"
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


def _load_hands_mean() -> np.ndarray | None:
    """Lit hands_mean (45,) depuis MANO_RIGHT.pkl.

    Cherche dans plusieurs emplacements habituels du repo :
        pipeline/models/mano/MANO_RIGHT.pkl
        pipeline/models/hamer/_DATA/data/mano/MANO_RIGHT.pkl
        pipeline/envs/hamer/repo/_DATA/data/mano/MANO_RIGHT.pkl
    Retourne None si introuvable.
    """
    import pickle

    candidates = [
        REPO_ROOT / "pipeline" / "models" / "mano" / "MANO_RIGHT.pkl",
        REPO_ROOT / "pipeline" / "models" / "hamer" / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl",
        REPO_ROOT / "pipeline" / "envs" / "hamer" / "repo" / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with open(path, "rb") as f:
                mano = pickle.load(f, encoding="latin1")
            if "hands_mean" in mano:
                arr = np.asarray(mano["hands_mean"]).reshape(-1).astype(np.float32)
                logger.info("hands_mean chargé depuis %s (shape=%s, L2=%.3f)",
                            path, arr.shape, float(np.linalg.norm(arr)))
                return arr
        except Exception as e:
            logger.warning("Échec lecture %s : %s", path, e)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extraction HaMeR (mains)")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=str, default="{}")
    parser.add_argument(
        "--debug-overlay", type=Path, default=None,
        help="Dossier de sortie pour les PNG overlay mesh MANO (debug HaMeR).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    config = json.loads(args.config)
    run_hamer(args.video, args.input, args.output, config, debug_overlay_dir=args.debug_overlay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
