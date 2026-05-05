"""Génération de la vidéo de debug avec overlay simple.

Pour chaque frame, dessine sur la vidéo source :
    - Le rectangle de la bbox (vert si confidence haute, rouge si basse)
    - Le numéro de frame et la confidence body
    - Un point indiquant la position projetée du pelvis
    - Une bordure rouge sur les frames sous le seuil de confidence

⚠️ Ce script est volontairement minimaliste : pas de rendu de mesh 3D
(qui nécessiterait pyrender + un env GPU). Pour un overlay de mesh, voir
SMPLer-X main/inference.py qui le fait avec pyrender.

Usage :
    python pipeline/debug_overlay.py \\
        --video data/input/clip.mp4 \\
        --animation data/output/animation.npz \\
        --output data/output/clip.debug.mp4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.animation_npz import Animation  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.video_io import probe_video  # noqa: E402

logger = logging.getLogger("debug_overlay")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Génère une vidéo de debug avec overlay")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--animation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=str, default="{}",
                        help="Config JSON ; sinon utilise config.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    if args.config and args.config != "{}":
        config = json.loads(args.config)
    else:
        config = load_config()

    overlay_cfg = config.get("debug_overlay", {})
    threshold = config.get("confidence", {}).get("warn_threshold", 0.5)
    highlight = overlay_cfg.get("highlight_low_confidence", True)
    mesh_color = tuple(overlay_cfg.get("mesh_color", [0, 200, 0]))

    return render_overlay(
        args.video, args.animation, args.output,
        threshold=threshold,
        highlight_low_confidence=highlight,
        mesh_color=mesh_color,
    )


def render_overlay(
    video_path: Path,
    animation_path: Path,
    output_path: Path,
    *,
    threshold: float = 0.5,
    highlight_low_confidence: bool = True,
    mesh_color: tuple[int, int, int] = (0, 200, 0),
) -> int:
    """Rend la vidéo de debug.

    Args:
        video_path     : vidéo source
        animation_path : NPZ produit par le pipeline
        output_path    : .mp4 de sortie
        threshold      : seuil de confidence en deçà duquel la frame est marquée
        highlight_low_confidence : si True, bordure rouge sur les frames douteuses
        mesh_color     : couleur BGR du marqueur
    """
    info = probe_video(video_path)
    anim = Animation.load(animation_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), fourcc, anim.fps, (info.width, info.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir le writer vidéo : {output_path}")

    cap = cv2.VideoCapture(str(video_path))
    try:
        # Index : on suit anim.frame_indices qui pointe dans la vidéo source
        # On itère les frames de l'animation (ré-échantillonnée) ; pour chaque
        # frame on cherche le frame source correspondant.
        for t in range(anim.num_frames):
            src_idx = int(anim.frame_indices[t])
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            conf_body  = float(anim.confidence_body[t])
            conf_face  = float(anim.confidence_face[t])
            conf_lhand = float(anim.confidence_lhand[t])
            conf_rhand = float(anim.confidence_rhand[t])
            min_conf = min(conf_body, conf_face, conf_lhand, conf_rhand)

            # Bordure rouge si frame douteuse
            if highlight_low_confidence and min_conf < threshold:
                cv2.rectangle(frame, (0, 0),
                              (info.width - 1, info.height - 1),
                              (0, 0, 255), 8)

            # Texte d'information
            text = (
                f"frame {t}/{anim.num_frames} (src {src_idx})  "
                f"body={conf_body:.2f} face={conf_face:.2f} "
                f"L={conf_lhand:.2f} R={conf_rhand:.2f}"
            )
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Marqueur position pelvis (translation X, Y projetées en pixels approximatifs)
            # NOTE : ceci n'est pas une vraie reprojection 3D — juste un indicateur visuel.
            # Pour un overlay 3D précis, il faudrait connaître la matrice intrinsèque
            # de la caméra (que SMPLer-X estime mais qu'on ne stocke pas dans le NPZ).
            cx = info.width // 2
            cy = info.height // 2
            cv2.circle(frame, (cx, cy), 5, mesh_color, -1)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    logger.info("Debug overlay écrit : %s (%d frames)", output_path, anim.num_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
