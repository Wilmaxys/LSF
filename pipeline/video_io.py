"""Utilitaires vidéo : extraction de frames, ré-échantillonnage, métadonnées.

Pas de dépendance ML — opencv et ffmpeg uniquement.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Métadonnées d'une vidéo."""
    path: Path
    fps: float
    num_frames: int
    width: int
    height: int
    duration_seconds: float


def probe_video(video_path: str | Path) -> VideoInfo:
    """Lit les métadonnées d'une vidéo via OpenCV.

    Pour des données plus précises (fps réel sur vidéo VFR), utiliser ffprobe.

    Raises:
        FileNotFoundError : vidéo absente
        RuntimeError      : impossible d'ouvrir la vidéo
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Vidéo introuvable : {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if fps <= 0:
        logger.warning("FPS invalide rapporté par OpenCV (%.3f) — fallback à 30", fps)
        fps = 30.0

    duration = n / fps if fps > 0 else 0.0
    return VideoInfo(video_path, fps, n, w, h, duration)


def has_ffmpeg() -> bool:
    """True si ffmpeg est disponible dans le PATH."""
    return shutil.which("ffmpeg") is not None


def has_ffprobe() -> bool:
    """True si ffprobe est disponible dans le PATH."""
    return shutil.which("ffprobe") is not None


def probe_video_ffprobe(video_path: str | Path) -> dict:
    """Lit les métadonnées via ffprobe (plus précis que cv2).

    Retourne le dict JSON brut de ffprobe. Lève si ffprobe absent.
    """
    if not has_ffprobe():
        raise RuntimeError("ffprobe non installé — `apt install ffmpeg` requis")
    video_path = Path(video_path)
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def extract_frames_to_dir(
    video_path: str | Path,
    output_dir: str | Path,
    fps_out: float | None = None,
) -> int:
    """Extrait les frames d'une vidéo dans un dossier en .png.

    Si fps_out est fourni, ré-échantillonne via ffmpeg. Sinon, dump frame-by-frame.
    Retourne le nombre de frames écrites.

    Args:
        video_path : vidéo source
        output_dir : dossier de sortie (créé si absent, vidé si existant)
        fps_out    : fps cible (optionnel, ré-échantillonnage temporel)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not has_ffmpeg():
        raise RuntimeError("ffmpeg non installé — `apt install ffmpeg` requis")

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path)]
    if fps_out is not None:
        cmd.extend(["-vf", f"fps={fps_out}"])
    cmd.extend(["-q:v", "2", str(output_dir / "%06d.png")])

    logger.info("Extraction frames : %s → %s (fps_out=%s)", video_path, output_dir, fps_out)
    subprocess.run(cmd, check=True)

    frames = sorted(output_dir.glob("*.png"))
    logger.info("  %d frames extraites", len(frames))
    return len(frames)


def read_frames_iter(video_path: str | Path):
    """Itère les frames d'une vidéo en BGR uint8 via OpenCV.

    Yields:
        (frame_index: int, frame: np.ndarray (H, W, 3) uint8)
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def resample_indices(
    num_frames_in: int,
    fps_in: float,
    fps_out: float,
) -> np.ndarray:
    """Calcule les indices de frame à conserver pour ré-échantillonner.

    Méthode : nearest-neighbour temporel. Pour des fps proches (ex. 29.97 → 30),
    le résultat est quasi-identité. Pour des conversions importantes (60 → 30),
    on saute des frames.

    Args:
        num_frames_in : nombre de frames de la vidéo source
        fps_in        : fps source
        fps_out       : fps cible

    Returns:
        np.ndarray (T_out,) int32 — indices à conserver dans la séquence source.
    """
    if num_frames_in <= 0 or fps_in <= 0 or fps_out <= 0:
        raise ValueError(
            f"Paramètres invalides : num_frames_in={num_frames_in}, "
            f"fps_in={fps_in}, fps_out={fps_out}"
        )
    duration = num_frames_in / fps_in
    n_out = int(round(duration * fps_out))
    if n_out <= 0:
        return np.zeros(0, dtype=np.int32)
    # Échantillonne uniformément n_out timestamps sur [0, duration)
    timestamps = np.arange(n_out, dtype=np.float64) / fps_out
    indices = np.minimum(
        (timestamps * fps_in).round().astype(np.int32),
        num_frames_in - 1,
    )
    return indices
