"""Orchestrateur principal du pipeline LSF.

Enchaîne les 4 étapes :
    1. SMPLer-X (env smplerx) → params SMPL-X par frame
    2. HaMeR    (env hamer)   → raffinement mains MANO
    3. EMOCA    (env emoca)   → raffinement visage FLAME
    4. smoothing + retarget Blender → animation .vrma

Chaque étape ML est invoquée comme un sous-process avec son env Python dédié,
parce que les trois modèles ont des dépendances mutuellement incompatibles
(cf. docs/PIPELINE.md §0.1).

Usage :
    python pipeline/pipeline.py \\
        --video data/input/clip.mp4 \\
        --avatar data/avatars/alicia.vrm \\
        --output data/output/clip.vrma

    # Validation seule (vérifie les chemins et les imports, ne lance aucun modèle)
    python pipeline/pipeline.py --video … --avatar … --output … --dry-run

    # Génère en plus une vidéo de debug avec mesh superposé
    python pipeline/pipeline.py --video … --avatar … --output … --debug-overlay
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from pipeline.animation_npz import Animation
from pipeline.config import (
    REPO_ROOT, deep_merge, load_config, resolve_path, setup_logging,
)
from pipeline.smoothing import SmoothingParams, smooth_animation
from pipeline.video_io import probe_video
from pipeline.vrm_inspector import is_vrm_compatible

logger = logging.getLogger("pipeline")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline LSF vidéo → animation .vrma",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", type=Path, required=True,
                        help="Vidéo d'entrée (.mp4)")
    parser.add_argument("--avatar", type=Path, required=True,
                        help="Avatar VRM (.vrm) — utilisé pour le retargeting")
    parser.add_argument("--output", type=Path, required=True,
                        help="Animation de sortie (.vrma)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Chemin vers config.yaml alternatif")
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS de sortie (override config.pipeline.output_fps)")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Conserve les fichiers temporaires intermédiaires")
    parser.add_argument("--dry-run", action="store_true",
                        help="Valide les chemins/envs/imports sans charger les modèles")
    parser.add_argument("--debug-overlay", action="store_true",
                        help="Génère une vidéo .mp4 avec mesh superposé en parallèle")
    parser.add_argument("--skip-hands", action="store_true",
                        help="Ne pas lancer HaMeR (mains de SMPLer-X gardées)")
    parser.add_argument("--skip-face", action="store_true",
                        help="Ne pas lancer EMOCA (visage de SMPLer-X gardé)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config)

    # Override fps si demandé
    if args.fps is not None:
        config["pipeline"]["output_fps"] = float(args.fps)

    # Override skip flags
    if args.skip_hands:
        config["pipeline"]["use_hamer_for_hands"] = False
    if args.skip_face:
        config["pipeline"]["use_emoca_for_face"] = False

    # Résolution des chemins
    video_path = args.video.resolve()
    avatar_path = args.avatar.resolve()
    output_path = args.output.resolve()

    logger.info("=" * 70)
    logger.info("Pipeline LSF — démarrage")
    logger.info("  Vidéo  : %s", video_path)
    logger.info("  Avatar : %s", avatar_path)
    logger.info("  Sortie : %s", output_path)
    logger.info("  Mode   : %s", "DRY-RUN" if args.dry_run else "PRODUCTION")
    logger.info("=" * 70)

    # ── Phase 0 : validation des entrées et de l'environnement ────────────
    _validate_inputs(video_path, avatar_path, output_path)
    _validate_environments(config, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("[DRY-RUN] Validation OK — aucun modèle chargé.")
        return 0

    # ── Phase 1 → 4 : extraction et retargeting ──────────────────────────
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="lsf-pipeline-") as tmp_str:
        tmp_dir = Path(tmp_str)
        if args.keep_tmp:
            # Si --keep-tmp, on copie en sortie au lieu de laisser le with le supprimer
            tmp_dir = output_path.parent / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)

        # Étape 1 — SMPLer-X
        smplerx_npz = tmp_dir / "smplerx.npz"
        _run_smplerx(video_path, smplerx_npz, config)

        # Étape 2 — HaMeR (optionnelle)
        hamer_npz = smplerx_npz
        if config["pipeline"]["use_hamer_for_hands"]:
            hamer_npz = tmp_dir / "hamer.npz"
            _run_hamer(video_path, smplerx_npz, hamer_npz, config)
        else:
            logger.info("HaMeR désactivé — mains de SMPLer-X conservées")

        # Étape 3 — EMOCA (optionnelle)
        emoca_npz = hamer_npz
        if config["pipeline"]["use_emoca_for_face"]:
            emoca_npz = tmp_dir / "emoca.npz"
            _run_emoca(video_path, hamer_npz, emoca_npz, config)
        else:
            logger.info("EMOCA désactivé — visage de SMPLer-X conservé")

        # Étape 4a — lissage One-Euro (pure Python)
        anim = Animation.load(emoca_npz)
        params = SmoothingParams.from_config(
            fps=config["pipeline"]["output_fps"],
            smoothing_cfg=config.get("smoothing", {}),
        )
        if config.get("smoothing", {}).get("enabled", True):
            anim = smooth_animation(anim, params)
        else:
            logger.info("Lissage désactivé via config.smoothing.enabled=false")

        anim = anim.with_meta(
            generated_at=datetime.now(timezone.utc).isoformat(),
            output_fps=config["pipeline"]["output_fps"],
            smoothing_params=params.__dict__,
            avatar=str(avatar_path),
        )
        animation_npz = tmp_dir / "animation.npz"
        anim.save(animation_npz)

        # Étape 4b — retargeting Blender → .vrma
        _run_retarget(animation_npz, avatar_path, output_path, config)

        # Étape 5 — debug overlay (optionnel)
        if args.debug_overlay:
            overlay_path = output_path.with_suffix(".debug.mp4")
            _run_debug_overlay(video_path, animation_npz, overlay_path, config)

        # --keep-tmp : déplacer les fichiers intermédiaires à côté de l'output
        if args.keep_tmp:
            logger.info("Fichiers intermédiaires conservés dans %s", tmp_dir)

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("Pipeline terminé en %.1f s — %s", elapsed, output_path)
    logger.info("=" * 70)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Étapes
# ──────────────────────────────────────────────────────────────────────────────

def _validate_inputs(video: Path, avatar: Path, output: Path) -> None:
    if not video.exists():
        raise FileNotFoundError(f"Vidéo introuvable : {video}")
    if video.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        logger.warning("Extension vidéo inattendue : %s — tentative quand même", video.suffix)

    if not avatar.exists():
        raise FileNotFoundError(f"Avatar introuvable : {avatar}")
    if avatar.suffix.lower() != ".vrm":
        raise ValueError(f"Avatar doit être un .vrm, reçu : {avatar.suffix}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() not in {".vrma", ".glb"}:
        logger.warning("Extension de sortie inattendue : %s (recommandé : .vrma)", output.suffix)

    info = probe_video(video)
    logger.info("Vidéo : %dx%d @ %.2f fps, %d frames, %.1f s",
                info.width, info.height, info.fps, info.num_frames, info.duration_seconds)

    compatible, problems = is_vrm_compatible(avatar)
    if not compatible:
        logger.error("Avatar VRM incompatible :")
        for p in problems:
            logger.error("  %s", p)
        raise ValueError(f"Avatar {avatar} ne contient pas tous les bones requis")
    logger.info("Avatar VRM validé : %s", avatar.name)


def _validate_environments(config: dict, *, dry_run: bool) -> None:
    """Vérifie que les binaires Python par env existent (sauf si dry-run et envs absents).

    En dry-run on tolère l'absence des envs (pour permettre la validation sur Mac
    avant de cloner sur Linux). En mode production, les envs doivent exister.
    """
    paths = config.get("paths", {})
    needed = {
        "smplerx": paths.get("python_smplerx", "pipeline/envs/smplerx/venv/bin/python"),
        "hamer":   paths.get("python_hamer",   "pipeline/envs/hamer/venv/bin/python"),
        "emoca":   paths.get("python_emoca",   "pipeline/envs/emoca/venv/bin/python"),
    }

    if not config["pipeline"]["use_hamer_for_hands"]:
        needed.pop("hamer", None)
    if not config["pipeline"]["use_emoca_for_face"]:
        needed.pop("emoca", None)

    missing: list[str] = []
    for name, p in needed.items():
        path = resolve_path(p)
        if not path.exists():
            missing.append(f"  env {name}: {path}")

    if missing:
        msg = "Environnements Python ML manquants :\n" + "\n".join(missing)
        if dry_run:
            logger.warning("[DRY-RUN] %s", msg)
            logger.warning("[DRY-RUN] (acceptable en validation locale, "
                           "mais devra être corrigé sur la machine GPU)")
        else:
            logger.error(msg)
            raise FileNotFoundError(
                "Lancer `bash scripts/setup.sh` sur la machine cible pour installer les envs"
            )


def _run_smplerx(video: Path, output_npz: Path, config: dict) -> None:
    python_bin = resolve_path(config["paths"]["python_smplerx"])
    script = REPO_ROOT / "pipeline" / "envs" / "smplerx" / "extract_smplerx.py"
    logger.info(">>> Étape 1/4 : SMPLer-X")
    _run_subprocess([
        str(python_bin), str(script),
        "--video", str(video),
        "--output", str(output_npz),
        "--config", json.dumps(config["pipeline"]),
    ])


def _run_hamer(video: Path, input_npz: Path, output_npz: Path, config: dict) -> None:
    python_bin = resolve_path(config["paths"]["python_hamer"])
    script = REPO_ROOT / "pipeline" / "envs" / "hamer" / "extract_hamer.py"
    logger.info(">>> Étape 2/4 : HaMeR")
    _run_subprocess([
        str(python_bin), str(script),
        "--video", str(video),
        "--input", str(input_npz),
        "--output", str(output_npz),
        "--config", json.dumps(config["pipeline"]),
    ])


def _run_emoca(video: Path, input_npz: Path, output_npz: Path, config: dict) -> None:
    python_bin = resolve_path(config["paths"]["python_emoca"])
    script = REPO_ROOT / "pipeline" / "envs" / "emoca" / "extract_emoca.py"
    logger.info(">>> Étape 3/4 : EMOCA")
    _run_subprocess([
        str(python_bin), str(script),
        "--video", str(video),
        "--input", str(input_npz),
        "--output", str(output_npz),
        "--config", json.dumps(config["pipeline"]),
    ])


def _run_retarget(animation_npz: Path, avatar: Path, output: Path, config: dict) -> None:
    blender = config["paths"].get("blender_bin", "blender")
    if not shutil.which(blender):
        raise FileNotFoundError(
            f"Binaire Blender introuvable : {blender}. "
            "Installer Blender 4.5 LTS — cf. scripts/setup.sh"
        )
    retarget_script = REPO_ROOT / "pipeline" / "retarget.py"
    logger.info(">>> Étape 4/4 : Retargeting Blender → %s", output.suffix)
    _run_subprocess([
        blender, "-b", "--addons", "io_scene_vrm",
        "--python", str(retarget_script),
        "--",
        "--avatar", str(avatar),
        "--animation", str(animation_npz),
        "--output", str(output),
    ])


def _run_debug_overlay(video: Path, animation_npz: Path, output: Path, config: dict) -> None:
    """Lance la génération de la vidéo de debug avec mesh superposé.

    Utilise le Python orchestrateur (pure numpy + opencv), pas un env ML —
    on rend juste le mesh SMPL-X projeté en wireframe simple. Le rendu
    photoréaliste avec pyrender nécessiterait un env GPU et n'est pas inclus.
    """
    overlay_script = REPO_ROOT / "pipeline" / "debug_overlay.py"
    logger.info(">>> Debug overlay : %s", output)
    _run_subprocess([
        sys.executable, str(overlay_script),
        "--video", str(video),
        "--animation", str(animation_npz),
        "--output", str(output),
        "--config", json.dumps(config),
    ])


def _run_subprocess(cmd: list[str]) -> None:
    """Lance un sous-process et propage stdout/stderr en streaming.

    Lève subprocess.CalledProcessError si le code de retour ≠ 0.
    """
    logger.debug("Sous-process : %s", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


if __name__ == "__main__":
    sys.exit(main())
