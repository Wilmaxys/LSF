"""Dispatcher CLI alternatif vers les scripts d'extraction par-env.

Permet d'exécuter une seule étape ML pour debug ou test, sans passer par
l'orchestrateur complet.

Usage :
    # Étape SMPLer-X seule (env smplerx activé)
    python pipeline/extract.py --phase smplerx --video X.mp4 --output smplerx.npz

    # Étape HaMeR seule
    python pipeline/extract.py --phase hamer --video X.mp4 --input smplerx.npz --output hamer.npz

    # Étape EMOCA seule
    python pipeline/extract.py --phase emoca --video X.mp4 --input hamer.npz --output emoca.npz

Note : ce script s'exécute dans l'env actif, il ne dispatch pas vers les
sous-process. Pour la pipeline complète avec env-switching, utiliser
pipeline.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pipeline.config import REPO_ROOT, load_config, setup_logging

logger = logging.getLogger("extract")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dispatcher d'extraction ML mono-étape")
    parser.add_argument("--phase", choices=["smplerx", "hamer", "emoca"], required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None,
                        help="NPZ d'entrée (requis pour hamer/emoca, ignoré pour smplerx)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config)

    if args.phase != "smplerx" and args.input is None:
        parser.error(f"--input est requis pour la phase {args.phase}")

    # Import paresseux : ne charge pas torch si --dry-run
    if args.dry_run:
        logger.info("[DRY-RUN] phase=%s, video=%s, output=%s",
                    args.phase, args.video, args.output)
        return 0

    if args.phase == "smplerx":
        sys.path.insert(0, str(REPO_ROOT / "pipeline" / "envs" / "smplerx"))
        from extract_smplerx import run_smplerx
        run_smplerx(args.video, args.output, config["pipeline"])
    elif args.phase == "hamer":
        sys.path.insert(0, str(REPO_ROOT / "pipeline" / "envs" / "hamer"))
        from extract_hamer import run_hamer
        run_hamer(args.video, args.input, args.output, config["pipeline"])
    elif args.phase == "emoca":
        sys.path.insert(0, str(REPO_ROOT / "pipeline" / "envs" / "emoca"))
        from extract_emoca import run_emoca
        run_emoca(args.video, args.input, args.output, config["pipeline"])
    else:
        raise ValueError(f"Phase inconnue : {args.phase}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
