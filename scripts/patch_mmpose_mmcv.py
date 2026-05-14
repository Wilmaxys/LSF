#!/usr/bin/env python3
"""Patch mmpose 0.29.0 pour accepter mmcv-full 1.7.1+

Le problème : mmpose 0.29.0 a un assert qui exige mmcv <= 1.7.0, mais notre
env utilise mmcv-full 1.7.1 (la dernière wheel cu113/torch1.12 fournie par
OpenMMLab). Les changements entre 1.7.0 et 1.7.1 sont mineurs et n'affectent
pas mmpose en pratique.

Fix : on bump simplement `mmcv_maximum_version` dans `mmpose/__init__.py` de
'1.7.0' à '1.99.99'. Idempotent — peut être relancé.

Le script trouve automatiquement le chemin de mmpose dans l'env lsf-smplerx
(pas besoin de l'activer pour exécuter ce patch).

Usage :
    python3 scripts/patch_mmpose_mmcv.py
"""
from pathlib import Path
import sys

# Locate mmpose dans l'env conda lsf-smplerx (sans avoir besoin de l'activer)
HOME = Path.home()
ENV_ROOT = HOME / "miniforge3" / "envs" / "lsf-smplerx"


def find_mmpose_init() -> Path | None:
    candidates = list(ENV_ROOT.glob("lib/python*/site-packages/mmpose/__init__.py"))
    return candidates[0] if candidates else None


def main() -> int:
    init_path = find_mmpose_init()
    if init_path is None:
        print(f"[fail] mmpose introuvable sous {ENV_ROOT}/lib/python*/site-packages/",
              file=sys.stderr)
        print("       L'env lsf-smplerx est-il créé et mmpose installé ?",
              file=sys.stderr)
        return 1

    content = init_path.read_text()
    if "# PATCH LSF" in content:
        print(f"[ok] Déjà patché : {init_path}")
        return 0

    old = "mmcv_maximum_version = '1.7.0'"
    new = "mmcv_maximum_version = '1.99.99'  # PATCH LSF: allow newer mmcv (1.7.1+)"

    if old not in content:
        print(f"[warn] Pattern non trouvé dans {init_path}", file=sys.stderr)
        print("       mmpose a peut-être changé de version. Vérifier manuellement.",
              file=sys.stderr)
        return 2

    init_path.write_text(content.replace(old, new))
    print(f"[ok] Patché : {init_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
