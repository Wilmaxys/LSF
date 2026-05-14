#!/usr/bin/env python3
"""Patch SMPLer-X's common/utils/human_models.py pour rendre SMPLX_to_J14.pkl
optionnel.

Le fichier J14 (regressor 14-joints LSP) est utilisé uniquement pour l'évaluation
sur le dataset EHF. Pour l'inférence vidéo, on n'en a pas besoin. Or :
- SMPLer-X le charge inconditionnellement à l'import de human_models.py
- Le fichier n'est pas disponible en standalone sur smpl-x.is.tue.mpg.de
  (il faudrait télécharger les 695 MB du pack EHF)

Ce patch wrap la lecture dans un try/except : si le fichier est présent, il est
chargé ; sinon `self.j14_regressor = None`. Idempotent — peut être relancé.

Usage :
    python scripts/patch_smplerx_j14.py
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "pipeline/envs/smplerx/repo/common/utils/human_models.py"

OLD = """        with open(osp.join(cfg.human_model_path, 'smplx', 'SMPLX_to_J14.pkl'), 'rb') as f:
            self.j14_regressor = pickle.load(f, encoding='latin1')"""

NEW = """        # PATCH LSF : SMPLX_to_J14.pkl est uniquement utilisé pour l'éval EHF.
        # Pour l'inférence vidéo on s'en passe ; le fichier n'est pas non plus
        # disponible en standalone côté MPI.
        _j14_path = osp.join(cfg.human_model_path, 'smplx', 'SMPLX_to_J14.pkl')
        if osp.exists(_j14_path):
            with open(_j14_path, 'rb') as f:
                self.j14_regressor = pickle.load(f, encoding='latin1')
        else:
            self.j14_regressor = None"""


def main() -> int:
    if not TARGET.exists():
        print(f"[fail] Fichier introuvable : {TARGET}", file=sys.stderr)
        print("       Le repo SMPLer-X est-il cloné ? Lance scripts/setup.sh d'abord.",
              file=sys.stderr)
        return 1

    content = TARGET.read_text()
    if "# PATCH LSF : SMPLX_to_J14.pkl" in content:
        print(f"[ok] Déjà patché : {TARGET}")
        return 0

    if OLD not in content:
        print(f"[warn] Pattern non trouvé dans {TARGET}", file=sys.stderr)
        print("       SMPLer-X upstream a peut-être changé. Vérifier manuellement.",
              file=sys.stderr)
        return 2

    TARGET.write_text(content.replace(OLD, NEW))
    print(f"[ok] Patché : {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
