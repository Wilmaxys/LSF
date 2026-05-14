#!/usr/bin/env python3
"""Patch torchgeometry 0.1.2 pour torch >= 1.5

torchgeometry/core/conversions.py contient des `1 - mask_*` où les masks sont
des bool tensors. Depuis torch 1.5, la soustraction sur bool est interdite :

    RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported.

Fix : remplacer `(1 - mask_X)` par `(~mask_X)` (inversion logique, équivalente
pour bool, et reste utilisable dans la multiplication bool*bool qui suit).

Idempotent. Cherche torchgeometry dans l'env lsf-smplerx.

Usage :
    python3 scripts/patch_torchgeometry.py
"""
from pathlib import Path
import sys

HOME = Path.home()
ENV_ROOT = HOME / "miniforge3" / "envs" / "lsf-smplerx"


def find_conversions() -> Path | None:
    candidates = list(
        ENV_ROOT.glob("lib/python*/site-packages/torchgeometry/core/conversions.py")
    )
    return candidates[0] if candidates else None


def main() -> int:
    path = find_conversions()
    if path is None:
        print(f"[fail] torchgeometry introuvable sous {ENV_ROOT}", file=sys.stderr)
        return 1

    content = path.read_text()
    if "# PATCH LSF: bool tensor" in content:
        print(f"[ok] Déjà patché : {path}")
        return 0

    replacements = [
        ("(1 - mask_d2)", "(~mask_d2)"),
        ("(1 - mask_d0_d1)", "(~mask_d0_d1)"),
        ("(1 - mask_d0_nd1)", "(~mask_d0_nd1)"),
    ]
    new = content
    n_changed = 0
    for old, repl in replacements:
        if old in new:
            new = new.replace(old, repl)
            n_changed += 1

    if n_changed == 0:
        print(f"[warn] Aucun pattern à patcher dans {path}", file=sys.stderr)
        print("       torchgeometry a peut-être déjà été patché upstream.",
              file=sys.stderr)
        return 2

    # Marqueur pour idempotence
    new = "# PATCH LSF: bool tensor subtraction fix for torch>=1.5\n" + new
    path.write_text(new)
    print(f"[ok] Patché ({n_changed} remplacements) : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
