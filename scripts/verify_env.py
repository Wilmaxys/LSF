#!/usr/bin/env python3
"""verify_env.py — vérifie que l'environnement est prêt pour le pipeline LSF.

À lancer dans l'env orchestrateur (`conda activate lsf-orchestrator`) APRÈS
setup.sh + download_weights.sh.

Vérifie :
    1. Python ≥ 3.10 dans l'env courant
    2. CUDA disponible (via nvidia-smi)
    3. VRAM suffisante (≥ 16 GB recommandé)
    4. Imports orchestrateur (numpy, opencv-python-headless, pygltflib, etc.)
    5. Fichiers de poids présents avec tailles plausibles
    6. Checksums (si CHECKSUMS.sha256 présent)
    7. Binaires Python par env ML existent
    8. Blender + VRM addon en mode headless
    9. Inspection d'un VRM de référence (si fourni)

Tout problème est listé ; le script retourne 0 si tout est OK, 1 sinon.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# Couleurs ANSI minimalistes (le terminal moderne les supporte ; sinon ignorées)
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
NC = "\033[0m"


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    severity: str = "error"  # error | warning


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, message: str = "", severity: str = "error") -> None:
        self.results.append(CheckResult(name, ok, message, severity))
        marker = f"{GREEN}✓{NC}" if ok else (
            f"{YELLOW}⚠{NC}" if severity == "warning" else f"{RED}✗{NC}"
        )
        line = f"  {marker} {name}"
        if message:
            line += f"  — {message}"
        print(line)

    def has_errors(self) -> bool:
        return any(not r.ok and r.severity == "error" for r in self.results)

    def has_warnings(self) -> bool:
        return any(not r.ok and r.severity == "warning" for r in self.results)

    def summary(self) -> None:
        n_ok = sum(1 for r in self.results if r.ok)
        n_err = sum(1 for r in self.results if not r.ok and r.severity == "error")
        n_warn = sum(1 for r in self.results if not r.ok and r.severity == "warning")
        print()
        print(f"  {n_ok} OK, {n_err} erreurs, {n_warn} warnings")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vérification de l'environnement LSF")
    parser.add_argument("--vrm", type=Path, default=None,
                        help="Avatar VRM de test (optionnel)")
    parser.add_argument("--skip-checksums", action="store_true",
                        help="Ne pas vérifier les SHA-256 (plus rapide)")
    args = parser.parse_args(argv)

    report = Report()

    print(f"{BLUE}=== Vérification de l'environnement LSF ==={NC}")

    print(f"\n{BLUE}1. Python orchestrateur{NC}")
    check_python(report)

    print(f"\n{BLUE}2. CUDA + GPU{NC}")
    check_cuda(report)

    print(f"\n{BLUE}3. Imports orchestrateur{NC}")
    check_orchestrator_imports(report)

    print(f"\n{BLUE}4. Fichiers de poids ML{NC}")
    check_weights(report, skip_checksums=args.skip_checksums)

    print(f"\n{BLUE}5. Environnements ML{NC}")
    check_ml_envs(report)

    print(f"\n{BLUE}6. Blender + VRM addon{NC}")
    check_blender(report)

    if args.vrm is not None:
        print(f"\n{BLUE}7. Inspection VRM de test{NC}")
        check_vrm(report, args.vrm)

    report.summary()
    if report.has_errors():
        print(f"\n{RED}✗ Environnement non-prêt.{NC} Corriger les erreurs ci-dessus.")
        return 1
    if report.has_warnings():
        print(f"\n{YELLOW}⚠ Environnement opérationnel avec avertissements.{NC}")
    else:
        print(f"\n{GREEN}✓ Environnement entièrement prêt.{NC}")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Checks individuels
# ──────────────────────────────────────────────────────────────────────────────

def check_python(report: Report) -> None:
    v = sys.version_info
    ok = v >= (3, 10)
    report.add(
        f"Python {v.major}.{v.minor}.{v.micro}",
        ok,
        "Attendu ≥ 3.10 pour l'orchestrateur" if not ok else "",
    )


def check_cuda(report: Report) -> None:
    if not shutil.which("nvidia-smi"):
        report.add("nvidia-smi présent", False, "Driver NVIDIA non installé ?")
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        report.add("nvidia-smi exécutable", False, f"Erreur : {exc}")
        return

    line = out.strip().splitlines()[0]
    name, mem_str, driver = [x.strip() for x in line.split(",")]
    mem_mib = int(mem_str)
    report.add(f"GPU détecté : {name}", True, f"{mem_mib} MiB, driver {driver}")
    report.add(
        "VRAM ≥ 16 GiB",
        mem_mib >= 16000,
        f"VRAM = {mem_mib} MiB (recommandé ≥ 16384 MiB pour SMPLer-X H32)",
        severity="warning" if mem_mib < 16000 else "error",
    )

    driver_major = int(driver.split(".")[0])
    report.add(
        "Driver ≥ 525",
        driver_major >= 525,
        f"Driver = {driver} (recommandé ≥ 525 pour CUDA 12.x ; ≥ 470 minimum pour cu113)",
        severity="warning",
    )


def check_orchestrator_imports(report: Report) -> None:
    modules = ["numpy", "cv2", "yaml", "pygltflib", "tqdm"]
    for m in modules:
        try:
            __import__(m)
            report.add(f"import {m}", True)
        except ImportError as exc:
            report.add(f"import {m}", False, str(exc))


def check_weights(report: Report, *, skip_checksums: bool = False) -> None:
    """Vérifie la présence et la taille des fichiers de poids."""
    models_dir = REPO_ROOT / "pipeline" / "models"

    expected_files: list[tuple[str, str, int]] = [
        # (nom, chemin relatif à models_dir, taille minimale en bytes)
        ("SMPLer-X H32*",                  "smplerx/smpler_x_h32_correct.pth.tar",     2_000_000_000),
        ("mmdet detector",                 "mmdet/faster_rcnn_r50_fpn_1x_coco.pth",     150_000_000),
        ("HaMeR ViTDet",                   "hamer/model_final_f05665.pkl",            2_000_000_000),
        ("SMPL-X NEUTRAL",                 "smplx/SMPLX_NEUTRAL.npz",                   100_000_000),
        ("MANO right hand",                "mano/MANO_RIGHT.pkl",                         3_000_000),
        ("EMOCA v2 model",                 "emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20", 0),  # dossier
    ]

    for name, rel_path, min_size in expected_files:
        path = models_dir / rel_path
        if not path.exists():
            report.add(f"Fichier : {name}", False,
                       f"Manquant : {path}. Lancer scripts/download_weights.sh")
            continue
        if path.is_file():
            size = path.stat().st_size
            if size < min_size:
                report.add(f"Fichier : {name}", False,
                           f"Taille suspecte : {size} bytes < {min_size} attendus")
                continue
            report.add(f"Fichier : {name}", True, f"{_human_size(size)}")
        elif path.is_dir():
            report.add(f"Dossier : {name}", True, str(path))

    if skip_checksums:
        return

    checksums = models_dir / "CHECKSUMS.sha256"
    if not checksums.exists():
        report.add("CHECKSUMS.sha256", False,
                   "Fichier de checksums absent ; relancer download_weights.sh",
                   severity="warning")
        return

    print(f"\n  {BLUE}Vérification SHA-256 (peut prendre 1-2 minutes)…{NC}")
    bad = 0
    total = 0
    with checksums.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sha, _, rel = line.partition("  ")
            rel = rel.lstrip("./")
            path = models_dir / rel
            if not path.is_file():
                continue
            total += 1
            actual = _sha256(path)
            if actual != sha:
                bad += 1
                print(f"    {RED}✗{NC} {rel} : attendu {sha[:16]}…, reçu {actual[:16]}…")
    if total == 0:
        report.add("Checksums", False, "Aucun fichier référencé dans CHECKSUMS.sha256",
                   severity="warning")
    else:
        report.add(f"Checksums ({total} fichiers)",
                   bad == 0,
                   f"{bad} corrompu(s)" if bad > 0 else "Tous OK")


def check_ml_envs(report: Report) -> None:
    """Vérifie que les binaires Python des envs ML sont accessibles."""
    envs = {
        "smplerx": REPO_ROOT / "pipeline" / "envs" / "smplerx" / "venv" / "bin" / "python",
        "hamer":   REPO_ROOT / "pipeline" / "envs" / "hamer"   / "venv" / "bin" / "python",
        "emoca":   REPO_ROOT / "pipeline" / "envs" / "emoca"   / "venv" / "bin" / "python",
    }

    for name, py in envs.items():
        # Le path peut être un symlink vers conda — on suit
        if not py.exists():
            report.add(f"Env {name}", False, f"Python introuvable : {py}")
            continue
        try:
            out = subprocess.check_output([str(py), "--version"], text=True, timeout=10)
            report.add(f"Env {name}", True, out.strip())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            report.add(f"Env {name}", False, str(exc))

    # Test rapide d'import torch + CUDA dans chaque env
    for name, py in envs.items():
        if not py.exists():
            continue
        try:
            out = subprocess.check_output([
                str(py), "-c",
                "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')",
            ], text=True, timeout=30)
            ok = "cuda=True" in out
            report.add(f"  Env {name} : torch+CUDA", ok, out.strip())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            report.add(f"  Env {name} : torch+CUDA", False, str(exc))


def check_blender(report: Report) -> None:
    """Test que Blender se lance en headless avec le VRM addon."""
    blender = REPO_ROOT / ".tools" / "blender" / "blender"
    if not blender.exists():
        # Fallback : Blender installé système-wide
        if shutil.which("blender"):
            blender = Path(shutil.which("blender"))  # type: ignore[arg-type]
        else:
            report.add("Blender installé", False,
                       f"Binaire absent : {blender}. Lancer scripts/setup.sh")
            return

    try:
        out = subprocess.check_output(
            [str(blender), "--version"],
            text=True, timeout=15,
        )
        report.add(f"Blender exécutable", True, out.splitlines()[0])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        report.add("Blender exécutable", False, str(exc))
        return

    # Test addon VRM en mode background
    try:
        out = subprocess.check_output(
            [str(blender), "-b", "--addons", "io_scene_vrm",
             "--python-expr",
             "import bpy; print('VRM_ADDON_OK')"],
            text=True, timeout=30,
        )
        ok = "VRM_ADDON_OK" in out
        report.add("VRM addon en headless", ok, "" if ok else "Addon non chargé")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        report.add("VRM addon en headless", False, str(exc))


def check_vrm(report: Report, vrm_path: Path) -> None:
    """Inspecte un VRM de référence pour valider qu'il est compatible."""
    if not vrm_path.exists():
        report.add(f"VRM test : {vrm_path}", False, "Fichier absent")
        return

    try:
        from pipeline.vrm_inspector import inspect, is_vrm_compatible
    except ImportError as exc:
        report.add("Import vrm_inspector", False, str(exc))
        return

    try:
        meta = inspect(vrm_path)
        report.add(f"VRM lisible : {vrm_path.name}", True,
                   f"version {meta.version}, {len(meta.humanoid_bones)} bones, "
                   f"{len(meta.expressions)} expressions")
    except Exception as exc:  # noqa: BLE001
        report.add(f"VRM lisible : {vrm_path.name}", False, str(exc))
        return

    ok, problems = is_vrm_compatible(vrm_path)
    if ok:
        report.add("VRM bones obligatoires", True)
    else:
        report.add("VRM bones obligatoires", False, "; ".join(problems[:3]))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024  # type: ignore[assignment]
    return f"{size:.1f} PB"


if __name__ == "__main__":
    sys.exit(main())
