#!/usr/bin/env bash
#
# setup.sh — installation complète du pipeline LSF sur serveur Ubuntu 22.04 sans GUI.
#
# Suppose :
#   - utilisateur sudo
#   - GPU NVIDIA présent avec driver ≥ 525 (vérifié par nvidia-smi)
#   - connexion internet
#
# Étapes :
#   0. Vérifications préalables (OS, GPU, sudo)
#   1. Paquets système (apt) — sans deps GUI
#   2. Miniforge (conda + mamba) pour gérer les envs ML
#   3. Blender 4.5 LTS + VRM addon (mode headless)
#   4. Trois envs ML : smplerx, hamer, emoca
#   5. Env orchestrateur (Python pure, pour l'utilisateur final)
#   6. Récap final
#
# Toutes les versions sont pinned dans docs/PIPELINE.md §1.

set -euo pipefail

# Couleurs pour les logs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

log()    { echo -e "${BLUE}[setup]${NC} $*"; }
ok()     { echo -e "${GREEN}[ok]${NC} $*"; }
warn()   { echo -e "${YELLOW}[warn]${NC} $*"; }
fail()   { echo -e "${RED}[fail]${NC} $*"; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log "Racine du repo : $REPO_ROOT"

# ──────────────────────────────────────────────────────────────────────────────
# 0. Vérifications préalables
# ──────────────────────────────────────────────────────────────────────────────

step_0_preflight() {
    log "0. Vérifications préalables…"

    # OS
    if ! grep -qi "ubuntu" /etc/os-release 2>/dev/null && ! grep -qi "fedora" /etc/os-release 2>/dev/null; then
        warn "OS non-Ubuntu/Fedora détecté — l'installation peut échouer"
    fi

    # sudo (skip si root)
    if [[ $EUID -ne 0 ]]; then
        if ! sudo -v; then
            fail "sudo requis. Lance : sudo -v puis relance ce script."
        fi
    fi

    # Disque dispo (besoin minimal ~50 GB pour modèles + envs + Blender)
    AVAIL_GB=$(df -BG "$REPO_ROOT" | awk 'NR==2 {gsub("G","",$4); print $4}')
    log "  Espace disque : ${AVAIL_GB} GB"
    if [[ "$AVAIL_GB" -lt 50 ]]; then
        warn "< 50 GB libres — peut être insuffisant"
    fi

    ok "Vérifications préalables OK"
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. Paquets système (apt) — versions headless uniquement
# ──────────────────────────────────────────────────────────────────────────────

step_1_apt_packages() {
    log "1. Installation des paquets système (apt)…"

    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        curl \
        ca-certificates \
        bzip2 \
        unzip \
        xz-utils \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libosmesa6-dev \
        libegl1 \
        libgles2

    # libgl1 / libegl1 / libgles2 sont nécessaires à Blender en headless pour le moteur
    # de rendu Eevee/Cycles offscreen. Pas de X11 (xvfb) installé par défaut ; ajouter
    # `xvfb` si tu veux faire des renders d'aperçu plus tard.

    ok "Paquets système installés"
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Miniforge (conda + mamba)
# ──────────────────────────────────────────────────────────────────────────────

CONDA_DIR="$HOME/miniforge3"

step_2_miniforge() {
    log "2. Installation de Miniforge (conda + mamba)…"

    if [[ -d "$CONDA_DIR" ]]; then
        ok "Miniforge déjà installé : $CONDA_DIR"
        return
    fi

    local installer="$HOME/Miniforge3-Linux-x86_64.sh"
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O "$installer"
    bash "$installer" -b -p "$CONDA_DIR"
    rm -f "$installer"

    # Initialise conda dans le bashrc utilisateur (mais sans l'activer maintenant)
    "$CONDA_DIR/bin/conda" init bash || true

    ok "Miniforge installé : $CONDA_DIR"
}

source_conda() {
    # shellcheck disable=SC1091
    source "$CONDA_DIR/etc/profile.d/conda.sh"
}

# ──────────────────────────────────────────────────────────────────────────────
# 3. Blender 4.5 LTS + VRM addon
# ──────────────────────────────────────────────────────────────────────────────

BLENDER_VERSION="4.5.9"
BLENDER_DIR="$REPO_ROOT/.tools/blender"
BLENDER_BIN="$BLENDER_DIR/blender"
VRM_ADDON_VERSION="3.27.0"

step_3_blender() {
    log "3. Installation de Blender $BLENDER_VERSION (headless)…"

    if [[ -x "$BLENDER_BIN" ]]; then
        ok "Blender déjà installé : $BLENDER_BIN"
    else
        mkdir -p "$BLENDER_DIR"
        local url="https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
        local tarball="$REPO_ROOT/.tools/blender.tar.xz"
        log "  Téléchargement : $url"
        wget -q "$url" -O "$tarball" || fail "Téléchargement Blender échoué — version $BLENDER_VERSION non disponible ?"

        log "  Extraction…"
        tar -xJf "$tarball" -C "$BLENDER_DIR" --strip-components=1
        rm -f "$tarball"
        [[ -x "$BLENDER_BIN" ]] || fail "Blender extrait mais binaire introuvable : $BLENDER_BIN"
    fi

    # Smoke test : Blender se lance bien en headless
    log "  Smoke test headless…"
    if "$BLENDER_BIN" --background --python-expr "import sys; sys.exit(0)" &>/dev/null; then
        ok "  Blender headless OK"
    else
        fail "  Blender headless KO — vérifier libgl1/libegl1"
    fi

    # Installation du VRM addon
    # Note : le nom du fichier release utilise des underscores (ex. 3_27_0), pas des points.
    log "  Installation VRM addon v$VRM_ADDON_VERSION…"
    local version_underscore="${VRM_ADDON_VERSION//./_}"
    local addon_url="https://github.com/saturday06/VRM-Addon-for-Blender/releases/download/v${VRM_ADDON_VERSION}/VRM_Addon_for_Blender-${version_underscore}.zip"
    local addon_zip="$REPO_ROOT/.tools/vrm_addon.zip"
    wget -q "$addon_url" -O "$addon_zip" || fail "Téléchargement VRM addon échoué — URL: $addon_url"

    # Le nom du module varie selon la version (io_scene_vrm vs VRM_Addon_for_Blender-release).
    # On installe puis on auto-détecte le module pour l'activer.
    "$BLENDER_BIN" --background --python-expr "
import bpy, addon_utils, sys
bpy.ops.preferences.addon_install(filepath='$addon_zip', overwrite=True)
addon_utils.modules_refresh()
vrm_modules = [m.__name__ for m in addon_utils.modules() if 'vrm' in m.__name__.lower()]
if not vrm_modules:
    sys.stderr.write('FAIL: aucun module VRM trouvé après install\n')
    sys.exit(1)
vrm_module = vrm_modules[0]
print(f'Enabling VRM module: {vrm_module}')
bpy.ops.preferences.addon_enable(module=vrm_module)
bpy.ops.wm.save_userpref()
print(f'VRM addon installé et activé: {vrm_module}')
" || fail "Installation VRM addon dans Blender a échoué"

    rm -f "$addon_zip"

    # Smoke test addon en headless — vérifie réellement le chargement
    log "  Smoke test VRM addon en headless…"
    "$BLENDER_BIN" --background --python-expr "
import addon_utils, sys
vrm_modules = [m.__name__ for m in addon_utils.modules() if 'vrm' in m.__name__.lower()]
if not vrm_modules:
    sys.stderr.write('FAIL: module VRM introuvable\n'); sys.exit(1)
vrm_module = vrm_modules[0]
_default, loaded = addon_utils.check(vrm_module)
if not loaded:
    sys.stderr.write(f'FAIL: VRM addon {vrm_module} non chargé\n'); sys.exit(1)
print(f'OK addon VRM chargé en headless: {vrm_module}')
" || fail "L'addon VRM ne se charge pas en mode headless"

    ok "Blender + VRM addon OK"
}

# ──────────────────────────────────────────────────────────────────────────────
# 4. Trois envs ML
# ──────────────────────────────────────────────────────────────────────────────

step_4_ml_envs() {
    log "4. Création des 3 environnements ML…"
    source_conda

    setup_smplerx_env
    setup_hamer_env
    setup_emoca_env

    ok "Environnements ML créés"
}

# ── 4a. SMPLer-X ──
setup_smplerx_env() {
    local env_name="lsf-smplerx"
    local env_dir="$REPO_ROOT/pipeline/envs/smplerx"
    local repo_dir="$env_dir/repo"

    log "  4a. Env SMPLer-X (Python 3.8 + torch 1.12 + cu113)…"

    if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
        warn "    Env $env_name existe déjà — skip création"
    else
        conda create -y -n "$env_name" python=3.8 pip
    fi
    conda activate "$env_name"
    # conda-forge récent n'installe pas pip par défaut — on garantit qu'il est dans l'env
    [[ -x "$CONDA_DIR/envs/$env_name/bin/pip" ]] || conda install -y -n "$env_name" pip

    # Torch 1.12.0 + cu113 (cf. PIPELINE.md §1.1)
    pip install --no-cache-dir \
        torch==1.12.0+cu113 \
        torchvision==0.13.0+cu113 \
        torchaudio==0.12.0 \
        --extra-index-url https://download.pytorch.org/whl/cu113

    # mmcv-full prebuilt wheel
    pip install --no-cache-dir \
        mmcv-full==1.7.1 \
        -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html

    # Reste des deps
    pip install --no-cache-dir -r "$env_dir/requirements.txt"

    # Clone SMPLer-X au commit pinned
    if [[ ! -d "$repo_dir" ]]; then
        log "    Clone SMPLer-X…"
        git clone https://github.com/MotrixLab/SMPLer-X.git "$repo_dir"
        (cd "$repo_dir" && git checkout 064baef0e4ab5277a3297691bc1d46ea5412586f)
    fi

    # Lien symbolique du venv vers pipeline/envs/smplerx/venv pour l'orchestrateur
    local conda_python
    conda_python="$(conda info --base)/envs/$env_name/bin/python"
    mkdir -p "$env_dir/venv/bin"
    ln -sf "$conda_python" "$env_dir/venv/bin/python"

    conda deactivate
    ok "    Env smplerx OK"
}

# ── 4b. HaMeR ──
setup_hamer_env() {
    local env_name="lsf-hamer"
    local env_dir="$REPO_ROOT/pipeline/envs/hamer"
    local repo_dir="$env_dir/repo"

    log "  4b. Env HaMeR (Python 3.10 + torch 1.13 + cu117)…"

    if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
        warn "    Env $env_name existe déjà — skip création"
    else
        conda create -y -n "$env_name" python=3.10 pip
    fi
    conda activate "$env_name"
    [[ -x "$CONDA_DIR/envs/$env_name/bin/pip" ]] || conda install -y -n "$env_name" pip

    # Torch ≤ 1.13 (mmcv 1.3.9 ne supporte pas torch 2.x)
    pip install --no-cache-dir \
        torch==1.13.1+cu117 \
        torchvision==0.14.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117

    # detectron2 from git
    pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

    pip install --no-cache-dir -r "$env_dir/requirements.txt"

    # Clone HaMeR au commit pinned
    if [[ ! -d "$repo_dir" ]]; then
        log "    Clone HaMeR…"
        git clone --recursive https://github.com/geopavlakos/hamer.git "$repo_dir"
        (cd "$repo_dir" && git checkout 3a01849f4148352e9260b69bf28b65d1671a4905)
    fi

    # Install ViTPose vendored dans le repo
    if [[ -d "$repo_dir/third-party/ViTPose" ]]; then
        pip install -e "$repo_dir/third-party/ViTPose"
    fi
    pip install -e "$repo_dir"

    local conda_python
    conda_python="$(conda info --base)/envs/$env_name/bin/python"
    mkdir -p "$env_dir/venv/bin"
    ln -sf "$conda_python" "$env_dir/venv/bin/python"

    conda deactivate
    ok "    Env hamer OK"
}

# ── 4c. EMOCA v2 ──
setup_emoca_env() {
    local env_name="lsf-emoca"
    local env_dir="$REPO_ROOT/pipeline/envs/emoca"
    local repo_dir="$env_dir/repo"

    log "  4c. Env EMOCA v2 (Python 3.8 + torch 1.12.1 + cu113)…"

    if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
        warn "    Env $env_name existe déjà — skip création"
    else
        conda create -y -n "$env_name" python=3.8 pip
    fi
    conda activate "$env_name"
    [[ -x "$CONDA_DIR/envs/$env_name/bin/pip" ]] || conda install -y -n "$env_name" pip

    # Torch 1.12.1 + cu113
    pip install --no-cache-dir \
        torch==1.12.1+cu113 \
        torchvision==0.13.1+cu113 \
        torchaudio==0.12.1 \
        --extra-index-url https://download.pytorch.org/whl/cu113

    # Cython AVANT pytorch3d (cf. README EMOCA)
    pip install --no-cache-dir Cython==0.29.14

    # PyTorch3D 0.6.2 build depuis source (étape la plus fragile)
    log "    Build pytorch3d 0.6.2 depuis source (peut prendre 10 min)…"
    pip install --no-cache-dir 'git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2' \
        || warn "    pytorch3d build failed — voir docs/TROUBLESHOOTING.md"

    pip install --no-cache-dir -r "$env_dir/requirements.txt"

    # Clone EMOCA branche release/EMOCA_v2
    if [[ ! -d "$repo_dir" ]]; then
        log "    Clone EMOCA…"
        git clone --recursive https://github.com/radekd91/emoca.git "$repo_dir"
        (cd "$repo_dir" && git checkout e0be0dbc2d32629ae384ae10c0b7974948c994fd)
    fi
    pip install -e "$repo_dir"

    local conda_python
    conda_python="$(conda info --base)/envs/$env_name/bin/python"
    mkdir -p "$env_dir/venv/bin"
    ln -sf "$conda_python" "$env_dir/venv/bin/python"

    conda deactivate
    ok "    Env emoca OK"
}

# ──────────────────────────────────────────────────────────────────────────────
# 5. Env orchestrateur
# ──────────────────────────────────────────────────────────────────────────────

step_5_orchestrator_env() {
    log "5. Env orchestrateur (pure-Python, Python 3.10)…"
    source_conda

    local env_name="lsf-orchestrator"
    if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
        warn "  Env $env_name existe déjà — skip"
    else
        conda create -y -n "$env_name" python=3.10 pip
    fi
    conda activate "$env_name"
    [[ -x "$CONDA_DIR/envs/$env_name/bin/pip" ]] || conda install -y -n "$env_name" pip
    pip install --no-cache-dir -r "$REPO_ROOT/pipeline/requirements.txt"
    conda deactivate

    ok "Env orchestrateur OK"
}

# ──────────────────────────────────────────────────────────────────────────────
# 6. Récap final
# ──────────────────────────────────────────────────────────────────────────────

step_6_summary() {
    cat <<EOF

${GREEN}╔════════════════════════════════════════════════════════════════════╗
║                    INSTALLATION TERMINÉE                           ║
╚════════════════════════════════════════════════════════════════════╝${NC}

Environnements créés :
  • lsf-orchestrator (Python 3.10) — pour lancer le pipeline
  • lsf-smplerx      (Python 3.8)  — extraction corps
  • lsf-hamer        (Python 3.10) — raffinement mains
  • lsf-emoca        (Python 3.8)  — raffinement visage

Blender headless installé : $BLENDER_BIN

${YELLOW}PROCHAINES ÉTAPES${NC}

1. Crée tes comptes sur les sites MPI (gratuit, 5 min) — cf. docs/MODELS.md :
   - https://smpl-x.is.tue.mpg.de
   - https://smpl.is.tue.mpg.de
   - https://mano.is.tue.mpg.de
   - https://flame.is.tue.mpg.de
   - https://emoca.is.tue.mpg.de

2. Lance le téléchargement des poids :
   ${BLUE}bash scripts/download_weights.sh${NC}

3. Vérifie que tout est OK :
   ${BLUE}conda activate lsf-orchestrator${NC}
   ${BLUE}python scripts/verify_env.py${NC}

4. Lance ton premier pipeline :
   ${BLUE}python pipeline/pipeline.py --video data/input/X.mp4 --avatar data/avatars/Y.vrm --output data/output/Z.vrma${NC}

EOF
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

main() {
    step_0_preflight
    step_1_apt_packages
    step_2_miniforge
    step_3_blender
    step_4_ml_envs
    step_5_orchestrator_env
    step_6_summary
}

main "$@"
