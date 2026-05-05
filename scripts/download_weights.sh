#!/usr/bin/env bash
#
# download_weights.sh — télécharge les poids ML pour le pipeline LSF.
#
# Composants :
#   1. SMPLer-X (HuggingFace, public)
#   2. mmdet detector (OpenMMLab, public)
#   3. ViTDet (Facebook AI, public)
#   4. HaMeR demo data (UT Austin, public)
#   5. SMPL-X / SMPL / MANO / FLAME / EMOCA (MPI, **REGISTRATION REQUISE**)
#
# Pour les composants MPI, le script ne peut pas télécharger directement (les
# URLs requièrent un cookie d'authentification post-login). Le script :
#   - vérifie que les fichiers existent (téléchargés à la main)
#   - donne les URLs et instructions claires à suivre
#
# Le script est idempotent : on peut le relancer après un téléchargement manuel.

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${BLUE}[download]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
need_manual() { echo -e "${YELLOW}[manuel]${NC} $*"; }
fail() { echo -e "${RED}[fail]${NC} $*"; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/pipeline/models"
mkdir -p "$MODELS_DIR"

# Compteur d'éléments manquants à télécharger manuellement.
MANUAL_PENDING=0

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# download URL DEST [SHA256]
download() {
    local url="$1"
    local dest="$2"
    local expected_sha="${3:-}"

    if [[ -f "$dest" ]]; then
        ok "  Déjà présent : $(basename "$dest")"
        if [[ -n "$expected_sha" ]]; then
            verify_sha "$dest" "$expected_sha"
        fi
        return
    fi

    mkdir -p "$(dirname "$dest")"
    log "  Téléchargement : $(basename "$dest") depuis $url"
    if ! wget --progress=bar:force -O "$dest.tmp" "$url"; then
        rm -f "$dest.tmp"
        fail "Téléchargement échoué : $url"
    fi
    mv "$dest.tmp" "$dest"
    ok "  Téléchargé : $(basename "$dest") ($(du -h "$dest" | cut -f1))"

    if [[ -n "$expected_sha" ]]; then
        verify_sha "$dest" "$expected_sha"
    fi
}

verify_sha() {
    local file="$1"
    local expected="$2"
    local actual
    actual=$(sha256sum "$file" | awk '{print $1}')
    if [[ "$actual" != "$expected" ]]; then
        fail "Checksum SHA-256 incorrect pour $file : attendu $expected, reçu $actual"
    fi
    ok "    SHA-256 OK"
}

# require_manual NOM CHEMIN URL DESCRIPTION
require_manual() {
    local name="$1"
    local path="$2"
    local url="$3"
    local desc="$4"

    if [[ -f "$path" || -d "$path" ]]; then
        ok "  $name déjà présent : $path"
        return
    fi

    MANUAL_PENDING=$((MANUAL_PENDING + 1))
    cat <<EOF

${YELLOW}═══ TÉLÉCHARGEMENT MANUEL REQUIS — $name ═══${NC}

  Description : $desc
  URL         : $url
  Cible       : $path

  Étapes :
    1. Crée un compte sur le site MPI (si pas déjà fait)
    2. Connecte-toi et accepte la licence
    3. Télécharge le(s) fichier(s)
    4. Place-les dans le dossier cible ci-dessus
    5. Relance ce script (idempotent)

EOF
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. SMPLer-X (HuggingFace, public)
# ──────────────────────────────────────────────────────────────────────────────

step_1_smplerx() {
    log "1. SMPLer-X — poids principaux (HuggingFace)…"
    local dir="$MODELS_DIR/smplerx"
    mkdir -p "$dir"

    # Recommandé : H32* (correct), 662M params, ~2.6 GB
    download \
        "https://huggingface.co/caizhongang/SMPLer-X/resolve/main/smpler_x_h32_correct.pth.tar?download=true" \
        "$dir/smpler_x_h32_correct.pth.tar"

    # Variants alternatifs — décommenter si tu veux les avoir
    # download "https://huggingface.co/caizhongang/SMPLer-X/resolve/main/smpler_x_l32.pth.tar?download=true" "$dir/smpler_x_l32.pth.tar"
    # download "https://huggingface.co/caizhongang/SMPLer-X/resolve/main/smpler_x_b32.pth.tar?download=true" "$dir/smpler_x_b32.pth.tar"
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. mmdet Faster R-CNN R50 (OpenMMLab, public)
# ──────────────────────────────────────────────────────────────────────────────

step_2_mmdet() {
    log "2. mmdet Faster R-CNN R50 (détecteur de personnes)…"
    local dir="$MODELS_DIR/mmdet"
    mkdir -p "$dir"

    download \
        "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth" \
        "$dir/faster_rcnn_r50_fpn_1x_coco.pth"
}

# ──────────────────────────────────────────────────────────────────────────────
# 3. HaMeR : ViTDet detector + checkpoint principal
# ──────────────────────────────────────────────────────────────────────────────

step_3_hamer() {
    log "3. HaMeR — ViTDet + demo data…"
    local dir="$MODELS_DIR/hamer"
    mkdir -p "$dir"

    download \
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl" \
        "$dir/model_final_f05665.pkl"

    # HaMeR demo_data : Google Drive (gdown) + fallback HTTP
    if [[ ! -f "$dir/_DATA/data/checkpoints/hamer.ckpt" ]]; then
        log "  Téléchargement HaMeR demo data (~1-2 GB)…"
        local tarball="$dir/hamer_demo_data.tar.gz"
        if ! wget --progress=bar:force \
                "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz" \
                -O "$tarball.tmp"; then
            rm -f "$tarball.tmp"
            fail "Téléchargement HaMeR demo data échoué"
        fi
        mv "$tarball.tmp" "$tarball"
        log "  Extraction…"
        tar -xzf "$tarball" -C "$dir"
        rm -f "$tarball"
        ok "  HaMeR demo data extrait"
    else
        ok "  HaMeR demo data déjà présent"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# 4. SMPL-X / SMPL / MANO / FLAME — REGISTRATION MPI REQUISE
# ──────────────────────────────────────────────────────────────────────────────

step_4_mpi_models() {
    log "4. Modèles MPI (SMPL-X, SMPL, MANO, FLAME)…"

    require_manual \
        "SMPL-X body models" \
        "$MODELS_DIR/smplx/SMPLX_NEUTRAL.npz" \
        "https://smpl-x.is.tue.mpg.de" \
        "Modèles corps SMPL-X (NEUTRAL, MALE, FEMALE) + fichiers d'extension :
                  SMPLX_NEUTRAL.npz, SMPLX_MALE.npz, SMPLX_FEMALE.npz,
                  MANO_SMPLX_vertex_ids.pkl, SMPL-X__FLAME_vertex_ids.npy,
                  SMPLX_to_J14.pkl"

    require_manual \
        "SMPL body models" \
        "$MODELS_DIR/smpl/SMPL_NEUTRAL.pkl" \
        "https://smpl.is.tue.mpg.de" \
        "Modèles corps SMPL (NEUTRAL, MALE, FEMALE) :
                  SMPL_NEUTRAL.pkl, SMPL_MALE.pkl, SMPL_FEMALE.pkl"

    require_manual \
        "MANO right hand model" \
        "$MODELS_DIR/mano/MANO_RIGHT.pkl" \
        "https://mano.is.tue.mpg.de" \
        "Modèle main MANO. HaMeR n'a besoin QUE de la main droite (gauche miroitée) :
                  MANO_RIGHT.pkl"

    require_manual \
        "FLAME assets" \
        "$MODELS_DIR/flame/FLAME.pkl" \
        "https://flame.is.tue.mpg.de" \
        "Modèle visage FLAME. Si tu télécharges via le bundle EMOCA (étape 5),
                  ce dossier sera créé automatiquement et tu peux ignorer cet item."
}

# ──────────────────────────────────────────────────────────────────────────────
# 5. EMOCA assets bundle — REGISTRATION MPI REQUISE
# ──────────────────────────────────────────────────────────────────────────────

step_5_emoca() {
    log "5. EMOCA v2 + assets (bundle MPI)…"

    local dir="$MODELS_DIR/emoca/assets"
    mkdir -p "$dir"

    cat <<EOF

${YELLOW}═══ EMOCA assets — TÉLÉCHARGEMENT MANUEL VIA SCRIPT EMOCA ═══${NC}

  EMOCA fournit son propre script d'auto-téléchargement qui prompte pour
  l'acceptation de licence. Procédure :

    cd pipeline/envs/emoca/repo
    bash gdl_apps/EMOCA/demos/download_assets.sh

  Le script demandera confirmation pour FLAME et EMOCA, puis téléchargera :
    - EMOCA_v2_lr_mse_20.zip  (variant recommandé)
    - DECA.zip + DECA assets
    - FaceRecognition.zip
    - FLAME.zip
    - EMOCA_test_example_data.zip

  Une fois fait, déplace tout vers $dir :

    mv pipeline/envs/emoca/repo/assets/* $dir/

EOF

    if [[ -d "$dir/EMOCA/models/EMOCA_v2_lr_mse_20" ]]; then
        ok "  EMOCA v2 présent"
    else
        MANUAL_PENDING=$((MANUAL_PENDING + 1))
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# 6. Génération CHECKSUMS.sha256
# ──────────────────────────────────────────────────────────────────────────────

step_6_checksums() {
    log "6. Calcul des checksums SHA-256…"
    local checksums="$MODELS_DIR/CHECKSUMS.sha256"

    # On (re)génère ; en prod on devrait versionner ce fichier.
    (cd "$MODELS_DIR" && find . -type f \
        -not -name "*.tmp" \
        -not -name "README.md" \
        -not -name "CHECKSUMS.sha256" \
        -exec sha256sum {} \;) > "$checksums"

    local n=$(wc -l < "$checksums")
    ok "  $n fichiers répertoriés dans $checksums"
}

# ──────────────────────────────────────────────────────────────────────────────
# Récap
# ──────────────────────────────────────────────────────────────────────────────

step_summary() {
    if [[ "$MANUAL_PENDING" -gt 0 ]]; then
        cat <<EOF

${YELLOW}╔════════════════════════════════════════════════════════════════╗
║          $MANUAL_PENDING ÉLÉMENT(S) MANUEL(S) RESTANT(S)                  ║
╚════════════════════════════════════════════════════════════════╝${NC}

Téléchargements automatiques effectués. Pour terminer :
  1. Suivre les instructions ${YELLOW}[manuel]${NC} ci-dessus
  2. Relancer ce script (il repère les fichiers déjà présents)
  3. Relancer ${BLUE}python scripts/verify_env.py${NC} pour valider

EOF
        exit 1
    else
        cat <<EOF

${GREEN}╔════════════════════════════════════════════════════════════════╗
║          TOUS LES POIDS PRÉSENTS                               ║
╚════════════════════════════════════════════════════════════════╝${NC}

Lance maintenant :
    ${BLUE}python scripts/verify_env.py${NC}

EOF
    fi
}

main() {
    step_1_smplerx
    step_2_mmdet
    step_3_hamer
    step_4_mpi_models
    step_5_emoca
    step_6_checksums
    step_summary
}

main "$@"
