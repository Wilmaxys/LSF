#!/usr/bin/env bash
#
# clean.sh — désinstalle ce que setup.sh + download_weights.sh ont installé.
#
# Interactif : prompte oui/non par catégorie pour éviter de tout perdre par
# accident. Les poids MPI (30+ GB) demandent une re-auth pour être re-downloadés.
#
# Usage :
#   bash scripts/clean.sh              # interactif
#   bash scripts/clean.sh --all        # tout supprimer sans prompt (NUCLÉAIRE)
#   bash scripts/clean.sh --dry-run    # liste ce qui serait supprimé

set -uo pipefail   # pas de -e : on veut continuer même si certaines suppressions échouent

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${BLUE}[clean]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
red()  { echo -e "${RED}[!]${NC} $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="${HOME}/miniforge3"

DRY_RUN=0
ALL=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --all)     ALL=1 ;;
        -h|--help)
            grep '^#' "$0" | head -20
            exit 0
            ;;
        *) red "Argument inconnu : $arg" ; exit 1 ;;
    esac
done

# confirm "Question" returns 0 if user says yes
confirm() {
    if [[ "$ALL" -eq 1 ]]; then return 0; fi
    local prompt="$1"
    read -r -p "$(echo -e "${YELLOW}?${NC} $prompt [y/N] ")" reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "    [dry-run] $*"
    else
        eval "$@"
    fi
}

size_of() {
    [[ -e "$1" ]] && du -sh "$1" 2>/dev/null | cut -f1 || echo "absent"
}

echo "════════════════════════════════════════════════════════════"
echo "  LSF clean script"
echo "  Repo : $REPO_ROOT"
[[ "$DRY_RUN" -eq 1 ]] && warn "MODE DRY-RUN — rien ne sera supprimé"
[[ "$ALL" -eq 1 ]] && red "MODE --all — tout sera supprimé sans prompt !"
echo "════════════════════════════════════════════════════════════"
echo

# ──────────────────────────────────────────────────────────────────────────────
# 1. Envs conda lsf-* (quick à recréer)
# ──────────────────────────────────────────────────────────────────────────────
log "1. Environnements conda lsf-* (~5-10 GB chacun)"
if [[ -x "$CONDA_DIR/bin/conda" ]]; then
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    envs=$(conda env list | awk '{print $1}' | grep -E '^lsf-' || true)
    if [[ -n "$envs" ]]; then
        echo "  Trouvés :"
        echo "$envs" | sed 's/^/    /'
        if confirm "Supprimer ces 4 envs conda ?"; then
            for env in $envs; do
                run "conda env remove -y -n \"$env\""
            done
            ok "  Envs conda supprimés"
        fi
    else
        ok "  Aucun env lsf-* trouvé"
    fi
else
    warn "  Miniforge introuvable à $CONDA_DIR — skip"
fi
echo

# ──────────────────────────────────────────────────────────────────────────────
# 2. Repos ML clonés
# ──────────────────────────────────────────────────────────────────────────────
log "2. Repos ML clonés (SMPLer-X, HaMeR, EMOCA)"
repos=()
for env in smplerx hamer emoca; do
    d="$REPO_ROOT/pipeline/envs/$env/repo"
    [[ -d "$d" ]] && repos+=("$d ($(size_of "$d"))")
done
if [[ ${#repos[@]} -gt 0 ]]; then
    echo "  Trouvés :"
    printf '    %s\n' "${repos[@]}"
    if confirm "Supprimer ces repos ?"; then
        for env in smplerx hamer emoca; do
            run "rm -rf \"$REPO_ROOT/pipeline/envs/$env/repo\""
        done
        ok "  Repos ML supprimés"
    fi
else
    ok "  Aucun repo cloné trouvé"
fi
echo

# ──────────────────────────────────────────────────────────────────────────────
# 3. Blender + VRM addon
# ──────────────────────────────────────────────────────────────────────────────
log "3. Blender (.tools/blender) + VRM addon utilisateur"
blender_dir="$REPO_ROOT/.tools"
vrm_addon_dir="$HOME/.config/blender/4.5/scripts/addons/VRM_Addon_for_Blender-release"
items=()
[[ -d "$blender_dir" ]] && items+=("$blender_dir ($(size_of "$blender_dir"))")
[[ -d "$vrm_addon_dir" ]] && items+=("$vrm_addon_dir ($(size_of "$vrm_addon_dir"))")
if [[ ${#items[@]} -gt 0 ]]; then
    echo "  Trouvés :"
    printf '    %s\n' "${items[@]}"
    if confirm "Supprimer Blender + VRM addon ?"; then
        run "rm -rf \"$blender_dir\""
        run "rm -rf \"$vrm_addon_dir\""
        ok "  Blender + VRM supprimés"
    fi
else
    ok "  Blender absent"
fi
echo

# ──────────────────────────────────────────────────────────────────────────────
# 4. Poids modèles (LE GROS — 30+ GB, MPI auth pour re-DL)
# ──────────────────────────────────────────────────────────────────────────────
log "4. Poids modèles téléchargés (pipeline/models/)"
models_dir="$REPO_ROOT/pipeline/models"
if [[ -d "$models_dir" ]]; then
    total=$(du -sh "$models_dir" 2>/dev/null | cut -f1)
    echo "  Taille totale : $total"
    echo "  Détail :"
    du -sh "$models_dir"/*/ 2>/dev/null | sed 's/^/    /' || true
    red "  ATTENTION : MPI (SMPL-X, SMPL, MANO, FLAME, EMOCA) demanderont une re-auth + relicence."
    if confirm "Supprimer tous les poids modèles ?"; then
        # Préserve README et CHECKSUMS si présents
        run "find \"$models_dir\" -mindepth 1 -maxdepth 1 -not -name 'README.md' -exec rm -rf {} +"
        ok "  Poids supprimés"
    fi
else
    ok "  Dossier models/ absent"
fi
echo

# ──────────────────────────────────────────────────────────────────────────────
# 5. Caches temporaires
# ──────────────────────────────────────────────────────────────────────────────
log "5. Fichiers temporaires (cookies MPI, .error.html, tarballs partiels)"
tmp_items=()
ls /tmp/lsf_mpi_*.cookies 2>/dev/null && tmp_items+=("/tmp/lsf_mpi_*.cookies")
find "$REPO_ROOT/pipeline/models" -name "*.error.html" 2>/dev/null | head -5 | while read -r f; do
    echo "    $f"
done
if confirm "Nettoyer les fichiers temporaires ?"; then
    run "rm -f /tmp/lsf_mpi_*.cookies"
    run "find \"$REPO_ROOT/pipeline/models\" -name '*.error.html' -delete 2>/dev/null"
    run "find \"$REPO_ROOT/pipeline/models\" -name '*.tmp' -delete 2>/dev/null"
    ok "  Temp nettoyés"
fi
echo

# ──────────────────────────────────────────────────────────────────────────────
# 6. NUCLÉAIRE : Miniforge entier (~5 GB)
# ──────────────────────────────────────────────────────────────────────────────
log "6. Miniforge ENTIER ($CONDA_DIR)"
if [[ -d "$CONDA_DIR" ]]; then
    echo "  Taille : $(size_of "$CONDA_DIR")"
    red "  ATTENTION : ça supprime conda/mamba complètement (utile pour AUTRES projets aussi)."
    if confirm "Supprimer Miniforge ?"; then
        run "rm -rf \"$CONDA_DIR\""
        warn "  Miniforge supprimé. Pense à retirer le bloc 'conda init' de ~/.bashrc"
        warn "  (commande : sed -i '/>>> conda initialize >>>/,/<<< conda initialize <<</d' ~/.bashrc)"
    fi
else
    ok "  Miniforge absent"
fi
echo

# ──────────────────────────────────────────────────────────────────────────────
# Récap
# ──────────────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
ok "Cleanup terminé"
if [[ "$DRY_RUN" -eq 1 ]]; then
    warn "C'était un DRY-RUN — rien n'a réellement été supprimé"
fi
echo "════════════════════════════════════════════════════════════"
