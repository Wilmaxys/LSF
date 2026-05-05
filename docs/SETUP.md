# Installation — Ubuntu 22.04 vierge → premier run

Guide pas-à-pas pour installer le pipeline LSF sur un serveur Linux **headless**
(sans GUI) avec GPU NVIDIA.

## Pré-requis matériels

- Ubuntu 22.04 LTS (ou Fedora 40+, à adapter)
- GPU NVIDIA Ampere ou plus récent (RTX 30xx, A4000+, A5000+, A4500 recommandé)
- 16 Go de VRAM minimum, 20 Go conseillé
- 50 Go d'espace disque libre minimum (modèles + envs ML + Blender + repos)
- Connexion internet (téléchargement de ~15 Go pendant l'install)

## Pré-requis logiciels

- Driver NVIDIA ≥ 525 (pour CUDA 12.x). Vérifier avec `nvidia-smi`.
- Accès `sudo`
- `git` installé
- Comptes créés sur les sites MPI (étape 0 ci-dessous)

## Étape 0 — Comptes MPI (gratuits, ~5 min)

Avant tout, créer un compte gratuit sur chacun de ces 5 sites et **accepter
la licence** (lecture rapide ; cocher la case "non-commercial use only") :

1. https://smpl-x.is.tue.mpg.de
2. https://smpl.is.tue.mpg.de
3. https://mano.is.tue.mpg.de
4. https://flame.is.tue.mpg.de
5. https://emoca.is.tue.mpg.de

Garder les credentials à portée — tu en auras besoin à l'étape 4.

## Étape 1 — Cloner le repo

```bash
git clone <url-du-repo> lsf-pipeline
cd lsf-pipeline
```

## Étape 2 — Vérifier le GPU et le driver

```bash
nvidia-smi
```

Tu dois voir ton GPU listé avec un `Driver Version: 525.x` ou supérieur. Si
le driver est plus ancien :

```bash
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

## Étape 3 — Lancer setup.sh

```bash
bash scripts/setup.sh
```

Durée totale : **~30 minutes** sur connexion fibre, dont :
- ~5 min pour les paquets système (apt)
- ~3 min pour Miniforge
- ~5 min pour Blender + addon VRM
- ~10 min pour les 3 environnements ML (le build de PyTorch3D pour EMOCA est le
  plus long)
- ~5 min pour le clone des repos ML

À la fin, tu auras :
- `$HOME/miniforge3/` — Miniforge avec 4 envs conda :
  - `lsf-orchestrator` (Python 3.10) — pour lancer le pipeline
  - `lsf-smplerx` (Python 3.8) — extraction corps
  - `lsf-hamer` (Python 3.10) — raffinement mains
  - `lsf-emoca` (Python 3.8) — raffinement visage
- `lsf-pipeline/.tools/blender/blender` — Blender 4.5 LTS headless
- `lsf-pipeline/pipeline/envs/{name}/repo/` — repos ML clonés aux commits pinned

### En cas d'erreur

Le script s'arrête à la première erreur (`set -e`). Erreurs courantes :

- **Driver NVIDIA trop ancien** → `nvidia-driver-535` puis reboot.
- **`pytorch3d` build échoue** → manque `gcc-9`/`gcc-10` ou `nvcc` (CUDA toolkit).
  Cf. [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md).
- **Disque plein** → libérer ≥ 50 Go.
- **Blender headless KO au smoke test** → manque `libegl1` ou `libgl1` ; le
  script tente de les installer via apt.

## Étape 4 — Télécharger les poids

```bash
bash scripts/download_weights.sh
```

Ce script :
- Télécharge automatiquement ce qui est public (HuggingFace, OpenMMLab, Facebook AI)
- T'imprime les **instructions exactes** pour les poids requérant un compte MPI

Pour les poids MPI :

```bash
# 1. Télécharger SMPL-X depuis https://smpl-x.is.tue.mpg.de (section Downloads)
# 2. Décompresser et déplacer vers pipeline/models/smplx/
# 3. Idem pour SMPL, MANO, FLAME (cf. docs/MODELS.md)

# Pour EMOCA, lance le script auto fourni avec le repo :
cd pipeline/envs/emoca/repo
bash gdl_apps/EMOCA/demos/download_assets.sh

# Le script demande confirmation pour FLAME et EMOCA puis télécharge tout.
mv pipeline/envs/emoca/repo/assets/* pipeline/models/emoca/assets/

cd ../../../..  # retour à la racine du repo
```

Une fois tout en place, relance `download_weights.sh` — il vérifie que tout
est présent et génère un fichier `pipeline/models/CHECKSUMS.sha256`.

## Étape 5 — Vérifier l'environnement

```bash
conda activate lsf-orchestrator
python scripts/verify_env.py
```

Le script affiche un rapport coloré avec ✓ / ⚠ / ✗ pour chaque check :

- Python ≥ 3.10
- nvidia-smi accessible, GPU détecté, VRAM suffisante
- Imports orchestrateur OK
- Tous les fichiers de poids présents avec tailles plausibles
- Checksums SHA-256 OK
- Chaque env ML peut importer torch + voir CUDA
- Blender headless + addon VRM OK

S'il y a des **✗ erreurs** : corriger avant d'aller plus loin.
S'il y a juste des **⚠ warnings** (ex : VRAM légèrement basse) : tu peux
continuer mais certaines vidéos longues peuvent OOM.

Pour valider aussi un avatar VRM de référence :

```bash
python scripts/verify_env.py --vrm /chemin/vers/mon_avatar.vrm
```

## Étape 6 — Premier run

Place une vidéo et un avatar dans le dossier `data/` :

```bash
mkdir -p data/avatars
cp /chemin/vers/mon_avatar.vrm data/avatars/alicia.vrm
cp /chemin/vers/clip.mp4 data/input/clip.mp4
```

Puis lance :

```bash
conda activate lsf-orchestrator
python pipeline/pipeline.py \
    --video data/input/clip.mp4 \
    --avatar data/avatars/alicia.vrm \
    --output data/output/clip.vrma
```

Pour une vidéo de 60 s à 30 fps (1800 frames), prévoir **15-25 minutes** de
processing sur RTX A4500 (cf. [PIPELINE.md](PIPELINE.md) §10.4).

### Validation sans modèles (`--dry-run`)

```bash
python pipeline/pipeline.py --video X --avatar Y --output Z --dry-run
```

Vérifie les chemins, l'avatar VRM et la disponibilité des envs sans rien
exécuter de coûteux. Utile pour tester rapidement la config.

### Vidéo de debug avec mesh superposé

```bash
python pipeline/pipeline.py ... --debug-overlay
```

Produit en plus un fichier `clip.debug.mp4` avec :
- Le numéro de frame et la confidence par région
- Une bordure rouge sur les frames sous le seuil de confidence

## Étape 7 — Récupérer le résultat sur ton Mac

```bash
# Depuis ton Mac
scp user@serveur:lsf-pipeline/data/output/clip.vrma ./viewer/public/animations/
```

## Étape 8 — Lancer le viewer (sur ton Mac)

Une seule fois :
```bash
cd lsf-pipeline/viewer
npm install
```

Puis à chaque fois :
```bash
npm run dev
# Ouvre http://localhost:5173/?animation=animations/clip.vrma
```

Tu peux aussi déposer un `.vrm` ou `.vrma` directement dans la page par
drag & drop.

## Mises à jour du pipeline

Pour ré-installer les dépendances après un `git pull` :

```bash
# Refaire l'install des deps (sans re-télécharger Blender ni les modèles)
bash scripts/setup.sh   # idempotent : skippe ce qui est déjà installé
python scripts/verify_env.py
```

Pour forcer un re-clone des repos ML :

```bash
rm -rf pipeline/envs/*/repo
bash scripts/setup.sh
```
