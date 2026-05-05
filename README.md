# LSF Pipeline — Vidéo → Animation 3D

Pipeline 100 % offline et automatisé qui transforme une vidéo de Langue des Signes
Française en une animation 3D rejouable sur un avatar VRM dans un viewer web.

```
Vidéo .mp4  +  Avatar .vrm  →  [Pipeline ML]  →  Animation .vrma
                                                       ↓
                                            Viewer Three.js (web)
```

## Architecture matérielle

Le projet est conçu pour fonctionner sur deux machines distinctes :

| Machine | Rôle | Outils installés |
|---|---|---|
| **Serveur Linux GPU** (sans GUI) | Pipeline ML : extrait l'animation à partir de la vidéo | Python, CUDA, Blender headless |
| **Ton Mac** (ou autre poste) | Visualisation : affiche l'animation dans un navigateur | Node.js + viewer Three.js |

Tu n'as **pas besoin de GPU sur ton Mac**. Le serveur produit un petit fichier
`.vrma` (quelques Mo), tu le copies sur ton Mac, et tu le visualises localement.

## Quickstart

### Sur le serveur Linux GPU (Ubuntu 22.04)

```bash
# 1. Cloner le repo
git clone <url-du-repo> lsf-pipeline && cd lsf-pipeline

# 2. Installer le pipeline ML (~30 min, télécharge Blender, conda, etc.)
bash scripts/setup.sh

# 3. Télécharger les poids ML (créer d'abord les comptes MPI — cf. docs/MODELS.md)
bash scripts/download_weights.sh

# 4. Lancer le pipeline sur une vidéo
python pipeline/pipeline.py \
    --video data/input/clip_lsf.mp4 \
    --avatar data/avatars/mon_avatar.vrm \
    --output data/output/clip_lsf.vrma
```

Le serveur produit `data/output/clip_lsf.vrma`.

### Sur ton Mac

```bash
# 5. Récupérer l'animation depuis le serveur
scp user@serveur:lsf-pipeline/data/output/clip_lsf.vrma ./viewer/public/animations/

# 6. (Une seule fois) cloner le repo et installer le viewer
git clone <url-du-repo> lsf-pipeline && cd lsf-pipeline/viewer
npm install

# 7. Lancer le viewer
npm run dev
# Ouvre http://localhost:5173/?animation=animations/clip_lsf.vrma
```

Tu peux aussi déposer un `.vrm` (avatar) ou un `.vrma` (animation) directement
dans la page par drag & drop pour tester sans modifier l'URL.

## Documentation

- [docs/SETUP.md](docs/SETUP.md) — installation pas-à-pas sur Ubuntu 22.04 vierge
- [docs/PIPELINE.md](docs/PIPELINE.md) — spécification technique complète
- [docs/MODELS.md](docs/MODELS.md) — modèles ML, licences, comptes à créer
- [docs/AVATARS.md](docs/AVATARS.md) — comment utiliser un avatar VRM personnalisé
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — erreurs courantes

## Structure

```
lsf-pipeline/
├── pipeline/           # Pipeline ML Python (3 envs séparés + orchestrateur)
├── viewer/             # Viewer Three.js / TypeScript
├── scripts/            # Scripts d'installation et de vérification
├── data/               # Vidéos d'entrée et animations de sortie (gitignored)
├── docs/               # Documentation complète
└── config.yaml         # Configuration centrale
```

## Licences — IMPORTANT

Tous les modèles ML utilisés (SMPL-X, MANO, FLAME, EMOCA, SMPLer-X) ont des
**licences strictement non-commerciales**. Ce projet et ses sorties ne peuvent
pas être utilisés commercialement tant qu'ils intègrent l'un de ces modèles.

Voir [docs/MODELS.md](docs/MODELS.md) pour le détail.

## Composants

- **SMPLer-X** — pose et forme du corps (https://github.com/MotrixLab/SMPLer-X)
- **HaMeR** — raffinement des mains (https://github.com/geopavlakos/hamer)
- **EMOCA v2** — raffinement du visage (https://github.com/radekd91/emoca)
- **VRM_Addon_for_Blender** — retargeting vers VRM (https://github.com/saturday06/VRM-Addon-for-Blender)
- **@pixiv/three-vrm** — affichage VRM dans Three.js (https://github.com/pixiv/three-vrm)
