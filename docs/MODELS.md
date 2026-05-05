# Modèles ML — licences, sources, comptes à créer

Le pipeline utilise 5 modèles ML, chacun sous licence stricte. **Tous sont
non-commerciaux.** Ce document liste l'origine, la licence et la procédure
de téléchargement de chacun.

## ⚠️ Avertissement licence

Tant que ton projet utilise l'un de ces modèles, **tu ne peux pas
le commercialiser**. Pour un usage commercial, il faut soit :
- contacter chaque labo pour négocier une licence commerciale ;
- remplacer chaque modèle par une alternative permissive (lourd refactor).

## Tableau récapitulatif

| # | Modèle | Auteur | Licence | Compte requis | Auto via script ? |
|---|---|---|---|---|---|
| 1 | SMPLer-X | MotrixLab / S-Lab | S-Lab License 1.0 (non-commercial) | Non | Oui (HuggingFace) |
| 2 | mmdet Faster R-CNN | OpenMMLab | Apache 2.0 | Non | Oui |
| 3 | HaMeR | UC Regents (UT Austin) | MIT (code) | Non pour le code | Oui |
| 4 | ViTDet | Facebook AI | Apache 2.0 | Non | Oui |
| 5 | SMPL-X (modèle corps) | MPI-IS | MPI non-commercial | Oui | Non — manuel |
| 6 | SMPL (modèle corps) | MPI-IS | MPI non-commercial | Oui | Non — manuel |
| 7 | MANO (modèle main) | MPI-IS | MPI non-commercial | Oui | Non — manuel |
| 8 | FLAME (modèle visage) | MPI-IS | MPI non-commercial | Oui | Non — manuel |
| 9 | EMOCA v2 | MPI-IS | MPI non-commercial | Oui | Semi-auto (script EMOCA) |

## Détail par modèle

### 1. SMPLer-X

- **Repo** : https://github.com/MotrixLab/SMPLer-X
- **Licence** : S-Lab License 1.0 — https://raw.githubusercontent.com/MotrixLab/SMPLer-X/main/LICENSE
  - Citation : *"Redistribution and use for non-commercial purpose in source and binary forms ... are permitted ..."*
  - Hard blocker pour usage commercial.
- **Variants disponibles** :
  - `smpler_x_s32` (32M params, ~130 MB)
  - `smpler_x_b32` (103M, ~410 MB)
  - `smpler_x_l32` (327M, ~1.3 GB)
  - `smpler_x_h32` (662M, ~2.6 GB) — **buggy, ne pas utiliser**
  - `smpler_x_h32_correct` (662M, ~2.6 GB) — **recommandé**, fix du bug 3DPW
- **URL** : https://huggingface.co/caizhongang/SMPLer-X
- **Compte** : non requis. Téléchargé par `download_weights.sh`.

### 2. mmdet Faster R-CNN R50-FPN COCO

- **Auteur** : OpenMMLab — https://github.com/open-mmlab/mmdetection
- **Licence** : Apache 2.0
- **URL** : https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/
- **Taille** : ~160 MB
- **Compte** : non requis. Téléchargé par `download_weights.sh`.

### 3. HaMeR (code)

- **Repo** : https://github.com/geopavlakos/hamer
- **Licence** : MIT — https://raw.githubusercontent.com/geopavlakos/hamer/main/LICENSE.md
  - Le **code** est MIT.
  - **Mais** HaMeR dépend du modèle MANO (#7) qui est non-commercial.
- **Poids** : auto-téléchargés par `bash fetch_demo_data.sh` (UT Austin Pavlakos),
  ~1-2 GB.
- **Compte** : non requis pour les poids du modèle HaMeR lui-même ;
  **requis** pour MANO (cf. #7).

### 4. ViTDet (cascade_mask_rcnn_vitdet_h)

- **Auteur** : Facebook AI Research
- **Licence** : Apache 2.0
- **URL** : https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl
- **Taille** : ~2.6 GB
- **Compte** : non requis. Téléchargé par `download_weights.sh`.

### 5. SMPL-X (modèle corps)

- **Auteur** : Max Planck Institute for Intelligent Systems
- **Licence** : MPI non-commercial — https://smpl-x.is.tue.mpg.de/modellicense
- **URL** : https://smpl-x.is.tue.mpg.de (section Downloads après login)
- **Fichiers requis** :
  - `SMPLX_NEUTRAL.npz`
  - `SMPLX_MALE.npz`
  - `SMPLX_FEMALE.npz`
  - `MANO_SMPLX_vertex_ids.pkl` (extension)
  - `SMPL-X__FLAME_vertex_ids.npy` (extension)
  - `SMPLX_to_J14.pkl` (extension)
- **Taille totale** : ~150 MB
- **Compte requis** : oui. Créer sur https://smpl-x.is.tue.mpg.de
- **Cible** : `pipeline/models/smplx/`

### 6. SMPL (modèle corps original)

- **Auteur** : Max Planck Institute for Intelligent Systems
- **Licence** : MPI non-commercial — https://smpl.is.tue.mpg.de
- **Fichiers requis** :
  - `SMPL_NEUTRAL.pkl`
  - `SMPL_MALE.pkl`
  - `SMPL_FEMALE.pkl`
- **Compte requis** : oui. Créer sur https://smpl.is.tue.mpg.de
- **Cible** : `pipeline/models/smpl/`

### 7. MANO (modèle main)

- **Auteur** : Max Planck Institute for Intelligent Systems
- **Licence** : MPI non-commercial — https://mano.is.tue.mpg.de
- **Fichier requis** : `MANO_RIGHT.pkl` (HaMeR n'a besoin que du droit ; le
  gauche est miroité automatiquement)
- **Taille** : ~5 MB
- **Compte requis** : oui. Créer sur https://mano.is.tue.mpg.de
- **Cible** : `pipeline/models/mano/MANO_RIGHT.pkl`

### 8. FLAME (modèle visage)

- **Auteur** : Max Planck Institute for Intelligent Systems
- **Licence** : MPI non-commercial — https://flame.is.tue.mpg.de
- **Fichiers requis** : modèle FLAME v2020 + textures + masks
- **Compte requis** : oui. Créer sur https://flame.is.tue.mpg.de
- **Cible** : `pipeline/models/flame/`
- **Note** : EMOCA fournit un script auto (étape 9) qui télécharge aussi FLAME.

### 9. EMOCA v2 + assets

- **Repo** : https://github.com/radekd91/emoca (branche `release/EMOCA_v2`)
- **Licence** : MPI non-commercial — https://emoca.is.tue.mpg.de/license.html
- **Compte requis** : oui (utilise les credentials FLAME + EMOCA).
  Créer sur https://emoca.is.tue.mpg.de
- **Procédure** : `setup.sh` clone le repo. Une fois les comptes créés,
  exécuter le script auto fourni :

  ```bash
  cd pipeline/envs/emoca/repo
  bash gdl_apps/EMOCA/demos/download_assets.sh
  ```

  Le script prompte pour confirmer FLAME et EMOCA puis télécharge :
  - `EMOCA_v2_lr_mse_20.zip` — variant recommandé (lip-reading aware)
  - `DECA.zip` + assets DECA
  - `FaceRecognition.zip`
  - `FLAME.zip`
  - `EMOCA_test_example_data.zip`
- **Cible** : déplacer le contenu de `pipeline/envs/emoca/repo/assets/` vers
  `pipeline/models/emoca/assets/`.

## Vérification post-téléchargement

```bash
conda activate lsf-orchestrator
python scripts/verify_env.py
```

Le script vérifie présence + tailles plausibles + checksums SHA-256.

## Note sur la dépréciation d'EMOCA v2

EMOCA v2 est officiellement **déprécié** par son auteur (cf. README du repo).
L'auteur redirige les utilisateurs vers son successeur **inferno** :
https://github.com/radekd91/inferno

Pour la v1 de notre pipeline, nous gardons EMOCA v2 (conforme au cahier des
charges). Une migration vers inferno reste à explorer en v2 si la maintenance
d'EMOCA devient pénible (URLs MPI qui changent, bugs non corrigés, etc.).
