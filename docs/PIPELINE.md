# Pipeline LSF Vidéo → Animation 3D — Spécification technique

> Document de référence pour la Phase 1. Toutes les versions, formats et mappings ci-dessous sont les fondations sur lesquelles le code de la Phase 2 sera écrit. Les points d'incertitude sont **explicitement flaggés** ; ne rien inventer ne pas explicitement marqué comme tel.
>
> **Date de rédaction** : 2026-05-05.
> **Cible runtime** : Ubuntu 22.04 (ou Fedora 40+), NVIDIA RTX A4500 (Ampere, sm_86, 20 GB VRAM).
> **Cible dev** : Intel Mac sans GPU — code seulement, aucune exécution ML.

---

## 0. Décisions d'architecture importantes (à valider avant la Phase 2)

Ces points changent la forme du repo. Je les liste en tête car ils impactent la structure que tu as proposée.

### 0.1 Trois environnements Python isolés, pas un seul

Les trois modèles (SMPLer-X, HaMeR, EMOCA v2) ont des dépendances mutuellement incompatibles. Il est **impossible** de les faire cohabiter dans un seul venv :

| Repo | Python | PyTorch | CUDA wheels | Blocker dur |
|---|---|---|---|---|
| SMPLer-X | 3.8 | 1.12.0 | cu113 | mmcv-full 1.7.1 + mmdet 2.26.0 (cassés sur torch 2.x) |
| HaMeR | 3.10 | ≤1.13 (recommandé cu117) | cu117 | `mmcv==1.3.9` hard pin (cassé sur torch 2.x) |
| EMOCA v2 | 3.8 | 1.12.1 | cu113 | pytorch3d 0.6.2 (ne build pas sur torch 2.x) |

**Conséquence sur la structure du repo** : la Phase 2 doit produire **trois** sous-arborescences `pipeline/envs/{smplerx,hamer,emoca}/` chacune avec son `requirements.txt`, son `requirements-lock.txt` et son script d'activation. L'orchestrateur `pipeline/pipeline.py` invoque chaque étape comme un sous-process avec l'env approprié, et passe les données via des fichiers `.npz` intermédiaires sur disque.

> Alternative à explorer plus tard : remplacer EMOCA v2 par son successeur **inferno** (https://github.com/radekd91/inferno) que l'auteur recommande désormais ; c'est le même type de modèle FLAME-régresseur, potentiellement compatible torch 2.x. **Non vérifié pour cette phase** — à investiguer si la maintenance triple-env devient pénible.

### 0.2 Format de sortie animation : `.vrma` plutôt que `.glb` baké

Tu as spécifié dans le brief « Export `.glb` animation bakée ». L'écosystème VRM officiel a depuis 2024 un format dédié, **VRM Animation (`.vrma`)**, qui :
- est un glTF avec l'extension `VRMC_vrm_animation-1.0` ;
- stocke les pistes d'animation **déjà retargettées sur les bones humanoïdes VRM nommés** (`hips`, `spine`, `leftUpperArm`, …) plutôt que sur des nœuds glTF arbitraires ;
- inclut nativement les pistes d'expressions VRM et de `lookAt` ;
- est chargé et appliqué à n'importe quel avatar VRM par `@pixiv/three-vrm-animation` via `createVRMAnimationClip(vrmAnimation, vrm)` qui fait le retargeting au runtime ;
- est exporté par `VRM_Addon_for_Blender` via `File → Export → VRM Animation (.vrma)`.

**Recommandation** : produire des `.vrma` à la place des `.glb` bakés. Cela colle exactement au cas d'usage « avatar paramétrable » du brief (l'animation est portable d'un VRM à l'autre par construction). Le viewer Three.js charge un `.vrma` exactement comme un `.glb` (c'est un fichier glTF binaire valide).

Si tu préfères garder l'extension `.glb` pour ne pas dépendre d'un format moins répandu, on peut simplement renommer le fichier en sortie : un `.vrma` reste un glTF binaire chargeable par n'importe quel `GLTFLoader`. Le plugin `VRMAnimationLoaderPlugin` est ce qui décode l'extension.

**Décision à valider** : `.vrma` (recommandé) ou `.glb` baké. Le reste du document suppose `.vrma`.

### 0.3 Toutes les licences ML sont **non-commerciales**

Les cinq éléments ML du pipeline ont des licences non-commerciales strictes :

| Composant | Licence | Source |
|---|---|---|
| SMPLer-X | S-Lab License 1.0 (non-commercial) | https://raw.githubusercontent.com/MotrixLab/SMPLer-X/main/LICENSE |
| Modèle SMPL-X | MPI non-commercial | https://smpl-x.is.tue.mpg.de/modellicense |
| Modèle MANO | MPI non-commercial | https://mano.is.tue.mpg.de |
| Modèle FLAME | MPI non-commercial | https://flame.is.tue.mpg.de |
| EMOCA v2 | MPI non-commercial | https://emoca.is.tue.mpg.de/license.html |
| HaMeR (code) | MIT — mais **dépend de MANO** (non-commercial) | https://raw.githubusercontent.com/geopavlakos/hamer/main/LICENSE.md |

**À documenter dans `docs/MODELS.md`** et à afficher dans le `README.md` : tout produit dérivé du pipeline est non-commercial tant qu'il intègre l'un quelconque de ces modèles. Si commercialisation future, il faudra remplacer ces composants.

### 0.4 EMOCA v2 est officiellement déprécié

Le README dit textuellement « EMOCA is now deprecated. » L'auteur redirige vers https://github.com/radekd91/inferno (module `FaceReconstruction`). Le brief dit « EMOCA non-négociable », donc on garde EMOCA v2 pour la Phase 2, mais il faut être conscient que :
- le repo n'évolue plus, les bugs ouverts ne seront pas corrigés ;
- les scripts `download_assets.sh` peuvent casser sans préavis si les URLs MPI changent ;
- toute migration vers inferno sera un refactor majeur (API différente).

---

## 1. Versions pinned

### 1.1 ML stack (machine cible)

| Composant | Version | Commit / source | Notes |
|---|---|---|---|
| **OS** | Ubuntu 22.04 LTS | — | Kernel ≥ 5.15. Driver NVIDIA ≥ 535 recommandé pour cu118+. |
| **CUDA system** | non requis | — | Les wheels PyTorch sont auto-suffisantes. CUDA toolkit système non nécessaire. |
| **NVIDIA driver** | ≥ 525 (cu121) ou ≥ 535 (cu126/128) | `nvidia-smi` ≥ 535.x | Pour cu113 (legacy), driver ≥ 470 suffit ; ≥ 525 fonctionne aussi. |
| **Python (SMPLer-X)** | 3.8.x | — | imposé par mmcv 1.7.1 |
| **Python (HaMeR)** | 3.10.x | — | imposé par le repo |
| **Python (EMOCA)** | 3.8.x | — | imposé par pytorch3d 0.6.2 |
| **SMPLer-X** | commit `064baef0e4ab5277a3297691bc1d46ea5412586f` (2026-02-12) | https://github.com/MotrixLab/SMPLer-X | repo transféré de `caizhongang` à `MotrixLab` ; redirection en place |
| **HaMeR** | commit `3a01849f4148352e9260b69bf28b65d1671a4905` (2026-02-07) | https://github.com/geopavlakos/hamer | |
| **EMOCA v2** | commit `e0be0dbc2d32629ae384ae10c0b7974948c994fd` (2024-12-06) | https://github.com/radekd91/emoca, branche `release/EMOCA_v2` | gel de fait depuis 2024 |
| **PyTorch (SMPLer-X)** | 1.12.0 + cu113 | `conda install pytorch==1.12.0 ... cudatoolkit=11.3 -c pytorch` | |
| **PyTorch (HaMeR)** | 1.13.x + cu117 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117` | torch non pinned dans `setup.py`, mais bloqué ≤1.13 par mmcv 1.3.9 |
| **PyTorch (EMOCA)** | 1.12.1 + cu113 | idem SMPLer-X | |
| **PyTorch3D (EMOCA)** | 0.6.2 | `pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2` | build C++ depuis source, fragile |
| **mmcv-full (SMPLer-X)** | 1.7.1 | wheel pré-build cu113/torch1.12 | |
| **mmdet (SMPLer-X)** | 2.26.0 | pip | |
| **mmcv (HaMeR)** | 1.3.9 | hard pin dans `setup.py` | source du blocage torch 1.x |
| **detectron2 (HaMeR)** | HEAD | git+https://github.com/facebookresearch/detectron2 | non pinned — risque de dérive |
| **smplx (vchoutas)** | 0.1.28 | https://github.com/vchoutas/smplx | partagé entre les 3 envs |
| **chumpy** | git HEAD `mattloper/chumpy` | dépendance legacy | nécessite numpy < 1.24 |

### 1.2 Viewer / Web stack

| Composant | Version | Source |
|---|---|---|
| **Node.js** | 24 LTS (Active LTS, support jusqu'à 2028-04-30) | https://nodejs.org/en/about/previous-releases |
| **Three.js** | r184 / `three@0.184.0` (2026-04-16) | https://github.com/mrdoob/three.js/releases |
| **@pixiv/three-vrm** | 3.5.2 | https://www.npmjs.com/package/@pixiv/three-vrm |
| **@pixiv/three-vrm-animation** | 3.5.2 | https://www.npmjs.com/package/@pixiv/three-vrm-animation |
| **Vite** | 8.0.10 (Vite 8 GA 2026-03-12, Rolldown bundler par défaut) | https://www.npmjs.com/package/vite |
| **TypeScript** | 6.0.3 | https://www.npmjs.com/package/typescript |

> **Pin recommandé pour `three`** : `~0.180.0`. three-vrm 3.5.2 est testé en dev contre `three@0.180.0`. Le `peerDependencies` déclare `>=0.137` (très permissif) mais ne garantit pas la compat avec r184. Démarrer prudent à 0.180, bumper plus tard.
>
> **Pin pour TypeScript** : 6.0.3, pas 7.0. TS 7 est en bêta depuis 2026-04-21 (réécriture en Go) — pas pour de la prod.

### 1.3 Blender + addon

| Composant | Version | Source |
|---|---|---|
| **Blender** | 4.5 LTS, point release 4.5.9 (2026-04-21) | https://www.blender.org/download/releases/ — choix LTS pour stabilité |
| **VRM_Addon_for_Blender** | v3.27.0 (2026-05-02) | https://github.com/saturday06/VRM-Addon-for-Blender |

> Le repo de l'addon a migré de underscores (`VRM_Addon_for_Blender`) vers tirets (`VRM-Addon-for-Blender`) ; la forme avec underscores reste accessible via redirect 301.

> **Mode headless de l'addon** : implicitement supporté (le scripting API ne dépend pas de l'UI), mais **non explicitement documenté**. Smoke test recommandé sur la machine cible avant de bâtir le pipeline dessus :
> ```bash
> blender -b --addons io_scene_vrm \
>   --python-expr "import bpy; bpy.ops.import_scene.vrm(filepath='test.vrm'); bpy.ops.export_scene.vrm_animation(filepath='out.vrma')"
> ```

---

## 2. Modèles et poids

### 2.1 Tableau récapitulatif

Les modèles avec « registration required » ne peuvent **pas** être téléchargés automatiquement par script — il faut créer un compte sur le site MPI, accepter la licence, puis copier les fichiers à la main. `download_weights.sh` ne pourra automatiser **que** les téléchargements directs (HuggingFace, dl.fbaipublicfiles.com, OpenMMLab).

| # | Composant | Source | Taille (estimée*) | Compte requis | URL |
|---|---|---|---|---|---|
| 1 | SMPLer-X H32* (recommandé) | HuggingFace | ~2.6 GB | non | https://huggingface.co/caizhongang/SMPLer-X/resolve/main/smpler_x_h32_correct.pth.tar |
| 2 | SMPLer-X L32 (fallback) | HuggingFace | ~1.3 GB | non | https://huggingface.co/caizhongang/SMPLer-X/resolve/main/smpler_x_l32.pth.tar |
| 3 | ViTPose pretrained (huge) | OpenMMLab / ViTPose repo | ~2.3 GB | non | https://github.com/ViTAE-Transformer/ViTPose (suivre instructions OSX) |
| 4 | mmdet Faster R-CNN R50 | OpenMMLab | ~160 MB | non | https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth |
| 5 | **SMPL-X** body models | smpl-x.is.tue.mpg.de | ~150 MB total | **oui** | https://smpl-x.is.tue.mpg.de — fichiers `SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz`, `MANO_SMPLX_vertex_ids.pkl`, `SMPL-X__FLAME_vertex_ids.npy`, `SMPLX_to_J14.pkl` |
| 6 | **SMPL** body models | smpl.is.tue.mpg.de | ~40 MB total | **oui** | https://smpl.is.tue.mpg.de — `SMPL_NEUTRAL.pkl`, etc. |
| 7 | HaMeR demo data | Google Drive / UT Austin | ~1–2 GB (non vérifié) | non | gdown id `1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT` ; fallback https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz |
| 8 | ViTDet detector | Facebook AI | ~2.6 GB | non | https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl |
| 9 | **MANO** model | mano.is.tue.mpg.de | ~5 MB | **oui** | https://mano.is.tue.mpg.de — `MANO_RIGHT.pkl` (HaMeR n'a besoin que du droit, le gauche est miroité) |
| 10 | EMOCA v2 LR MSE 20 (recommandé) | download.is.tue.mpg.de | ? (non documenté) | **oui** (license FLAME + EMOCA) | https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20.zip |
| 11 | DECA model | download.is.tue.mpg.de | ? | **oui** | https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/DECA.zip |
| 12 | DECA assets | download.is.tue.mpg.de | ? | **oui** | https://download.is.tue.mpg.de/emoca/assets/DECA.zip |
| 13 | FaceRecognition assets (EMOCA) | download.is.tue.mpg.de | ? | **oui** | https://download.is.tue.mpg.de/emoca/assets/FaceRecognition.zip |
| 14 | **FLAME** assets | download.is.tue.mpg.de | ~150 MB | **oui** | https://download.is.tue.mpg.de/emoca/assets/FLAME.zip — équivalent à un téléchargement direct depuis flame.is.tue.mpg.de |

\* Les tailles avec `?` ne sont **pas documentées dans les repos** ; à mesurer via `curl -IL` au premier téléchargement et noter dans `MODELS.md`.

### 2.2 Checksums

> **Aucun checksum officiel n'est publié par les repos amont.** Stratégie pour `verify_env.py` :
> 1. Au premier `download_weights.sh` réussi sur la machine cible, calculer SHA-256 de chaque fichier et écrire un `models/CHECKSUMS.sha256` versionné dans le repo (par toi, manuellement, après vérification visuelle).
> 2. À chaque exécution suivante, `verify_env.py` re-calcule et compare. Toute dérive arrête le pipeline.

### 2.3 Comptes à créer (par toi, avant exécution)

1. https://smpl-x.is.tue.mpg.de — accès SMPL-X
2. https://smpl.is.tue.mpg.de — accès SMPL
3. https://mano.is.tue.mpg.de — accès MANO
4. https://flame.is.tue.mpg.de — accès FLAME
5. https://emoca.is.tue.mpg.de — accès EMOCA (utilise les credentials FLAME)

---

## 3. Format `animation.npz`

Format produit par `pipeline/extract.py` après lissage One-Euro, consommé par `pipeline/retarget.py`.

### 3.1 Conventions

- **Rotations** : axis-angle (Rodrigues), 3 floats par articulation, **radians**.
- **Translations** : mètres.
- **Repère** : Y-up, Z-forward (vers la caméra), X-left ; main droite. Repère canonique SMPL-X tel que les bones sont retargetés sans miroitage. *Note : ce repère est convention par consensus communautaire, le README smplx ne le spécifie pas formellement — vérifier au premier rendu.*
- **dtype** : `float32` partout, sauf `confidence` en `float32` aussi (∈ [0, 1]).
- **fps** : champ `fps` (scalar `float32`) explicitement stocké dans le NPZ. Choisi par l'utilisateur via `config.yaml` (défaut **30 fps**) ; les vidéos YouTube ont des fps variables (24/25/29.97/30/60), `extract.py` ré-échantillonne à `fps` avant lissage.

### 3.2 Clés et shapes

Soit `T` le nombre de frames de l'animation lissée.

| Clé | Shape | dtype | Description |
|---|---|---|---|
| `fps` | `()` (scalar) | `float32` | Cadence de l'animation après ré-échantillonnage |
| `transl` | `(T, 3)` | `float32` | Translation racine (pelvis) en mètres |
| `global_orient` | `(T, 3)` | `float32` | Rotation racine, axis-angle |
| `body_pose` | `(T, 21, 3)` | `float32` | 21 articulations corps, axis-angle. Ordre canonique SMPL-X (cf. §4.1) |
| `left_hand_pose` | `(T, 15, 3)` | `float32` | 15 articulations main gauche, axis-angle. **`use_pca=False`** côté SMPLer-X |
| `right_hand_pose` | `(T, 15, 3)` | `float32` | idem main droite |
| `jaw_pose` | `(T, 3)` | `float32` | Mâchoire FLAME, axis-angle |
| `leye_pose` | `(T, 3)` | `float32` | Œil gauche, axis-angle |
| `reye_pose` | `(T, 3)` | `float32` | Œil droit, axis-angle |
| `expression` | `(T, 50)` | `float32` | Coefficients PCA d'expression FLAME. Dim **50** par défaut (matche EMOCA v2). *Flag : à vérifier en lisant `gdl/models/EmocaModel.py` `n_exp` au runtime ; le default DECA est 50 mais SMPLer-X par défaut n'en sort que 10 — `extract.py` doit padder/concaténer EMOCA→SMPLer-X cohéremment.* |
| `betas` | `(10,)` | `float32` | Forme corps SMPL-X, **constante** sur toute l'animation (estimée comme moyenne ou médiane sur la séquence). Dim 10 — fallback vers la valeur par défaut SMPL-X. |
| `confidence_body` | `(T,)` | `float32` | Confiance globale corps par frame, ∈ [0, 1]. Dérivée du score bbox détecteur + résidu reprojection 2D. |
| `confidence_lhand` | `(T,)` | `float32` | Confiance main gauche par frame |
| `confidence_rhand` | `(T,)` | `float32` | Confiance main droite par frame |
| `confidence_face` | `(T,)` | `float32` | Confiance visage par frame |
| `frame_indices` | `(T,)` | `int32` | Indices de frame source (avant ré-échantillonnage). Permet de retracer une frame NPZ → frame vidéo originale pour debug. |
| `source_video` | `()` (string scalar) | `str` (numpy unicode) | Chemin/nom du fichier vidéo d'entrée |
| `source_fps` | `()` | `float32` | fps original de la vidéo source |
| `meta_json` | `()` (string scalar) | `str` | JSON sérialisé : versions des modèles utilisés, commit hashes, paramètres One-Euro, date d'extraction |

> **SMPLer-X ne sort pas de confidence native.** Les arrays `confidence_*` sont calculés par `extract.py` à partir de :
> - le score de bbox du détecteur mmdet (par région : body / lhand / rhand / face) ;
> - un score dérivé du résidu de reprojection 2D entre `smplx_joint_proj` et la sortie d'un détecteur 2D (ViTPose/RTMPose) si tu en utilises un en amont. *À discuter en Phase 2 selon ce que SMPLer-X expose effectivement.*

### 3.3 Validation

`extract.py` doit asserter à l'écriture, et `retarget.py` doit re-asserter à la lecture :
- toutes les clés présentes ;
- shapes correctes ;
- dtypes corrects ;
- pas de NaN/Inf ;
- `fps > 0`, `T > 0`.

Une fonction `pipeline/animation_npz.py::validate(path) -> bool` centralise ces assertions et est appelée à toutes les frontières d'étapes.

---

## 4. Mapping SMPL-X → bones humanoïdes VRM

### 4.1 Joints SMPL-X (référence canonique)

55 joints, ordre `JOINT_NAMES` du repo `vchoutas/smplx` (https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py). Indices 0–54.

| Idx | Nom SMPL-X | Parent (idx, nom) |
|---:|---|---|
| 0 | pelvis | -1 (root) |
| 1 | left_hip | 0 pelvis |
| 2 | right_hip | 0 pelvis |
| 3 | spine1 | 0 pelvis |
| 4 | left_knee | 1 left_hip |
| 5 | right_knee | 2 right_hip |
| 6 | spine2 | 3 spine1 |
| 7 | left_ankle | 4 left_knee |
| 8 | right_ankle | 5 right_knee |
| 9 | spine3 | 6 spine2 |
| 10 | left_foot | 7 left_ankle |
| 11 | right_foot | 8 right_ankle |
| 12 | neck | 9 spine3 |
| 13 | left_collar | 9 spine3 |
| 14 | right_collar | 9 spine3 |
| 15 | head | 12 neck |
| 16 | left_shoulder | 13 left_collar |
| 17 | right_shoulder | 14 right_collar |
| 18 | left_elbow | 16 left_shoulder |
| 19 | right_elbow | 17 right_shoulder |
| 20 | left_wrist | 18 left_elbow |
| 21 | right_wrist | 19 right_elbow |
| 22 | jaw | 15 head |
| 23 | left_eye_smplhf | 15 head |
| 24 | right_eye_smplhf | 15 head |
| 25–39 | left_{index,middle,pinky,ring,thumb}{1,2,3} | enracinés à 20 left_wrist |
| 40–54 | right_{index,middle,pinky,ring,thumb}{1,2,3} | enracinés à 21 right_wrist |

### 4.2 Bones humanoïdes VRM 1.0 (référence canonique)

Spec : https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm-1.0/humanoid.md

55 noms standards. **15 sont obligatoires**, les 40 autres sont optionnels.

**Obligatoires (15)** : `hips`, `spine`, `head`, `leftUpperArm`, `leftLowerArm`, `leftHand`, `rightUpperArm`, `rightLowerArm`, `rightHand`, `leftUpperLeg`, `leftLowerLeg`, `leftFoot`, `rightUpperLeg`, `rightLowerLeg`, `rightFoot`.

**Optionnels (40)** : `chest`, `upperChest`, `neck`, `leftShoulder`, `rightShoulder`, `leftToes`, `rightToes`, `leftEye`, `rightEye`, `jaw`, et 30 bones de doigts (3 par doigt × 5 doigts × 2 mains).

**Doigts VRM 1.0 (15 par main)** :
- Pouce : `leftThumbMetacarpal`, `leftThumbProximal`, `leftThumbDistal`
- Index : `leftIndexProximal`, `leftIndexIntermediate`, `leftIndexDistal`
- Majeur : `leftMiddleProximal`, `leftMiddleIntermediate`, `leftMiddleDistal`
- Annulaire : `leftRingProximal`, `leftRingIntermediate`, `leftRingDistal`
- Auriculaire : `leftLittleProximal`, `leftLittleIntermediate`, `leftLittleDistal`
- (idem côté droit avec préfixe `right`)

### 4.3 Table de mapping (SMPL-X → VRM 1.0)

Mapping **par nom**, indépendant de l'index. Le bone VRM doit exister sur l'avatar chargé pour que la rotation soit appliquée ; sinon `retarget.py` log un warning et passe.

| Joint SMPL-X | Bone VRM 1.0 | Notes |
|---|---|---|
| pelvis (root) | `hips` | + global_orient + transl |
| left_hip | `leftUpperLeg` | |
| right_hip | `rightUpperLeg` | |
| spine1 | `spine` | toujours présent |
| left_knee | `leftLowerLeg` | |
| right_knee | `rightLowerLeg` | |
| spine2 | `chest` *(si présent)* | sinon ré-injecté dans `spine` ou perdu |
| left_ankle | `leftFoot` | |
| right_ankle | `rightFoot` | |
| spine3 | `upperChest` *(si présent)* | sinon ré-injecté dans `chest` ou `spine` |
| left_foot | `leftToes` *(si présent)* | optionnel |
| right_foot | `rightToes` *(si présent)* | optionnel |
| neck | `neck` *(si présent)* | sinon transmis dans `head` |
| left_collar | `leftShoulder` *(si présent)* | optionnel |
| right_collar | `rightShoulder` *(si présent)* | optionnel |
| head | `head` | toujours présent |
| left_shoulder | `leftUpperArm` | |
| right_shoulder | `rightUpperArm` | |
| left_elbow | `leftLowerArm` | |
| right_elbow | `rightLowerArm` | |
| left_wrist | `leftHand` | |
| right_wrist | `rightHand` | |
| jaw | `jaw` *(si présent)* | optionnel — sinon utilisé via expression `aa`/`oh` |
| left_eye_smplhf | `leftEye` *(si présent)* | optionnel |
| right_eye_smplhf | `rightEye` *(si présent)* | optionnel |
| left_thumb1 | `leftThumbMetacarpal` | **point d'attention** : VRM 1.0 a 3 segments thumb (Metacarpal, Proximal, Distal) ; SMPL-X a 3 segments thumb (1, 2, 3) — alignement direct par position dans la chaîne |
| left_thumb2 | `leftThumbProximal` | |
| left_thumb3 | `leftThumbDistal` | |
| left_index1 | `leftIndexProximal` | |
| left_index2 | `leftIndexIntermediate` | |
| left_index3 | `leftIndexDistal` | |
| left_middle1 | `leftMiddleProximal` | |
| left_middle2 | `leftMiddleIntermediate` | |
| left_middle3 | `leftMiddleDistal` | |
| left_ring1 | `leftRingProximal` | |
| left_ring2 | `leftRingIntermediate` | |
| left_ring3 | `leftRingDistal` | |
| left_pinky1 | `leftLittleProximal` | **renommage** : SMPL-X « pinky » = VRM « little » |
| left_pinky2 | `leftLittleIntermediate` | |
| left_pinky3 | `leftLittleDistal` | |
| right_thumb1 | `rightThumbMetacarpal` | (idem côté droit) |
| right_thumb2 | `rightThumbProximal` | |
| right_thumb3 | `rightThumbDistal` | |
| right_index1 | `rightIndexProximal` | |
| right_index2 | `rightIndexIntermediate` | |
| right_index3 | `rightIndexDistal` | |
| right_middle1 | `rightMiddleProximal` | |
| right_middle2 | `rightMiddleIntermediate` | |
| right_middle3 | `rightMiddleDistal` | |
| right_ring1 | `rightRingProximal` | |
| right_ring2 | `rightRingIntermediate` | |
| right_ring3 | `rightRingDistal` | |
| right_pinky1 | `rightLittleProximal` | |
| right_pinky2 | `rightLittleIntermediate` | |
| right_pinky3 | `rightLittleDistal` | |

> **Compat VRM 0.x** : si l'avatar fourni est un VRM 0.x, le pouce a la chaîne `Proximal/Intermediate/Distal` (3 segments mais nommés différemment). `vrm_inspector.py` doit détecter la version VRM (champ `version` dans l'extension `VRM` ou `VRMC_vrm`) et appliquer un mapping alternatif :
>
> | Pouce SMPL-X | VRM 0.x | VRM 1.0 |
> |---|---|---|
> | thumb1 | `leftThumbProximal` | `leftThumbMetacarpal` |
> | thumb2 | `leftThumbIntermediate` | `leftThumbProximal` |
> | thumb3 | `leftThumbDistal` | `leftThumbDistal` |

### 4.4 Calcul des offsets de rest-pose

Le mapping ci-dessus aligne les noms. Reste à corriger les **différences d'orientation de rest-pose** entre SMPL-X (T-pose, bras étendus horizontalement) et le VRM (souvent A-pose, légèrement vers le bas, parfois mains pré-pliées).

Algorithme (à implémenter dans `retarget.py`) :

```
Pour chaque (bone_smplx, bone_vrm) du mapping :
    R_smplx_rest = identité (T-pose canonique SMPL-X par construction)
    R_vrm_rest   = orientation locale du bone dans le VRM chargé (lue dynamiquement)
    R_offset     = R_vrm_rest^-1
    
    À chaque frame t :
        R_anim_smplx = rotation_axis_angle(animation[t][bone_smplx])
        R_anim_vrm   = R_offset · R_anim_smplx · R_vrm_rest
        # (à valider en pratique — l'ordre exact dépend de la convention parent/local de Blender et du VRM)
```

> **Cette formule est l'approximation standard mais doit être validée empiriquement au rendu.** L'ordre de composition (pré- ou post-multiplication) varie selon que les rotations sont exprimées en local-frame ou world-frame, et selon le sens des conventions Blender. La Phase 2 inclura un test de retargeting sur un VRM connu (ex. AliciaSolid) avec une animation T-pose statique pour valider l'identité.

`vrm_inspector.py` lit `R_vrm_rest` :
- si on travaille via Blender + VRM addon : `bpy.data.armatures[arm_name].bones[bone_name].matrix_local` ;
- si on travaille via pygltflib : on lit les `nodes[].rotation` (quaternions) glTF et on accumule les transformées le long de la chaîne hiérarchique.

---

## 5. Mapping FLAME → expressions VRM

### 5.1 Constat de fond

**Il n'existe aucun mapping canonique FLAME → ARKit/visemes/expressions sémantiques.** Les coefficients d'expression FLAME (10 dans SMPL-X par défaut, 50 dans EMOCA v2) sont des composantes PCA orthogonales **non-nommées** apprises par variance sur des scans 4D. Une « composante 1 » n'est pas « happy ».

Les 17 presets VRM 1.0 (`happy`, `sad`, `angry`, `relaxed`, `surprised`, `aa`, `ih`, `ou`, `ee`, `oh`, `blink`, `blinkLeft`, `blinkRight`, `lookUp`, `lookDown`, `lookLeft`, `lookRight`, `neutral`) sont, eux, des cibles **sémantiques** définies par l'auteur de chaque avatar via des morph targets ou des bindings de matériaux.

### 5.2 Stratégies possibles

Trois approches, par ordre de réalisme à mettre en œuvre :

#### A. Mapping géométrique (bones FLAME directs)
- **Mâchoire** : `jaw_pose` (FLAME) → **bone VRM `jaw`** (si présent). Direct, géométrique, fonctionne sans expression. C'est ce qui doit être fait pour la lip-sync par défaut. Si `jaw` absent → fallback sur expressions vowel `aa`/`ih`/`ou`/`ee`/`oh` (cf. ci-dessous).
- **Yeux** : `leye_pose`, `reye_pose` (FLAME) → bones VRM `leftEye`, `rightEye` (si présents). Sinon, dériver des expressions `lookLeft`/`lookRight`/`lookUp`/`lookDown` à partir de l'angle de regard.
- **Cligne d'œil** : non capturé directement par FLAME (pas de bone paupière). Détecter heuristiquement à partir de `expression` PCA : la composante PCA dominante pour le cligne est typiquement la première ou deuxième (à vérifier expérimentalement). Par défaut : ne pas tenter de mapper le cligne, le laisser être géré par le système procédural VRM (`overrideBlink: none`).

#### B. Lip-sync vowel (visemes)
FLAME ne sort pas de visemes. Approximation possible :
1. À chaque frame, calculer la déformation 3D de la bouche depuis FLAME (sommets de la région mouth selon `SMPL-X__FLAME_vertex_ids.npy`).
2. Comparer (par produit scalaire ou ACP supervisée préalable) à 5 cibles synthétiques `aa`, `ih`, `ou`, `ee`, `oh` générées en posant des coefficients FLAME prédéfinis.
3. Sortir 5 weights ∈ [0, 1] qui activent les expressions VRM correspondantes.

> **Cette stratégie est une approximation custom**, pas une méthode publiée. Pour la LSF ce n'est probablement pas critique (les composantes manuelles dominent), donc **option : skip dans la v1** et activer plus tard.

#### C. Émotions sémantiques (`happy`, `sad`, `angry`, `relaxed`, `surprised`)
EMOCA peut classifier l'émotion (basic 7) en sortie auxiliaire (cf. `gdl_apps/EMOCA/emotion/`). Si on récupère ce signal :
- mapping basic-7 → VRM 5 émotions :
  - happiness → `happy`
  - sadness → `sad`
  - anger → `angry`
  - surprise → `surprised`
  - calm/neutral → `relaxed` ou `neutral`
  - fear, disgust → pas de cible VRM standard ; ignorer ou logger
- les valeurs sont des probabilités softmax → utilisables directement comme weight VRM (∈ [0, 1]).

> **Recommandation v1 (Phase 2)** : implémenter A (jaw + eye bones directs) **uniquement**. C est optionnel-stretch. B est skip. La LSF se lit principalement aux mains et au corps ; la sur-ingénierie du visage en v1 est un mauvais investissement.

### 5.3 Logique d'inspection dynamique

`flame_to_vrm_mapping.py` implémente :

```python
def build_expression_mapping(vrm_metadata: dict) -> dict:
    """
    vrm_metadata : sortie de vrm_inspector.inspect(vrm_path).
    Retourne un dict {action: target} où :
      - action ∈ {'jaw', 'leye', 'reye'}  pour les bones
      - action ∈ {'happy', 'sad', ...}    pour les expressions
      - target = nom du bone/expression VRM, ou None si absent
    """
    available_bones = set(vrm_metadata['humanoid_bones'])
    available_expressions = set(vrm_metadata['expressions'])

    mapping = {}
    mapping['jaw'] = 'jaw' if 'jaw' in available_bones else None
    mapping['leye'] = 'leftEye' if 'leftEye' in available_bones else None
    mapping['reye'] = 'rightEye' if 'rightEye' in available_bones else None
    # … etc
    return mapping
```

Un VRM sans expression `aa` ou sans bone `jaw` → la lip-sync est dégradée mais le pipeline ne crash pas. Warning loggé.

---

## 6. Inspection dynamique du VRM

### 6.1 Stratégie retenue : Blender + VRM addon (côté retargeting)

`pipeline/retarget.py` est un script Blender invoqué en headless :

```bash
blender -b --addons io_scene_vrm \
  --python pipeline/retarget.py -- \
  --avatar /path/to/avatar.vrm \
  --animation /path/to/animation.npz \
  --output /path/to/output.vrma
```

Dans `retarget.py` :

```python
import bpy
import sys

# Args après "--"
argv = sys.argv[sys.argv.index("--") + 1:]
# ... parse argv ...

# 1. Importer le VRM
bpy.ops.import_scene.vrm(filepath=avatar_path)

# 2. Lire le squelette humanoïde via les PropertyGroups VRM addon
armature = bpy.context.active_object
ext = armature.data.vrm_addon_extension  # chemin exact à confirmer en Phase 2 — la doc 
                                          # vrm-addon-for-blender.info/en/scripting-api/ 
                                          # n'est pas accessible (404) au 2026-05-05 ;
                                          # à valider en lisant io_scene_vrm/ source code

# 3. Énumérer les bones humanoïdes mappés
humanoid_bones = {}
for human_bone in ext.vrm1.humanoid.human_bones:  # path approximatif
    if human_bone.node.bone_name:
        humanoid_bones[human_bone.bone] = human_bone.node.bone_name

# 4. Énumérer les expressions disponibles
expressions = list(ext.vrm1.expressions.preset.keys()) \
            + list(ext.vrm1.expressions.custom.keys())

# 5. Lire les rest poses (matrix_local) de chaque bone humanoïde
for vrm_bone_name, blender_bone_name in humanoid_bones.items():
    rest_matrix = armature.data.bones[blender_bone_name].matrix_local
    # ... stocker
```

> **Le chemin exact dans `vrm_addon_extension` est à vérifier à la Phase 2** en lisant la source `io_scene_vrm/` du repo. La doc en ligne renvoie 404 et les versions de l'API ont évolué.

### 6.2 Stratégie alternative : pygltflib (sans Blender)

Pour `vrm_inspector.py` pur Python (utilisé par exemple pour le `verify_env.py` qui valide qu'un VRM est compatible **sans** lancer Blender) :

```python
from pygltflib import GLTF2

gltf = GLTF2().load(vrm_path)
vrmc = gltf.extensions.get('VRMC_vrm') or gltf.extensions.get('VRM')
# Pour VRM 1.0 :
human_bones = vrmc['humanoid']['humanBones']  # dict bone_name -> {node: int}
# Pour VRM 0.x :
# vrmc['humanoid']['humanBones']  # liste de {bone, node}

for bone_name, info in human_bones.items():
    node_idx = info['node']
    node = gltf.nodes[node_idx]
    print(bone_name, node.name, node.rotation, node.translation)

# Expressions VRM 1.0 :
expressions = list(vrmc['expressions']['preset'].keys()) + \
              list(vrmc['expressions'].get('custom', {}).keys())
# VRM 0.x :
# vrmc['blendShapeMaster']['blendShapeGroups']
```

> Cette approche **ne calcule pas** la rest-pose hiérarchique cumulée (il faut la dériver en parcourant les parents). Suffisante pour valider la présence/absence d'un bone ; insuffisante pour calculer R_vrm_rest dans le retargeting → on garde Blender pour `retarget.py`, et `pygltflib` pour les checks rapides.

### 6.3 API de `vrm_inspector.py`

```python
@dataclass
class VRMMetadata:
    version: Literal["0.x", "1.0"]
    humanoid_bones: dict[str, str]       # vrm_bone_name -> blender_bone_name (ou node.name)
    expressions: list[str]               # noms (presets + custom) disponibles
    rest_poses_local: dict[str, np.ndarray]  # vrm_bone_name -> matrix 4×4 (rest pose locale)
    raw_metadata: dict                   # tout l'extension VRMC_vrm pour debug

def inspect(vrm_path: Path, *, use_blender: bool = False) -> VRMMetadata:
    """
    Inspecte un fichier VRM et retourne ses métadonnées.
    use_blender=False : pygltflib only, rapide, pas de rest-poses cumulées
    use_blender=True  : exec via subprocess Blender, complet
    """
```

Cette fonction est **isolée et unit-testable** (cf. exigence du brief « Tester le chargement et l'inspection d'un VRM est une fonction isolée »).

---

## 7. Flow de la pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  VIDEO (.mp4, fps variable)                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ENV: smplerx                                                │
│  pipeline/extract.py phase 1                                 │
│   - ffmpeg → frames PNG @ source_fps                         │
│   - mmdet Faster R-CNN → bbox personne par frame             │
│   - SMPLer-X H32* → SMPL-X params per frame                  │
│   - Output: tmp_smplerx.npz (params bruts)                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ENV: hamer (separate venv)                                  │
│  pipeline/extract.py phase 2                                 │
│   - ViTDet → hand bboxes                                     │
│   - HaMeR → MANO params per hand per frame                   │
│   - Mano2Smpl-X bridge → remplacer hand_pose dans SMPL-X     │
│   - Output: tmp_hamer.npz                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ENV: emoca (separate venv)                                  │
│  pipeline/extract.py phase 3                                 │
│   - FAN face detection                                       │
│   - EMOCA v2 → FLAME params (shape, exp, pose, jaw)          │
│   - Remplacer jaw_pose / expression dans SMPL-X              │
│   - Output: tmp_emoca.npz                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ENV: smplerx (orchestrator, smoothing pure-numpy)           │
│  pipeline/smoothing.py                                       │
│   - Resample → fps cible (config.yaml, défaut 30)            │
│   - One-Euro filter sur quaternions (cf. §8)                 │
│   - Compute confidence_* arrays                              │
│   - Output: animation.npz (format §3)                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Blender headless + VRM addon                                │
│  pipeline/retarget.py --avatar X.vrm                         │
│   - Charge avatar.vrm                                        │
│   - vrm_inspector → bones disponibles + rest poses           │
│   - Bake animation.npz sur l'armature VRM                    │
│   - Map FLAME jaw → VRM jaw bone                             │
│   - Map FLAME expression → expressions VRM (cf. §5)          │
│   - Export VRM Animation (.vrma)                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  output/clip1.vrma                                           │
│  Chargé par viewer Three.js + @pixiv/three-vrm-animation     │
└─────────────────────────────────────────────────────────────┘
```

### 7.1 Orchestrateur `pipeline/pipeline.py`

```python
def main(video_path, avatar_path, output_dir, config):
    tmp = output_dir / "tmp"
    
    run_subprocess(["envs/smplerx/bin/python", "pipeline/extract.py",
                    "--phase", "smplerx",
                    "--video", video_path,
                    "--output", tmp / "smplerx.npz"])
    
    run_subprocess(["envs/hamer/bin/python", "pipeline/extract.py",
                    "--phase", "hamer",
                    "--video", video_path,
                    "--input", tmp / "smplerx.npz",
                    "--output", tmp / "hamer.npz"])
    
    run_subprocess(["envs/emoca/bin/python", "pipeline/extract.py",
                    "--phase", "emoca",
                    "--video", video_path,
                    "--input", tmp / "hamer.npz",
                    "--output", tmp / "emoca.npz"])
    
    # Smoothing en pure numpy → run direct
    smoothing.smooth(tmp / "emoca.npz", output_dir / "animation.npz", config)
    
    # Retargeting via Blender headless
    run_subprocess(["blender", "-b", "--addons", "io_scene_vrm",
                    "--python", "pipeline/retarget.py", "--",
                    "--avatar", avatar_path,
                    "--animation", output_dir / "animation.npz",
                    "--output", output_dir / "clip.vrma"])
```

---

## 8. Lissage One-Euro

Référence : https://gery.casiez.net/1euro/ ; impl https://github.com/casiez/OneEuroFilter

### 8.1 Paramètres par groupe (defaults proposés, overridables via config.yaml)

| Groupe | min_cutoff (Hz) | beta | Justification |
|---|---:|---:|---|
| `transl` (translation racine) | 0.5 | 0.05 | Mouvement lent, peu de jitter |
| `body_pose` (corps) | 1.0 | 0.1 | Compromis classique mocap |
| `hand_pose` (mains) | 1.5 | 0.2 | Plus rapide en signation, plus de bruit |
| `face` (jaw, eyes, expression) | 1.5 | 0.1 | Vise stabilité |

> Ces valeurs sont des **points de départ** issus de la littérature mocap (MediaPipe, VIBE, PyMAF). Le site officiel One-Euro ne publie pas de défauts canoniques — il prescrit la procédure de tuning empirique (β=0, baisser min_cutoff jusqu'à élimination du jitter ; puis remonter β jusqu'à élimination du lag).

### 8.2 Lissage de rotations

L'axis-angle n'est **pas** filtrable composante-par-composante (manifold non-euclidien, multi-valué). Procédure :

1. Convertir chaque rotation par frame en quaternion unitaire.
2. **Hemisphere consistency** : pour chaque frame `t`, si `dot(q_t, q_{t-1}) < 0`, négativer `q_t` (q et -q représentent la même rotation).
3. Appliquer One-Euro sur chacune des 4 composantes du quaternion indépendamment.
4. **Re-normaliser** le quaternion filtré à la norme 1.
5. Reconvertir en axis-angle pour l'écriture NPZ.

`transl` et `expression` sont filtrables composante-par-composante directement (espace euclidien).

### 8.3 Implémentation

`pipeline/smoothing.py` est en **pur numpy/scipy**, sans dépendance ML. Il tourne dans n'importe quel des trois envs (par défaut on le lance dans `smplerx`). Une lib au choix :
- réimplémentation maison du 1€ filter (~50 lignes) ;
- `pyfilterbank` ou `OneEuroFilter` PyPI (à vérifier disponibilité et licence en Phase 2).

---

## 9. Viewer Three.js

### 9.1 Pile retenue

```json
{
  "dependencies": {
    "three": "~0.180.0",
    "@pixiv/three-vrm": "^3.5.2",
    "@pixiv/three-vrm-animation": "^3.5.2"
  },
  "devDependencies": {
    "vite": "^8.0.10",
    "typescript": "^6.0.3",
    "@types/three": "~0.180.0"
  }
}
```

### 9.2 Pattern de chargement avec retargeting auto

```typescript
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import {
  createVRMAnimationClip,
  VRMAnimationLoaderPlugin,
  VRMLookAtQuaternionProxy,
} from '@pixiv/three-vrm-animation';
import * as THREE from 'three';

const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));
loader.register((parser) => new VRMAnimationLoaderPlugin(parser));

// 1. Avatar
const avatarGltf = await loader.loadAsync(avatarUrl);
const vrm = avatarGltf.userData.vrm;
VRMUtils.removeUnnecessaryVertices(vrm.scene);
VRMUtils.removeUnnecessaryJoints(vrm.scene);

const lookAtQuatProxy = new VRMLookAtQuaternionProxy(vrm.lookAt);
lookAtQuatProxy.name = 'lookAtQuaternionProxy';
vrm.scene.add(lookAtQuatProxy);

scene.add(vrm.scene);

// 2. Animation
const vrmaGltf = await loader.loadAsync(animationUrl);
const vrmAnim = vrmaGltf.userData.vrmAnimations[0];

// Le retargeting bone-par-bone est fait par createVRMAnimationClip :
// les pistes du .vrma sont nommées par bones humanoïdes VRM ;
// la fonction résolve ces noms vers les nœuds three.js de CET avatar
// et corrige les rest-poses.
const clip = createVRMAnimationClip(vrmAnim, vrm);

const mixer = new THREE.AnimationMixer(vrm.scene);
const action = mixer.clipAction(clip);
action.play();

// 3. Render loop
function animate(dt: number) {
    mixer.update(dt);
    vrm.update(dt);
    renderer.render(scene, camera);
}
```

### 9.3 Drag & drop

`viewer/src/loader.ts` implémente un dispatcher : au drop d'un fichier sur la page, regarder l'extension et le contenu du glTF ; si l'extension `VRMC_vrm` est présente → c'est un avatar (le remplacer dans la scène) ; si `VRMC_vrm_animation` est présent → c'est une animation (la jouer sur l'avatar courant). Si ni l'un ni l'autre → afficher une erreur dans l'UI.

### 9.4 Avatar paramétrable via URL

```typescript
const params = new URLSearchParams(window.location.search);
const avatarUrl = params.get('avatar') ?? 'avatars/default.vrm';
const animationUrl = params.get('animation') ?? 'animations/default.vrma';
```

Si l'animation a été produite pour un autre VRM, `createVRMAnimationClip` fait quand même le retargeting (puisque les pistes du .vrma sont par nom de bone humanoïde standard). On log les bones cible non-trouvés sur l'avatar courant comme warnings dans la console.

---

## 10. Risques connus et points d'attention

### 10.1 Conflits / fragilité de dépendances

- **chumpy + numpy ≥ 1.24** : `np.bool` etc. supprimés → pin numpy < 1.24 dans les trois envs. Documenté dans setup.sh.
- **detectron2 from git HEAD (HaMeR)** : risque de dérive — pin sur un commit spécifique dans `requirements.txt`. À choisir un commit testé en Phase 2.
- **pytorch3d 0.6.2 (EMOCA)** : build C++ depuis source, échec compilation très commun. Documenter dans `TROUBLESHOOTING.md` les erreurs typiques (mismatch CUDA toolkit système vs conda, GCC version, ninja).
- **mmcv-full 1.7.1 (SMPLer-X)** : wheels pré-build sont versionnés `mmcv_full-1.7.1+cu113-torch1.12-py38`. La page `download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html` doit rester accessible.
- **Bug torchgeometry** : `RuntimeError: Subtraction, the '-' operator, with a bool tensor is not supported` — patch documenté à appliquer dans setup.sh.
- **SinePositionalEncoding registration conflict** : entre mmcv et mmdet 2.x, ajouter `force=True` dans la registration. Patch à scripter.

### 10.2 Variations entre VRM

- **VRM 0.x vs 1.0** : naming des bones du pouce divergent (cf. §4.3). Détection de version obligatoire dans `vrm_inspector.py`.
- **Bones optionnels manquants** : `chest`, `upperChest`, `neck`, `jaw`, eyes peuvent être absents. Le retargeting doit dégrader gracieusement (collapse vers le parent) plutôt que crasher.
- **Expressions custom** : un avatar peut avoir des expressions non-preset (`'smile_wide'`, etc.). Notre mapping FLAME→VRM ignore les customs en v1 ; les noms des expressions disponibles sont juste loggées.
- **VRM 0.x avec axe inversé** : certains VRM 0.x antérieurs à la convention finale ont des bones avec orientation Z inversée. La rest-pose dynamique compense, mais des artefacts résiduels peuvent apparaître. Tester avec plusieurs VRM (AliciaSolid, VRoid Studio default, AvatarMaker).
- **Spring bones (cheveux, jupes)** : `@pixiv/three-vrm` les anime automatiquement à partir des transformations bone parent + simulation. Aucune action côté pipeline.

### 10.3 Vidéos YouTube

- **fps variable** : `extract.py` doit forcer un ré-échantillonnage à fps constant via ffmpeg avant SMPLer-X.
- **Personnes multiples** : SMPLer-X accepte plusieurs détections par frame ; on retient la **bbox la plus grande** par défaut, avec option `--track-id` pour le tracking. La LSF a typiquement un signeur unique au centre.
- **Cropping vertical/portrait** : YouTube Shorts en 9:16 — vérifier que mmdet et SMPLer-X fonctionnent correctement, sinon padder en 16:9.
- **Watermarks / overlays** : peuvent dégrader la détection. Pas de mitigation prévue.
- **Sous-titres incrustés** : pareil. À cropper manuellement si besoin (`config.yaml` : `crop: [x, y, w, h]`).

### 10.4 Performance

- **Pipeline complet pour une vidéo de 60 s à 30 fps (1800 frames)** : estimation grossière sur A4500 :
  - SMPLer-X H32* : ~3–5 fps inférence → 6–10 min
  - HaMeR : ~5–10 fps → 3–6 min
  - EMOCA : ~10–15 fps → 2–3 min
  - Total : 15–25 min de processing pour 60 s de vidéo. À mesurer en Phase 4.
- **VRAM** : SMPLer-X H32* ~6 GB, HaMeR ~5 GB, EMOCA ~4 GB. **Largement** sous les 20 GB de l'A4500.

### 10.5 Confidence / robustesse

- **SMPLer-X ne sort pas de confidence native** (cf. §3.2). La proxy via score bbox + résidu reprojection est imparfaite. Pour la LSF où la qualité des mains est critique, c'est un risque : un fingerspelling raté ne sera pas détecté par le système et passera dans le NPZ comme bruit.
- **Mitigation** : ajouter un seuil dans `--debug-overlay` qui surligne les frames à confidence faible en rouge dans la vidéo de debug.

### 10.6 Format de sortie animation

- **Adoption .vrma** : si tu préfères tenir au .glb baké pour des questions d'écosystème (Mixamo etc.), il faut renoncer au retargeting auto runtime fourni par `@pixiv/three-vrm-animation`. À ce moment, le viewer doit faire le bone-name-rewriting à la main (cf. forum threejs) — c'est faisable mais fragile et plus de code à maintenir.

### 10.7 Mise à jour des modèles

- **MPI URLs** peuvent changer sans préavis. SMPLer-X : repo a déjà migré de namespace (caizhongang→MotrixLab). EMOCA : déprécié, l'auteur peut décider de fermer le bucket de download. **Mitigation** : au premier `download_weights.sh` réussi sur la machine cible, archiver les fichiers téléchargés dans un stockage durable (S3, GCS, Drive privé) et ajouter ces URLs comme fallback dans le script.

---

## 11. Récap des incertitudes explicites

| # | Sujet | Action en Phase 2 |
|---|---|---|
| 1 | Coordonnées SMPL-X (Y-up vs Z-up) | Vérifier au premier rendu test |
| 2 | EMOCA `n_exp` exact (50 ?) | Lire `gdl/models/EmocaModel.py` au runtime |
| 3 | Tailles exactes des poids EMOCA / HaMeR | `curl -IL` au premier téléchargement, écrire dans `MODELS.md` |
| 4 | Mode headless du VRM addon | Smoke test à Phase 3 (setup.sh inclura un test bpy.ops.import_scene.vrm + export) |
| 5 | Chemin exact `vrm_addon_extension` API | Lire la source `io_scene_vrm/` (lib Blender addon) en Phase 2 |
| 6 | Ordre de composition retargeting (R_offset · R_anim · R_rest) | Test sur AliciaSolid avec animation T-pose statique en Phase 2 |
| 7 | Compat three r184 + three-vrm 3.5.2 | Pin three à 0.180 ; si OK plus tard, bump |
| 8 | Mano2Smpl-X bridge (HaMeR → SMPL-X hands) | Lire le repo et décider à Phase 2 ; alternative : recoder un swap minimal |
| 9 | Confidence proxy effective (bbox score + reprojection) | Concevoir et valider en Phase 2 |
| 10 | Mapping émotions EMOCA → VRM (option C §5.2) | Décision à Phase 2, défaut = ne pas implémenter |

---

## 12. Sources

### ML
- SMPLer-X : https://github.com/MotrixLab/SMPLer-X
- HaMeR : https://github.com/geopavlakos/hamer
- Mano2Smpl-X (bridge HaMeR→SMPL-X) : https://github.com/VincentHu19/Mano2Smpl-X
- EMOCA : https://github.com/radekd91/emoca (branche `release/EMOCA_v2`)
- inferno (successeur EMOCA) : https://github.com/radekd91/inferno
- smplx (vchoutas) : https://github.com/vchoutas/smplx
- joint_names.py : https://raw.githubusercontent.com/vchoutas/smplx/main/smplx/joint_names.py
- body_models.py : https://raw.githubusercontent.com/vchoutas/smplx/main/smplx/body_models.py
- SMPL-X paper (Pavlakos+ 2019) : https://openaccess.thecvf.com/content_CVPR_2019/papers/Pavlakos_Expressive_Body_Capture_3D_Hands_Face_and_Body_From_a_CVPR_2019_paper.pdf
- FLAME paper (Li+ 2017) : https://3dvar.com/Li2017Learning.pdf
- One-Euro filter : https://gery.casiez.net/1euro/

### VRM / Three.js
- Spec VRM : https://github.com/vrm-c/vrm-specification
- Spec humanoid 1.0 : https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm-1.0/humanoid.md
- Spec expressions 1.0 : https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm-1.0/expressions.md
- @pixiv/three-vrm : https://github.com/pixiv/three-vrm
- VRM_Addon_for_Blender : https://github.com/saturday06/VRM-Addon-for-Blender
- VRMA export doc : https://vrm-addon-for-blender.info/en-us/ui/export_scene.vrma/

### Stack
- PyTorch versions : https://pytorch.org/get-started/previous-versions/
- Three.js releases : https://github.com/mrdoob/three.js/releases
- Vite : https://vite.dev/
- Node.js LTS : https://nodejs.org/en/about/previous-releases
- Blender releases : https://www.blender.org/download/releases/

### Comptes à créer
- https://smpl-x.is.tue.mpg.de
- https://smpl.is.tue.mpg.de
- https://mano.is.tue.mpg.de
- https://flame.is.tue.mpg.de
- https://emoca.is.tue.mpg.de
