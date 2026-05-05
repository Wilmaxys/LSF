# Troubleshooting — erreurs courantes et solutions

## Installation (`scripts/setup.sh`)

### `nvidia-smi: command not found`
Le driver NVIDIA n'est pas installé.
```bash
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

### Driver < 525 détecté
Le pipeline utilise CUDA 11.x (cu113, cu117) — driver ≥ 470 minimum, ≥ 525
recommandé. Pour mettre à jour :
```bash
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

### Échec download Blender
URL changée ou pas de connexion internet. Tester :
```bash
curl -I "https://download.blender.org/release/Blender4.5/blender-4.5.9-linux-x64.tar.xz"
```
Si 404, vérifier la dernière version sur https://www.blender.org/download/releases/
et ajuster `BLENDER_VERSION` dans `scripts/setup.sh`.

### `pytorch3d` build failure (env emoca)
Le build de `pytorch3d==0.6.2` est notoirement fragile. Causes courantes :

1. **CUDA toolkit système incompatible** : le cudatoolkit conda 11.3 ne suffit
   pas pour compiler ; il faut un CUDA toolkit système (nvcc) de la même série.
   ```bash
   nvcc --version
   ```
   Si nvcc absent ou version ≠ 11.x :
   ```bash
   sudo apt-get install -y cuda-toolkit-11-3
   # Ou télécharger depuis https://developer.nvidia.com/cuda-11-3-0-download-archive
   ```

2. **GCC trop récent** (>= 11) : pytorch3d 0.6.2 attend GCC ≤ 10.
   ```bash
   sudo apt-get install -y gcc-10 g++-10
   export CC=gcc-10 CXX=g++-10
   pip install --no-cache-dir 'git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2'
   ```

3. **Out of memory pendant le build** : le compilateur consomme beaucoup de RAM.
   Limiter la parallélisation :
   ```bash
   MAX_JOBS=2 pip install --no-cache-dir 'git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2'
   ```

### `mmcv-full` install échoue (env smplerx)
Le wheel pré-build cu113/torch1.12 est nécessaire :
```bash
conda activate lsf-smplerx
pip install mmcv-full==1.7.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
```

### Smoke test Blender headless échoue
Manque libgl1/libegl1. Le script tente de les installer mais sur certaines
distrib (Fedora, Debian sans `--no-install-recommends`), il faut compléter :
```bash
sudo apt-get install -y libgl1 libegl1 libgles2 libosmesa6
```

## Téléchargement des poids (`scripts/download_weights.sh`)

### URL MPI 404 / "Forbidden"
Le téléchargement direct des poids MPI requiert un cookie de session post-login.
**Tu ne peux pas curl/wget directement les URLs MPI.** Procédure :
1. Se connecter sur https://smpl-x.is.tue.mpg.de
2. Aller dans Downloads
3. Cliquer sur le bouton de téléchargement
4. Le fichier arrive dans ton ~/Downloads, transférer manuellement vers le serveur :
   ```bash
   scp ~/Downloads/SMPLX_v1_1.zip user@serveur:lsf-pipeline/pipeline/models/smplx/
   ```

### `Mauvais checksum SHA-256` après relance
Un fichier a été tronqué pendant le téléchargement. Solution :
```bash
rm pipeline/models/<chemin>/<fichier_corrompu>
bash scripts/download_weights.sh  # idempotent, re-téléchargera juste celui-ci
```

### Checksums absents (`CHECKSUMS.sha256` introuvable)
C'est normal au premier run. Le fichier est généré à la fin du
`download_weights.sh` — relancer le script jusqu'à la complétion.

## Lancement du pipeline (`pipeline/pipeline.py`)

### `Avatar VRM incompatible — bone obligatoire manquant : leftUpperArm`
L'avatar fourni n'a pas tous les bones humanoïdes obligatoires de la spec VRM.
Voir [docs/AVATARS.md](AVATARS.md). Solutions :
- Choisir un autre avatar (VRoid Studio export par défaut tous les bones)
- Ou : compléter le mapping humanoïde dans Blender + VRM addon, ré-exporter

### `Environnements Python ML manquants` en mode normal
Les envs n'ont pas été créés. Lancer ou relancer `bash scripts/setup.sh`.

### En mode `--dry-run` : `[DRY-RUN] Environnements Python ML manquants`
C'est un **warning, pas une erreur** — tu peux faire le dry-run sur ton Mac
sans avoir installé les envs. Le pipeline réel ne tournera évidemment pas.

### `RuntimeError: CUDA out of memory` (SMPLer-X H32)
Le variant H32 demande ~6 GB de VRAM, mais avec activations, batchnorm, etc.
le pic peut dépasser 12 GB. Solutions :
1. Utiliser le variant L32 (327M params) :
   ```yaml
   # Dans config.yaml
   pipeline:
     smplerx_model: l32
   ```
2. Réduire la résolution vidéo en amont (`ffmpeg -vf scale=720:-1`).
3. Fermer toute autre application GPU.

### `KeyError: 'SinePositionalEncoding is already registered'`
Conflit entre mmcv et mmdet. Patch :
```bash
conda activate lsf-smplerx
python -c "
import mmcv.cnn.bricks.transformer as t
# Force re-registration
"
```
Si le problème persiste, c'est probablement que mmcv-full et mmdet ont des
versions incompatibles. Re-installer dans l'ordre :
```bash
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html --force-reinstall
pip install mmdet==2.26.0 --force-reinstall
```

### `RuntimeError: Subtraction, the '-' operator, with a bool tensor is not supported`
Bug connu de `torchgeometry` (utilisé indirectement par SMPLer-X) avec PyTorch
1.12+. Patch :
```bash
conda activate lsf-smplerx
python -c "
import torchgeometry
import os
path = os.path.dirname(torchgeometry.__file__) + '/core/conversions.py'
print('Patch :', path)
# Remplacer manuellement '1 - mask_d2' par '~mask_d2' (ligne ~301-302)
"
```
Cf. https://github.com/PyMartin/I2L-MeshNet_RELEASE/issues/9 pour le diff exact.

### EMOCA `download_assets.sh` ne télécharge rien
Probablement le compte FLAME et/ou EMOCA n'a pas été créé. Vérifier en se
connectant manuellement sur :
- https://flame.is.tue.mpg.de
- https://emoca.is.tue.mpg.de

### Blender refuse de lancer en mode background : `Error: cannot open display`
Le smoke test du `setup.sh` passait, donc les libs sont là. Cause possible :
une variable d'env `DISPLAY` est définie alors que pas de X11.
```bash
unset DISPLAY
python pipeline/pipeline.py ...
```

### Le `.vrma` produit a une animation cassée / décalée
Plusieurs causes possibles :
1. **Repère SMPL-X mal interprété** : vérifier le rendu test sur AliciaSolid.
   Le repère canonique SMPL-X est Y-up ; le pipeline ne corrige pas pour les
   avatars en Z-up.
2. **VRM 0.x avec rest pose A-pose** : certains avatars VRM 0.x ont une rest
   pose en A-pose (bras légèrement vers le bas) au lieu de T-pose. Le
   retargeting calcule des offsets dynamiques mais ils peuvent ne pas être
   parfaits.
3. **Avatar avec scaling non-1.0 sur les bones** : vérifier dans Blender que
   tous les bones ont scale = (1,1,1) en rest pose.

Activer `--debug-overlay` pour voir si la confidence par frame est cohérente
avec le résultat visuel.

## Viewer Three.js

### `npm install` échoue (Node version)
Vérifier ta version Node :
```bash
node --version  # doit être ≥ 22 (idéal : 24 LTS)
```
Si trop ancien : nvm install 24, nvm use 24.

### Le viewer charge mais l'avatar n'est pas visible
- Vérifier la console JS pour les erreurs de chargement
- Le viewer cherche `avatars/default.vrm` par défaut. Si tu n'en as pas mis,
  passer un avatar via URL : `?avatar=avatars/<ton_fichier>.vrm`
- Tester par drag & drop : déposer ton `.vrm` directement sur la page.

### `Erreur : Le fichier ne contient aucune animation`
Tu as déposé un `.vrm` (avatar) au lieu d'un `.vrma` ou `.glb` (animation).
Le viewer charge automatiquement comme avatar — le warning vient de l'autre
input attendu.

### Les pistes d'animation sont là mais l'avatar ne bouge pas
- Vérifier dans la console qu'il y a des warnings "bone not found"
- L'avatar peut avoir des bones que le `.vrma` n'a pas (ou inversement) ;
  c'est OK, c'est juste partiel.
- Vérifier que `vrm.update(dt)` est bien appelé dans la boucle de rendu
  (par défaut oui, dans `Viewer.render()`).

### Le viewer plante avec `WebGL Context Lost`
Carte graphique faible. Tester avec :
- Désactiver les autres onglets
- Réduire la taille de la fenêtre
- Le viewer ne fait pas de rendu Cycles/Eevee — pour des avatars très
  haute résolution, c'est rare mais possible.

## Vérification d'environnement (`scripts/verify_env.py`)

### `Imports orchestrateur : import cv2 — KeyError ou ModuleNotFoundError`
Le pkg est `opencv-python-headless` côté orchestrateur ; il fournit le module
`cv2` standard. Si l'import échoue, ré-installer :
```bash
conda activate lsf-orchestrator
pip install --force-reinstall opencv-python-headless
```

### `Env smplerx : torch+CUDA — cuda=False`
Le venv smplerx n'arrive pas à voir le GPU. Vérifier :
```bash
conda activate lsf-smplerx
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```
Si `False` mais `nvidia-smi` voit le GPU : torch a été installé en version CPU.
Ré-installer :
```bash
pip install --force-reinstall torch==1.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

## Quand demander de l'aide

Si rien ci-dessus ne résout ton problème, ouvrir une issue avec :
1. La sortie complète de `python scripts/verify_env.py`
2. La sortie de `nvidia-smi`
3. Le fichier `pipeline.log` (si présent) ou la sortie complète du pipeline
4. Version OS : `cat /etc/os-release | head -2`
5. Versions des modèles : `ls -la pipeline/models/`
