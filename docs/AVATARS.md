# Avatars VRM — comment en utiliser un et vérifier sa compatibilité

## Qu'est-ce qu'un fichier VRM ?

VRM est un format de fichier 3D standard pour les avatars de personnages
humanoïdes (initialement créé pour VTubers / VR). Un `.vrm` contient :
- la géométrie 3D (mesh)
- le squelette humanoïde (~55 bones standardisés)
- des expressions faciales (sourire, voyelles pour lip-sync, etc.)
- des matériaux et textures

Le pipeline LSF utilise n'importe quel `.vrm` conforme **VRM 0.x** ou **VRM 1.0**
sans modification de code : le mapping des animations sur l'avatar est
calculé à l'exécution.

## Où trouver des avatars VRM gratuits

### Catalogues officiels
- **VRoid Hub** — https://hub.vroid.com/ — bibliothèque officielle, plusieurs
  milliers d'avatars en téléchargement libre (filtre "Use" pour vérifier les
  conditions d'utilisation)
- **Booth** — https://booth.pm/ — store japonais, beaucoup d'avatars VRM
  gratuits et payants
- **VRChat Avatars** — https://vrchat.com (souvent en VRM compatible)

### Créer son propre avatar
- **VRoid Studio** — https://vroid.com/en/studio
  - Application gratuite (Mac / Windows / iOS)
  - Interface non-technique pour personnaliser un avatar
  - Export direct en VRM 0.x ou 1.0

### Avatars classiques pour tester
- **AliciaSolid** (VRoid Hub) — référence VRM 1.0 souvent citée pour les tests
- **Vita** (VRoid Studio default)

## Utiliser un avatar dans le pipeline

### 1. Le placer dans le repo

```bash
# Sur le serveur GPU (pour le pipeline)
mkdir -p data/avatars
cp /chemin/vers/mon_avatar.vrm data/avatars/

# Sur ton Mac (pour le viewer)
mkdir -p viewer/public/avatars
cp /chemin/vers/mon_avatar.vrm viewer/public/avatars/
```

### 2. Lancer le pipeline avec cet avatar

```bash
python pipeline/pipeline.py \
    --video data/input/clip.mp4 \
    --avatar data/avatars/mon_avatar.vrm \
    --output data/output/clip.vrma
```

Le pipeline :
- Inspecte l'avatar (bones humanoïdes présents, expressions disponibles)
- Loggue un warning pour chaque bone optionnel manquant (ex: `jaw`, `leftEye`)
- Calcule le retargeting au runtime — pas de hardcoding par avatar

### 3. Charger dans le viewer

```
http://localhost:5173/?avatar=avatars/mon_avatar.vrm&animation=animations/clip.vrma
```

Ou par drag & drop sur la page.

## Vérifier qu'un VRM est compatible

Avant d'investir 20 min de processing GPU, valide ton avatar :

```bash
conda activate lsf-orchestrator
python scripts/verify_env.py --vrm data/avatars/mon_avatar.vrm
```

Ce script vérifie :
- Que le `.vrm` est un fichier glTF binaire valide
- Que l'extension VRM (`VRMC_vrm` ou `VRM`) est présente
- Que les **15 bones humanoïdes obligatoires** sont mappés :
  - `hips`, `spine`, `head`
  - `leftUpperArm`, `leftLowerArm`, `leftHand`, `rightUpperArm`, `rightLowerArm`, `rightHand`
  - `leftUpperLeg`, `leftLowerLeg`, `leftFoot`, `rightUpperLeg`, `rightLowerLeg`, `rightFoot`

Si l'un manque → le pipeline refusera de lancer (un avatar partiel ne donnerait
pas un retargeting correct).

## Contraintes pour un bon résultat

### Obligatoire
- **15 bones humanoïdes obligatoires** présents (cf. ci-dessus)

### Recommandé pour la LSF
La LSF mobilise fortement les mains et le visage. Pour un rendu fidèle :

- **Bones de doigts** (15 par main = 30 au total) — sans eux, pas de
  fingerspelling. Tous les VRM exportés depuis VRoid Studio les ont.
- **Bone `jaw`** — pour le mouvement de mâchoire (mimics labiaux). Sinon, la
  bouche reste fermée.
- **Bones `leftEye` / `rightEye`** — pour le regard. Optionnel mais améliore
  l'expressivité.
- **Expressions VRM standard** :
  - `aa`, `ih`, `ou`, `ee`, `oh` (voyelles, pour lip-sync — non implémenté en v1)
  - `blink`, `blinkLeft`, `blinkRight` (cligne d'yeux)

### Optionnel
- `chest`, `upperChest` — segments de torse intermédiaires (sinon collapse vers
  `spine`, légère perte de fidélité du dos)
- `neck` — sinon collapse vers `head`
- `leftToes`, `rightToes` — pour les orteils, peu visible en LSF
- Expressions émotionnelles (`happy`, `sad`, etc.) — non utilisées en v1

## Différences VRM 0.x vs VRM 1.0

Le pipeline supporte les deux versions. La principale différence pour nous :

| Aspect | VRM 0.x | VRM 1.0 |
|---|---|---|
| Pouce (3 segments) | `Proximal`, `Intermediate`, `Distal` | `Metacarpal`, `Proximal`, `Distal` |
| Expression `surprised` | Non disponible | Disponible |
| Renames émotions | `joy`, `sorrow`, `fun` | `happy`, `sad`, `relaxed` |

Le code détecte automatiquement la version via l'extension glTF présente
(`VRMC_vrm` pour 1.0, `VRM` pour 0.x) et applique le mapping correct.

## Avatars connus problématiques

- **VRM exportés depuis Blender directement (pas via le VRM Add-on)** :
  souvent il manque la déclaration humanoïde → l'avatar charge mais
  `is_vrm_compatible()` retourne `False`.
- **VRM 0.x très anciens (<2019)** : peuvent avoir des conventions de pose
  différentes (bras vers le bas au lieu de T-pose) ; le retargeting fonctionne
  mais le résultat peut être visuellement décalé.
- **Avatars sans morph targets faciaux** : le visage reste figé même quand
  EMOCA estime des expressions. Pas de blocker, juste un rendu plus statique.

## Cas d'usage : changer d'avatar à la volée dans le viewer

Le `.vrma` produit par le pipeline n'est pas lié à l'avatar avec lequel il
a été retargeté. Tu peux le jouer sur **n'importe quel autre VRM compatible** :

```
http://localhost:5173/?avatar=avatars/avatar_A.vrm&animation=animations/clip.vrma
http://localhost:5173/?avatar=avatars/avatar_B.vrm&animation=animations/clip.vrma  ← même clip, autre avatar
```

C'est rendu possible par `@pixiv/three-vrm-animation::createVRMAnimationClip`
qui résout au runtime les noms de bones humanoïdes vers les nœuds three.js
de l'avatar courant. Les bones cibles absents sur le nouvel avatar sont
loggés comme warnings dans la console mais n'arrêtent pas la lecture.
