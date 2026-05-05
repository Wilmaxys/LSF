# Modèles ML — emplacement local

Ce dossier reçoit tous les poids ML après téléchargement par
`scripts/download_weights.sh`.

**Aucun fichier de ce dossier (sauf ce README et CHECKSUMS.sha256) n'est
versionné** : ils sont trop volumineux et soumis à licences non-commerciales
qui interdisent la redistribution.

## Structure attendue

```
pipeline/models/
├── README.md                    (ce fichier)
├── CHECKSUMS.sha256             (généré au premier téléchargement)
├── smplerx/
│   ├── smpler_x_h32_correct.pth.tar   (~2.6 GB)
│   └── … autres variants si téléchargés
├── mmdet/
│   └── faster_rcnn_r50_fpn_1x_coco.pth (~160 MB)
├── vitpose/
│   └── … (poids ViT-S/B/L/H)
├── hamer/
│   ├── checkpoint.ckpt
│   └── model_final_f05665.pkl   (ViTDet, ~2.6 GB)
├── emoca/
│   └── assets/
│       ├── EMOCA/models/EMOCA_v2_lr_mse_20/
│       ├── DECA/
│       ├── FaceRecognition/
│       └── FLAME/
├── smplx/                       (modèles corps SMPL-X)
│   ├── SMPLX_NEUTRAL.npz
│   ├── SMPLX_MALE.npz
│   ├── SMPLX_FEMALE.npz
│   ├── MANO_SMPLX_vertex_ids.pkl
│   ├── SMPL-X__FLAME_vertex_ids.npy
│   └── SMPLX_to_J14.pkl
├── smpl/
│   └── SMPL_NEUTRAL.pkl … etc.
└── mano/
    └── MANO_RIGHT.pkl
```

## Comptes à créer (téléchargements manuels)

Avant de lancer `download_weights.sh`, créer un compte gratuit et accepter
la licence sur ces sites :

1. https://smpl-x.is.tue.mpg.de
2. https://smpl.is.tue.mpg.de
3. https://mano.is.tue.mpg.de
4. https://flame.is.tue.mpg.de
5. https://emoca.is.tue.mpg.de

## Licences

Tous les modèles sont sous licence **non-commerciale**. Voir
[docs/MODELS.md](../../docs/MODELS.md) pour le détail.
