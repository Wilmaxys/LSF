"""Wrapper minimal autour du repo SMPLer-X (cloné dans pipeline/envs/smplerx/repo/).

Isole l'API spécifique de SMPLer-X (yacs/mmcv config + Demoer base class) du
reste du pipeline.

API vérifiée le 2026-05-05 sur le commit `064baef0e4ab5277a3297691bc1d46ea5412586f`
de https://github.com/MotrixLab/SMPLer-X par lecture directe du code source :
    - main/inference.py (flow d'inférence canonique)
    - main/SMPLer_X.py (forward + clés de sortie)
    - common/base.py (chargement du checkpoint, Demoer)
    - common/utils/preprocessing.py (crop bbox + generate_patch_image)
"""
from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Le repo SMPLer-X est cloné par scripts/setup.sh dans :
SMPLERX_REPO = Path(__file__).resolve().parent / "repo"


def load_model(weights_path: Path, model_name: str):
    """Charge le modèle SMPLer-X depuis les poids fournis.

    Reproduit le flow de main/inference.py : config_fromfile + update_test_config
    + Demoer._make_model() + load_state_dict avec préfixe 'module.' et renames.

    Args:
        weights_path : .pth.tar
        model_name   : "s32" | "b32" | "l32" | "h32" | "h32_correct"

    Returns:
        Une instance Demoer prête pour l'inférence (.model.eval(), sur GPU).
    """
    if not SMPLERX_REPO.exists():
        raise FileNotFoundError(
            f"Repo SMPLer-X non cloné : {SMPLERX_REPO}. Lancer scripts/setup.sh"
        )

    # Path setup pour pouvoir importer main.* et common.*
    sys.path.insert(0, str(SMPLERX_REPO / "main"))
    sys.path.insert(0, str(SMPLERX_REPO / "common"))
    sys.path.insert(0, str(SMPLERX_REPO))

    import torch
    from main.config import cfg  # type: ignore[import-not-found]
    from base import Demoer  # type: ignore[import-not-found]

    # 1. Charger la config Python du variant — convention du repo :
    # main/config/config_smpler_x_<size>.py (ex : config_smpler_x_h32.py).
    # Le suffixe "_correct" partage la config "h32".
    config_size = model_name.replace("_correct", "")
    config_file = SMPLERX_REPO / "main" / "config" / f"config_smpler_x_{config_size}.py"
    if not config_file.exists():
        raise FileNotFoundError(f"Config SMPLer-X introuvable : {config_file}")
    cfg.get_config_fromfile(str(config_file))
    cfg.update_test_config(
        testset="EHF",
        agora_benchmark="na",
        shapy_eval_split=None,
        pretrained_model_path=str(weights_path),
        use_cache=False,
    )

    # 2. Instancier Demoer (helper interne du repo qui possède .model)
    demoer = Demoer()
    demoer._make_model()

    # 3. Charger le checkpoint en respectant la convention :
    #    - clé top-level = 'network'
    #    - ajouter 'module.' aux clés sans le préfixe (DataParallel)
    #    - renommer backbone→encoder, body_rotation_net→body_regressor,
    #      hand_rotation_net→hand_regressor (cf. common/base.py)
    ckpt = torch.load(str(weights_path), map_location="cpu")
    state_dict = ckpt["network"] if "network" in ckpt else ckpt

    new_state_dict: OrderedDict = OrderedDict()
    for k, v in state_dict.items():
        if "module" not in k:
            k = "module." + k
        k = k.replace("backbone", "encoder")
        k = k.replace("body_rotation_net", "body_regressor")
        k = k.replace("hand_rotation_net", "hand_regressor")
        new_state_dict[k] = v

    demoer.model.load_state_dict(new_state_dict, strict=False)
    demoer.model.eval()
    logger.info("SMPLer-X chargé : %s (%s)", weights_path.name, model_name)
    return demoer


def infer(demoer, frame_bgr: np.ndarray, bbox: np.ndarray):
    """Inférence sur une frame + bbox de personne (xyxy en pixels).

    Reproduit le pré-processing canonique de main/inference.py :
    process_bbox + generate_patch_image vers cfg.input_img_shape (typ. 512×384),
    puis division par 255 en RGB float.

    Args:
        demoer    : Demoer chargé par load_model()
        frame_bgr : (H, W, 3) uint8 BGR (sortie cv2)
        bbox      : (4,) xyxy en pixels

    Returns:
        params_dict : np arrays float32 — voir clés ci-dessous
        reproj_residual_pixels : float (proxy ; SMPLer-X n'expose pas un score natif)

    Clés de params_dict :
        transl(3,), global_orient(3,), body_pose(63,),
        left_hand_pose(45,), right_hand_pose(45,),
        jaw_pose(3,), leye_pose(3,), reye_pose(3,),
        expression(10,), betas(10,)
    """
    import sys
    sys.path.insert(0, str(SMPLERX_REPO / "common"))
    sys.path.insert(0, str(SMPLERX_REPO / "main"))

    import torch
    from main.config import cfg  # type: ignore[import-not-found]
    from common.utils.preprocessing import (  # type: ignore[import-not-found]
        process_bbox, generate_patch_image,
    )
    from torchvision import transforms

    # 1. bbox xyxy → xywh, puis process_bbox (ratio + clipping)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bbox_xywh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
    H, W = frame_bgr.shape[:2]
    bbox_processed = process_bbox(bbox_xywh, W, H)
    if bbox_processed is None:
        # Bbox invalide après processing — retour zéros + résidu max
        return _zero_params(), 999.0

    # 2. Crop en cfg.input_img_shape (typ. (512, 384) = (H, W))
    img_patch, img2bb_trans, bb2img_trans = generate_patch_image(
        frame_bgr.copy(), bbox_processed, scale=1.0, rot=0.0, do_flip=False,
        out_shape=cfg.input_img_shape,
    )
    # Convention SMPLer-X : RGB float ∈ [0, 1], pas de normalisation imagenet
    transform = transforms.ToTensor()
    img_tensor = transform(img_patch.astype(np.float32)).cuda()[None, ...] / 255.0

    # 3. Forward en mode test (Demoer.model est wrappé DataParallel → .module)
    inputs = {"img": img_tensor}
    targets: dict = {}
    meta_info: dict = {}
    with torch.no_grad():
        out = demoer.model(inputs, targets, meta_info, "test")

    def _to_np(t):
        return t.detach().cpu().numpy().squeeze(0).astype(np.float32)

    params = {
        "transl":          _to_np(out["cam_trans"]),
        "global_orient":   _to_np(out["smplx_root_pose"]),
        "body_pose":       _to_np(out["smplx_body_pose"]),
        "left_hand_pose":  _to_np(out["smplx_lhand_pose"]),
        "right_hand_pose": _to_np(out["smplx_rhand_pose"]),
        "jaw_pose":        _to_np(out["smplx_jaw_pose"]),
        # SMPLer-X n'estime pas explicitement les yeux ; init neutre.
        "leye_pose":       np.zeros(3, np.float32),
        "reye_pose":       np.zeros(3, np.float32),
        "expression":      _to_np(out["smplx_expr"]),
        "betas":           _to_np(out["smplx_shape"]),
    }

    # 4. Résidu de reprojection — SMPLer-X expose smplx_joint_proj mais on n'a
    # pas de GT 2D pour comparer ; renvoie 0 (la confidence dérive surtout du
    # score bbox du détecteur en amont).
    reproj_residual = 0.0
    return params, reproj_residual


def _zero_params() -> dict:
    """Retourne un dict de paramètres à zéro (utilisé en cas de bbox invalide)."""
    return {
        "transl":          np.zeros(3, np.float32),
        "global_orient":   np.zeros(3, np.float32),
        "body_pose":       np.zeros(63, np.float32),
        "left_hand_pose":  np.zeros(45, np.float32),
        "right_hand_pose": np.zeros(45, np.float32),
        "jaw_pose":        np.zeros(3, np.float32),
        "leye_pose":       np.zeros(3, np.float32),
        "reye_pose":       np.zeros(3, np.float32),
        "expression":      np.zeros(10, np.float32),
        "betas":           np.zeros(10, np.float32),
    }
