"""Pipeline LSF — vidéo → animation 3D VRM.

Modules disponibles dans tous les envs (pure-Python, sans torch) :
    - animation_npz       : I/O et validation du format animation.npz
    - smplx_to_vrm_mapping: tables statiques de mapping bones SMPL-X → VRM
    - flame_to_vrm_mapping: logique de mapping FLAME → expressions VRM
    - vrm_inspector       : inspection dynamique d'un fichier .vrm
    - one_euro_filter     : implémentation du filtre One-Euro
    - smoothing           : lissage temporel des paramètres SMPL-X
    - confidence          : calcul de la confidence par frame (proxy)
    - debug_overlay       : génération de la vidéo de debug avec mesh

Modules par-env (chargent torch et les modèles ML, lancés en sous-process) :
    - envs/smplerx/extract_smplerx.py
    - envs/hamer/extract_hamer.py
    - envs/emoca/extract_emoca.py

Orchestration :
    - pipeline.py : point d'entrée principal (--video, --avatar, --output)
    - extract.py  : dispatcher des extractions ML
    - retarget.py : script Blender headless de retargeting → .vrma
"""

__version__ = "0.1.0"
