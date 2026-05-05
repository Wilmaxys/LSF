"""Chargement et validation de config.yaml.

Charge la config par défaut, applique les overrides CLI, retourne un dict
typé. Utilisé par tous les scripts (orchestrateur, extracteurs ML, retargeter).
"""
from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Racine du repo : ../../ par rapport à ce fichier.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Charge config.yaml et retourne un dict.

    Args:
        path : chemin optionnel vers un config alternatif. Par défaut config.yaml
               à la racine du repo.

    Raises:
        FileNotFoundError : config absent
        yaml.YAMLError    : YAML invalide
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable : {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config racine doit être un dict, reçu {type(config)}")
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Merge récursif : les valeurs de `override` écrasent celles de `base`.

    Les dicts sont fusionnés récursivement, les autres types remplacés.
    """
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def resolve_path(p: str | Path) -> Path:
    """Résout un chemin par rapport à la racine du repo s'il est relatif."""
    p = Path(p)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def setup_logging(config: dict) -> None:
    """Configure le logging racine selon la section `logging` du config."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    fmt = log_cfg.get("format", "[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
