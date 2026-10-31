import copy
from pathlib import Path

import yaml


def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path, overrides=None) -> dict:
    """Load YAML. If it has `base: other.yaml` that file is loaded first.
    overrides: list like ['train.epochs=3', 'model.arch=convnext_nano']."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        bp = Path(base)
        if not bp.is_absolute() and not bp.exists():
            bp = path.parent / base
        cfg = deep_update(load_config(bp), cfg)
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg
