import json
import logging
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_logger(name="dfd", log_file=None):
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def env_info() -> dict:
    import timm, torchvision, cv2
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "timm": timm.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_callable(spec: str):
    """Load 'path/to/file.py:func' or 'package.module:func'."""
    import importlib
    import importlib.util
    target, _, fn = spec.rpartition(":")
    if not target or not fn:
        raise ValueError(f"Bad callable spec '{spec}', expected 'module_or_file.py:function'")
    if target.endswith(".py"):
        p = Path(target).resolve()
        s = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(s)
        sys.modules[p.stem] = mod
        s.loader.exec_module(mod)
    else:
        sys.path.insert(0, os.getcwd())
        mod = importlib.import_module(target)
    return getattr(mod, fn)
