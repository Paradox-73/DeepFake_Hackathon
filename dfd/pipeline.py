"""Glue shared by the CLI scripts."""
from .adapters import build_samples
from .prepare import prepare_samples


def load_section(cfg, key="data"):
    """Build + prepare samples for cfg[key] ('data' or 'test'). Returns (samples, failed_srcs)."""
    sec = cfg[key]
    base = cfg.get("data", {})
    pcfg = {"video": sec.get("video", base.get("video", {})), "face": sec.get("face", base.get("face", {})),
            "prepare_workers": sec.get("prepare_workers", base.get("prepare_workers", 1))}
    cache = sec.get("cache_dir", base.get("cache_dir", "cache"))
    return prepare_samples(build_samples(sec), pcfg, cache)
