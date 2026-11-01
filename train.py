"""Train. Usage: python train.py --config configs/default.yaml [--set train.epochs=3 data.root=/x]"""
import argparse
import csv
from pathlib import Path

import yaml

from dfd.config import load_config
from dfd.data import relabel, split_by_fold, split_by_src
from dfd.engine import evaluate, train
from dfd.models import load_checkpoint
from dfd.pipeline import load_section
from dfd.utils import env_info, get_logger, pick_device, save_json, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.epochs=3")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    out = Path(cfg.get("output_dir", "runs/exp"))
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(log_file=out / "train.log")
    seed_everything(cfg.get("seed", 42), cfg.get("deterministic", True))
    device = pick_device(cfg.get("device", "auto"))
    log.info("Device: %s | env: %s", device, env_info())
    with open(out / "config_resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    save_json(env_info(), out / "env.json")

    samples, failed = load_section(cfg, "data")
    if cfg["data"].get("folds_csv"):
        tr, va, tr_src, va_src = split_by_fold(samples, cfg["data"]["folds_csv"], cfg["data"]["fold"])
    else:
        tr, va, tr_src, va_src = split_by_src(samples, cfg["data"].get("val_fraction", 0.15), cfg.get("seed", 42))
    if cfg["data"].get("full"):  # final models train on every image, so the fold's metrics mean nothing here
        tr = tr + va
    if cfg["data"].get("relabel_csv"):
        tr = relabel(tr, cfg["data"]["relabel_csv"])
    save_json({"train": tr_src, "val": va_src}, out / "split.json")
    log.info("Items: %d train / %d val (frames/images: %d / %d)", len(tr_src), len(va_src), len(tr), len(va))

    res = train(cfg, tr, va, out, device)
    model, ck = load_checkpoint(out / "best.pt", device)
    ev = {"batch_size": 64, "num_workers": cfg["train"].get("num_workers", 0), "amp": cfg["train"].get("amp", True),
          "aggregate": cfg.get("predict", {}).get("aggregate", "mean")}
    srcs, vp, vl, m = evaluate(model, va, cfg["model"]["img_size"], device, ev, ck["threshold"])
    log.info("BEST epoch %d | val @thr %.2f: %s", res["best_epoch"], ck["threshold"], m)
    with open(out / "val_preds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label", "prob"])
        w.writerows((Path(s).stem, int(y), "%.6f" % p) for s, y, p in zip(srcs, vl, vp))
    save_json({"best_epoch": res["best_epoch"], "train_seconds": res["train_seconds"], "val": m,
               "history": res["history"]}, out / "metrics.json")
    log.info("Saved %s", out / "best.pt")


if __name__ == "__main__":
    main()
