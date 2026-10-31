"""Write the prediction CSV. Usage:
python predict.py --config configs/default.yaml --ckpt runs/exp/best.pt --out predictions.csv"""
import argparse
import csv
from pathlib import Path

from dfd.config import load_config
from dfd.data import aggregate
from dfd.engine import infer
from dfd.metrics import compute_metrics
from dfd.models import load_checkpoint
from dfd.pipeline import load_section
from dfd.utils import get_logger, pick_device, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    pc = cfg.get("predict", {})
    log = get_logger()
    seed_everything(cfg.get("seed", 42), cfg.get("deterministic", True))
    device = pick_device(cfg.get("device", "auto"))
    model, ck = load_checkpoint(a.ckpt, device)
    thr = pc.get("threshold") if pc.get("threshold") is not None else ck["threshold"]

    samples, failed = load_section(cfg, "test")
    probs = infer(model, samples, ck["img_size"], device, pc.get("batch_size", 64), pc.get("num_workers", 0),
                  pc.get("amp", True), tuple(pc.get("tta", [])), desc="predict")
    srcs, p, labels = aggregate(samples, probs, pc.get("aggregate", "mean"))

    idm = pc.get("id_mode", "name")  # name | stem | src

    def ident(s):
        return Path(s).name if idm == "name" else Path(s).stem if idm == "stem" else s

    names = {int(k): v for k, v in pc.get("label_names", {0: 0, 1: 1}).items()}
    cols = {"id": "id", "label": "label", "prob": "probability"}
    cols.update(pc.get("columns", {}))
    rows = [(ident(s), names[int(x >= thr)], x) for s, x in zip(srcs, p)]
    fb = names[int(pc.get("fallback_label", 1))]
    rows += [(ident(s), fb, 0.5) for s in failed]  # undecodable inputs still get a row
    if pc.get("sort", True):
        rows.sort(key=lambda r: r[0])
    with_prob = bool(cols.get("prob"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if pc.get("header", True):
            w.writerow([cols["id"], cols["label"]] + ([cols["prob"]] if with_prob else []))
        for i, lab, x in rows:
            w.writerow([i, lab] + (["%.6f" % x] if with_prob else []))
    log.info("Wrote %d rows to %s (threshold %.2f, tta=%s, %d undecodable)", len(rows), a.out, thr,
             pc.get("tta", []), len(failed))
    if (labels >= 0).all():
        log.info("Test labels present -> metrics: %s", compute_metrics(labels, p, thr))


if __name__ == "__main__":
    main()
