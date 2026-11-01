"""Pack trained fold checkpoints into the single model_weights.pt that inference_script.py reads.
Standardisation statistics and the threshold come from out-of-fold validation predictions only.
An all-data run has no unseen images, so --stats_from fullc=cfm takes its statistics from the five cfm fold runs.
Usage: python scripts/build_bundle.py --runs runs/a_f0 runs/a_f1 ... --out model_weights.pt [--flip_tta] [--stats_from PREFIX=FAMILY ...]"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


def logits(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    p = np.clip(np.array([float(r["prob"]) for r in rows]), 1e-6, 1 - 1e-6)
    return {r["image_id"]: x for r, x in zip(rows, np.log(p / (1 - p)))}, {r["image_id"]: int(r["label"]) for r in rows}


def stats_runs(run, stats_from):
    run = Path(run)
    for pair in stats_from:
        prefix, family = pair.split("=")
        if run.name.startswith(prefix):
            return [run.parent / f"{family}_f{k}" for k in range(5)]
    return [run]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="model_weights.pt")
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--stats_from", nargs="*", default=[])
    a = ap.parse_args()

    members, oof, labels = [], {}, {}
    for run in a.runs:
        ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
        lg, lab = {}, {}
        for src in stats_runs(run, a.stats_from):
            part_lg, part_lab = logits(src / "val_preds.csv")
            lg.update(part_lg)
            lab.update(part_lab)
        v = np.array(list(lg.values()))
        mean, std = float(v.mean()), float(v.std())
        members.append({"arch": ck["arch"], "img_size": ck["img_size"], "mean": ck["mean"], "std": ck["std"],
                        "view": ck.get("view", "rgb"), "global_pool": ck.get("global_pool"),
                        "logit_mean": mean, "logit_std": std, "run": run,
                        "state_dict": {k: x.half() if x.is_floating_point() else x for k, x in ck["state_dict"].items()}})
        labels.update(lab)
        for k, x in lg.items():
            oof.setdefault(k, []).append((x - mean) / std)

    ids = sorted(oof)
    y = np.array([labels[k] for k in ids])
    conf = 1 / (1 + np.exp(-np.array([np.mean(oof[k]) for k in ids]) * a.temperature))
    grid = np.linspace(0.05, 0.95, 181)
    f1 = [f1_score(y, conf >= t) for t in grid]
    thr = float(grid[int(np.argmax(f1))])
    print("members %d | out-of-fold images %d | threshold %.3f | F1 %.4f" % (len(members), len(ids), thr, max(f1)))
    torch.save({"members": members, "threshold": thr, "flip_tta": a.flip_tta, "temperature": a.temperature}, a.out)
    print("saved %s, %.0f MB" % (a.out, Path(a.out).stat().st_size / 1e6))


if __name__ == "__main__":
    main()
