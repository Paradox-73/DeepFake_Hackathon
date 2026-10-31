"""Paired comparison of run families against a baseline, on the same validation folds and the same noisy labels.
Usage: python scripts/compare.py --base deit dflip85 ddrop85 ..."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def fold_auc(path):
    d = pd.read_csv(path)
    return roc_auc_score(d.label, d.prob)


def robust(run):
    f = Path(run) / "robust_val.json"
    if not f.exists():
        return np.nan
    rows = [r for r in json.loads(f.read_text()) if r["name"] != "clean"]
    return float(np.mean([r["auc"] for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    for n in a.names:
        gains, rob = [], []
        for f in sorted(Path("runs").glob(n + "_f[0-9]/val_preds.csv")):
            b = Path("runs") / (a.base + "_f" + f.parent.name[-1])
            gains.append(fold_auc(f) - fold_auc(b / "val_preds.csv"))
            rob.append(robust(f.parent) - robust(b))
        print("%-14s folds %d | AUC gain vs %s %+.4f (per fold %s) | robust gain %+.4f" % (
            n, len(gains), a.base, np.mean(gains), " ".join("%+.4f" % g for g in gains), np.mean(rob)))


if __name__ == "__main__":
    main()
