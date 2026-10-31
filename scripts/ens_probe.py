"""Greedy forward selection over finished runs on one fold, plus a look at confident mistakes.
Usage: python scripts/ens_probe.py runs/a runs/b ..."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load(run):
    d = pd.read_csv(Path(run) / "val_preds.csv", dtype={"image_id": str}).set_index("image_id").sort_index()
    p = d.prob.clip(1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    d["z"] = (z - z.mean()) / z.std()
    return d


runs = {Path(r).name: load(r) for r in sys.argv[1:]}
y = next(iter(runs.values())).label.values
names = list(runs)
print("single model AUC:", {n: round(roc_auc_score(y, runs[n].z), 4) for n in names})
corr = pd.DataFrame({n: runs[n].z for n in names}).corr()
print("mean correlation between models: %.3f" % corr.values[np.triu_indices(len(names), 1)].mean())
print("ensemble of all, z mean: %.4f" % roc_auc_score(y, np.mean([runs[n].z for n in names], axis=0)))
print("ensemble of all, prob mean: %.4f" % roc_auc_score(y, np.mean([runs[n].prob for n in names], axis=0)))

chosen, best = [], 0
while len(chosen) < len(names):
    s = {n: roc_auc_score(y, np.mean([runs[m].z for m in chosen + [n]], axis=0)) for n in names if n not in chosen}
    n = max(s, key=s.get)
    chosen.append(n)
    print("add %-26s -> %.4f" % (n, s[n]))

z = np.mean([runs[n].z for n in names], axis=0)
conf = 1 / (1 + np.exp(-z))
wrong = (conf > 0.5).astype(int) != y
print("accuracy at 0.5: %.4f" % (1 - wrong.mean()))
for thr in (0.8, 0.9, 0.95):
    sure = (conf > thr) | (conf < 1 - thr)
    print("confidence above %.2f: %d images, %.1f%% wrong (%d)" % (thr, sure.sum(), 100 * wrong[sure].mean(), wrong[sure].sum()))
print("wrong per class, real %d, fake %d" % (wrong[y == 0].sum(), wrong[y == 1].sum()))
