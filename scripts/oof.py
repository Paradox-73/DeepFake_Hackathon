"""Score families of fold runs out-of-fold, then their ensembles.
A family is every run named <family>_f<k>. Usage: python scripts/oof.py --runs_dir runs --families a b c"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def family(runs_dir, name):
    parts = []
    for f in sorted(Path(runs_dir).glob(name + "_f[0-9]/val_preds.csv")):
        d = pd.read_csv(f, dtype={"image_id": str})
        p = d.prob.clip(1e-6, 1 - 1e-6)
        d["z"] = np.log(p / (1 - p))
        d["z"] = (d.z - d.z.mean()) / d.z.std()
        d["fold_auc"] = roc_auc_score(d.label, d.prob)
        parts.append(d)
    return pd.concat(parts).set_index("image_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--families", nargs="+", required=True)
    a = ap.parse_args()
    fams = {n: family(a.runs_dir, n) for n in a.families}
    ids = sorted(set.intersection(*[set(f.index) for f in fams.values()]))
    y = fams[a.families[0]].loc[ids, "label"].values
    print("images scored by every family:", len(ids))
    for n, f in fams.items():
        folds = f.fold_auc.groupby(level=0).first().unique()
        print("%-28s OOF AUC %.4f | folds %d, spread %.4f" % (n, roc_auc_score(y, f.loc[ids, "z"]), len(folds), folds.std()))
    for r in range(2, len(fams) + 1):
        for combo in itertools.combinations(a.families, r):
            z = np.mean([fams[n].loc[ids, "z"].values for n in combo], axis=0)
            pr = np.mean([fams[n].loc[ids, "prob"].values for n in combo], axis=0)
            rk = np.mean([pd.Series(fams[n].loc[ids, "prob"].values).rank().values for n in combo], axis=0)
            print("%-60s z %.4f  prob %.4f  rank %.4f" % (
                "+".join(combo), roc_auc_score(y, z), roc_auc_score(y, pr), roc_auc_score(y, rk)))


if __name__ == "__main__":
    main()
