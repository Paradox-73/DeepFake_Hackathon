"""List training images whose label strongly disagrees with out-of-fold predictions.
Each image is scored only by models that never trained on it. Output CSV: image_id, label, prob, new_label.
flip and drop act on suspects only. soft rewrites every image to (1 - alpha) * label + alpha * prob.
Usage: python scripts/clean_labels.py --families cxd deit --thr 0.85 --mode flip --out configs/relabel_flip85.csv"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--mode", choices=["flip", "drop", "soft"], default="flip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    a = ap.parse_args()
    fams = []
    for name in a.families:
        files = sorted(Path(a.runs_dir).glob(name + "_f[0-9]/val_preds.csv"))
        assert len(files) == a.n_folds, "%s has %d of %d folds" % (name, len(files), a.n_folds)
        fams.append(pd.concat(pd.read_csv(f, dtype={"image_id": str}) for f in files).set_index("image_id").sort_index())
    y = fams[0].label
    p = np.mean([f.prob.values for f in fams], axis=0)
    print("out-of-fold images %d, AUC %.4f, accuracy %.4f" % (len(y), roc_auc_score(y, p), ((p > 0.5) == y).mean()))
    real_bad = (y == 0) & (p >= a.thr)
    fake_bad = (y == 1) & (p <= 1 - a.thr)
    print("suspect labels: %d of %d real (%.1f%%), %d of %d fake (%.1f%%)" % (
        real_bad.sum(), (y == 0).sum(), 100 * real_bad.sum() / (y == 0).sum(),
        fake_bad.sum(), (y == 1).sum(), 100 * fake_bad.sum() / (y == 1).sum()))
    d = pd.DataFrame({"image_id": y.index, "label": y.values, "prob": p})
    if a.mode == "soft":
        d["new_label"] = ((1 - a.alpha) * d.label + a.alpha * d.prob).round(4)
    else:
        d = d[(real_bad | fake_bad).values]
        d["new_label"] = (1 - d.label) if a.mode == "flip" else -1
    d.to_csv(a.out, index=False)
    print("wrote", a.out, len(d))


if __name__ == "__main__":
    main()
