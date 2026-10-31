"""Look for wrong training labels: score distribution per label, confident disagreements, photos seen twice.
Usage: python scripts/label_audit.py runs/x/val_preds.csv [more val_preds.csv ...] --sheet runs/audit/suspects.jpg"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--root", default="data/dataset/hackathon_dataset/train")
    ap.add_argument("--sheet", default="runs/audit/suspects.jpg")
    a = ap.parse_args()
    d = pd.concat(pd.read_csv(p, dtype={"image_id": str}) for p in a.preds).groupby("image_id").agg(
        label=("label", "first"), prob=("prob", "mean")).reset_index()
    y, p = d.label.values, d.prob.values
    print("images %d, AUC %.4f" % (len(d), roc_auc_score(y, p)))
    bins = np.linspace(0, 1, 11)
    print("P(fake) bin    real  fake")
    for lo, hi in zip(bins[:-1], bins[1:]):
        print("%.1f-%.1f   %6d %6d" % (lo, hi, ((p >= lo) & (p < hi) & (y == 0)).sum(), ((p >= lo) & (p < hi) & (y == 1)).sum()))

    folds = pd.read_csv("configs/folds.csv", dtype={"image_id": str})
    m = d.merge(folds[["image_id", "group"]], on="image_id")
    sizes = folds.groupby("group").size()
    twos = sizes[sizes == 2].index
    pairs = [g for g in m[m.group.isin(twos)].groupby("group") if len(g[1]) == 2]
    same, diff = [], []
    for _, g in pairs:
        (same if g.label.nunique() == 1 else diff).append(abs(g.prob.iloc[0] - g.prob.iloc[1]))
    print("pairs in the scored set: same label %d, mean score gap %.3f | different label %d, mean score gap %.3f" % (
        len(same), np.mean(same) if same else 0, len(diff), np.mean(diff) if diff else 0))
    if diff:
        print("different-label pairs where the model scored both on one side of 0.5: %d of %d" % (
            sum((g.prob.iloc[0] > 0.5) == (g.prob.iloc[1] > 0.5) for _, g in pairs if g.label.nunique() == 2), len(diff)))

    e = d.assign(gap=np.abs(y - p)).sort_values("gap", ascending=False).head(32)
    tiles = []
    for r in e.itertuples():
        img = cv2.imread(str(Path(a.root) / ("fake" if r.label else "real") / (r.image_id + ".jpg")))
        cv2.putText(img, "%s p=%.2f" % ("FAKE" if r.label else "REAL", r.prob), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (168, 168)))
    cv2.imwrite(a.sheet, np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 32, 8)]), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
