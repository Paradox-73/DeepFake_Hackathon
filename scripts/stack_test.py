"""Three ideas that need no training, all tested with leave-one-fold-out so a gain cannot come from fitting the same images:
1. learn family weights instead of a plain mean
2. let simple image measurements correct the score, since one damage type may sit on a different scale
3. show AUC per kind of damage, to find where the ensemble is weak
Usage: python scripts/stack_test.py --families cfm dflipmix a_rn50"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oof import family  # noqa: E402


def measure(path):
    bgr = cv2.imread(path)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b, gr, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    hist = np.bincount(g.astype(np.uint8).ravel(), minlength=256) / g.size
    return {"sharp": np.log1p(cv2.Laplacian(g, cv2.CV_32F).var()), "speckle": np.abs(g - cv2.medianBlur(g, 3)).mean(),
            "bright": g.mean(), "contrast": g.std(), "colour": (np.abs(r - gr) + np.abs(gr - b)).mean(),
            "redness": (r - b).mean(), "flat": np.sort(hist)[-8:].sum(), "bytes": Path(path).stat().st_size}


def leave_fold_out(make, X, y, folds):
    out = np.zeros(len(y))
    for k in np.unique(folds):
        out[folds == k] = make().fit(X[folds != k], y[folds != k]).predict_proba(X[folds == k])[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--root", default="data/dataset/hackathon_dataset/train")
    a = ap.parse_args()
    fams = [family("runs", n) for n in a.families]
    ids = sorted(fams[0].index)
    y = fams[0].loc[ids, "label"].values
    z = np.stack([f.loc[ids, "z"].values for f in fams], 1)
    folds = pd.read_csv("configs/folds.csv", dtype={"image_id": str}).set_index("image_id").fold.reindex(ids).values
    base = z.mean(1)
    print("plain mean                    %.4f" % roc_auc_score(y, base))
    lr = leave_fold_out(lambda: LogisticRegression(C=1.0), z, y, folds)
    print("learned family weights        %.4f" % roc_auc_score(y, lr), LogisticRegression(C=1.0).fit(z, y).coef_.round(3))

    cache = Path("runs/audit/quality.csv")
    if cache.exists():
        q = pd.read_csv(cache, index_col=0, dtype={0: str})
    else:
        paths = [str(Path(a.root, "fake" if l else "real", i + ".jpg")) for i, l in zip(ids, y)]
        q = pd.DataFrame([measure(p) for p in paths], index=ids)
        cache.parent.mkdir(parents=True, exist_ok=True)
        q.to_csv(cache)
    q = q.loc[ids]
    X = np.c_[z, q.values]
    make = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=150, l2_regularization=5.0,
                                                  min_samples_leaf=100, random_state=42)
    gb_z = leave_fold_out(make, z, y, folds)
    gb = leave_fold_out(make, X, y, folds)
    print("boosting on scores only       %.4f" % roc_auc_score(y, gb_z))
    print("boosting on scores and damage %.4f" % roc_auc_score(y, gb))
    print("mean of plain and boosting    %.4f" % roc_auc_score(y, pd.Series(base).rank().values + pd.Series(gb).rank().values))

    print("\nAUC and mean score by kind of image")
    groups = {
        "grayscale": q.colour < 3,
        "strong colour": q.colour > q.colour.quantile(0.8),
        "very dark": q.bright < 50,
        "very bright": q.bright > 190,
        "low contrast": q.contrast < 30,
        "blurry": q.sharp < q.sharp.quantile(0.1),
        "noisy": q.speckle > q.speckle.quantile(0.9),
        "flat regions": q.flat > q.flat.quantile(0.9),
        "blue cast (inverted?)": q.redness < -5,
        "small file": q["bytes"] < q["bytes"].quantile(0.1),
    }
    for name, m in groups.items():
        m = m.values
        print("%-24s n %5d  AUC %.4f  mean score real %+.2f fake %+.2f" % (
            name, m.sum(), roc_auc_score(y[m], base[m]), base[m & (y == 0)].mean(), base[m & (y == 1)].mean()))
    rest = ~np.any([m.values for m in groups.values()], axis=0)
    print("%-24s n %5d  AUC %.4f" % ("none of the above", rest.sum(), roc_auc_score(y[rest], base[rest])))


if __name__ == "__main__":
    main()
