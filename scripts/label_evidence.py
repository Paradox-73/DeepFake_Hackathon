"""Test whether confident model/label disagreements are wrong labels or just hard images.

Test 1, no model involved: list candidate pairs of one photo. 64x64 cannot tell a copy from a swap, which keeps
pose, so scripts/exact_copies.py makes that call at full resolution from the CSV written here.
Test 2: true label flips hit clean and damaged images alike. Hard images pile up among the damaged ones.
Test 3: a fake and its source photo share a fold. If labels are right the fake should score higher than the source.
Only training images are read.
Usage: python scripts/label_evidence.py --families cfm dflipmix sfm"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gray(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def inner_mask(n):
    yy, xx = np.mgrid[:n, :n] / (n - 1) - 0.5
    return (xx / 0.30) ** 2 + (yy / 0.38) ** 2 <= 1  # shortcut: fixed oval, faces are pre-aligned by the organisers


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def compare_pair(a, b, mask):
    """Correlation inside the face oval and outside it, on 64x64 images, undoing inversion."""
    a, b = [cv2.resize(i, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() for i in (a, b)]
    if corr(a, b) < 0:
        b = 255 - b
    return corr(a[mask], b[mask]), corr(a[~mask], b[~mask])


def quality(path):
    g = gray(path).astype(np.float32)
    med = cv2.medianBlur(g, 3)
    return (np.log1p(cv2.Laplacian(g, cv2.CV_32F).var()), np.abs(g - med).mean(), g.mean(), Path(path).stat().st_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset/hackathon_dataset")
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--pairs_min", type=float, default=0.80, help="minimum outside-face correlation to call two images one photo")
    ap.add_argument("--same_face", type=float, default=0.03, help="largest outside-minus-inside gap for 'same face'")
    ap.add_argument("--out", default="runs/audit/label_evidence.csv")
    a = ap.parse_args()

    fams = [pd.concat(pd.read_csv(f, dtype={"image_id": str}) for f in sorted(Path("runs").glob(n + "_f[0-9]/val_preds.csv")))
            .set_index("image_id").sort_index() for n in a.families]
    assert all(len(f) == 7000 for f in fams)
    d = fams[0][["label"]].copy()
    d["prob"] = np.mean([f.prob.values for f in fams], axis=0)
    d["path"] = [str(Path(a.root, "train", "fake" if l else "real", i + ".jpg")) for i, l in zip(d.index, d.label)]
    d["suspect"] = ((d.label == 0) & (d.prob >= a.thr)) | ((d.label == 1) & (d.prob <= 1 - a.thr))
    folds = pd.read_csv("configs/folds.csv", dtype={"image_id": str}).set_index("image_id")
    d["group"] = folds.group.reindex(d.index)

    print("== Test 1: pairs of one photo, no model involved")
    mask = inner_mask(64).ravel()
    rows = []
    for _, g in d.groupby("group"):
        if len(g) < 2:
            continue
        ims = [gray(p) for p in g.path]
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                ins, out = compare_pair(ims[i], ims[j], mask)
                rows.append((g.index[i], g.index[j], g.label.iloc[i], g.label.iloc[j], ins, out, g.prob.iloc[i], g.prob.iloc[j]))
    p = pd.DataFrame(rows, columns="id_a id_b label_a label_b inside outside prob_a prob_b".split())
    p["gap"], p["mixed"] = p.outside - p.inside, p.label_a != p.label_b
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    p.to_csv(a.out, index=False)
    photo = p[p.outside >= a.pairs_min]
    print("pairs in near-duplicate groups %d, one photo (outside corr >= %.2f) %d" % (len(p), a.pairs_min, len(photo)))
    for name, s in (("same face", photo[photo.gap <= a.same_face]), ("different face", photo[photo.gap > a.same_face])):
        rate = s.mixed.mean() if len(s) else float("nan")
        print("  %-15s %5d pairs, opposite labels %4d (%.1f%%)" % (name, len(s), s.mixed.sum(), 100 * rate))

    print("== Test 2: suspect rate by image damage")
    q = pd.DataFrame([quality(x) for x in d.path], index=d.index, columns=["sharp", "speckle", "bright", "bytes"])
    for c in q:
        b = pd.qcut(q[c], 5, labels=False, duplicates="drop")
        cells = []
        for k in sorted(b.unique()):
            s = d[b == k]
            cells.append("%.1f%%/%.3f" % (100 * s.suspect.mean(), roc_auc_score(s.label, s.prob)))
        print("  %-8s low to high, suspect/AUC: %s" % (c, "  ".join(cells)))

    print("== Test 3: real and fake copies of one photo, different faces")
    cross = photo[photo.mixed & (photo.gap > a.same_face)]
    fake_p = np.where(cross.label_a == 1, cross.prob_a, cross.prob_b)
    real_p = np.where(cross.label_a == 1, cross.prob_b, cross.prob_a)
    if len(cross):
        print("  pairs %d, fake copy scored higher in %d (%.1f%%)" % (len(cross), (fake_p > real_p).sum(), 100 * (fake_p > real_p).mean()))


if __name__ == "__main__":
    main()
