"""Draw same-photo pairs with opposite labels, closest face match first: real copy, fake copy, amplified difference.
Also prints how the opposite-label rate moves with the thresholds, so a single cut-off is not doing the work.
Usage: python scripts/show_label_conflicts.py --n 24 --out runs/audit/label_conflicts.jpg"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="runs/audit/label_evidence.csv")
ap.add_argument("--root", default="data/dataset/hackathon_dataset/train")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--out", default="runs/audit/label_conflicts.jpg")
a = ap.parse_args()

p = pd.read_csv(a.pairs, dtype={"id_a": str, "id_b": str})
print("outside  gap<=   pairs  opposite  rate   flip rate if labels flip independently")
for out in (0.8, 0.9, 0.95):
    for gap in (0.005, 0.01, 0.02, 0.03):
        s = p[(p.outside >= out) & (p.gap <= gap)]
        r = s.mixed.mean() if len(s) else float("nan")
        print("%.2f     %.3f  %5d  %5d   %5.1f%%  %5.1f%%" % (out, gap, len(s), s.mixed.sum(), 100 * r, 50 * (1 - np.sqrt(max(0, 1 - 2 * r)))))

c = p[(p.outside >= 0.8) & p.mixed].sort_values("gap").head(a.n)
size, rows = 112, []
for r in c.itertuples():
    ids = (r.id_a, r.id_b) if r.label_a == 0 else (r.id_b, r.id_a)
    ims = [cv2.resize(cv2.imread(str(Path(a.root, k, i + ".jpg"))), (size, size)) for k, i in zip(("real", "fake"), ids)]
    diff = np.clip(cv2.absdiff(*ims).astype(np.float32) * 3, 0, 255).astype(np.uint8)
    rows.append(np.hstack(ims + [diff]))
while len(rows) % 3:
    rows.append(np.zeros_like(rows[0]))
cv2.imwrite(a.out, np.vstack([np.hstack(rows[i:i + 3]) for i in range(0, len(rows), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 92])
print("wrote", a.out, "each cell: labelled real, labelled fake, difference x3")
