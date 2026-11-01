"""Separate exact copies of a photo from face swaps at full resolution.
A swap keeps pose, so 64x64 correlation cannot tell it from a copy, but it replaces the skin texture inside the face.
Copy: high-pass correlation inside the face about equals outside. Swap: inside falls well below outside.
Usage: python scripts/exact_copies.py"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

root = "data/dataset/hackathon_dataset/train"
p = pd.read_csv("runs/audit/label_evidence.csv", dtype={"id_a": str, "id_b": str})
p = p[p.outside >= 0.8].copy()
yy, xx = np.mgrid[:224, :224] / 223 - 0.5
inner = ((xx / 0.30) ** 2 + (yy / 0.38) ** 2 <= 1).ravel()
outer = ((xx / 0.36) ** 2 + (yy / 0.46) ** 2 > 1).ravel()  # leave a ring out, the blend edge of a swap sits there


def hp(i, l):
    g = cv2.cvtColor(cv2.imread(str(Path(root, "fake" if l else "real", i + ".jpg"))), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return (g - cv2.GaussianBlur(g, (0, 0), 3)).ravel()


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


rows = []
for r in p.itertuples():
    a, b = hp(r.id_a, r.label_a), hp(r.id_b, r.label_b)
    if corr(a, b) < 0:
        b = -b
    rows.append((corr(a[inner], b[inner]), corr(a[outer], b[outer])))
p[["hp_in", "hp_out"]] = rows
p["hp_gap"] = p.hp_out - p.hp_in
p["both_high"], p["both_low"] = (p.prob_a > 0.5) & (p.prob_b > 0.5), (p.prob_a < 0.5) & (p.prob_b < 0.5)
p.to_csv("runs/audit/exact_copies.csv", index=False)
s = p[p.hp_out >= 0.6]
print("pairs with matching fine detail outside the face (hp_out >= 0.6):", len(s))
edges = [-1, 0.02, 0.05, 0.1, 0.2, 0.4, 2]
print("texture gap     pairs  opposite labels   of opposite: both models-scored same side")
for lo, hi in zip(edges, edges[1:]):
    c = s[(s.hp_gap > lo) & (s.hp_gap <= hi)]
    m = c[c.mixed]
    side = (m.both_high | m.both_low).sum()
    print("%5.2f to %5.2f  %5d  %5d (%5.1f%%)   %d of %d" % (lo, hi, len(c), len(m), 100 * c.mixed.mean() if len(c) else 0, side, len(m)))
copies = s[s.hp_gap <= 0.05]
n, m = len(copies), int(copies.mixed.sum())
if n:
    r = m / n
    print("exact copies %d, opposite labels %d (%.1f%%), implied independent flip rate %.1f%%" % (n, m, 100 * r, 50 * (1 - np.sqrt(max(0, 1 - 2 * r)))))
    cells = []
    for c in copies[copies.mixed].sort_values("hp_gap").itertuples():
        real, fake = (c.id_a, c.id_b) if c.label_a == 0 else (c.id_b, c.id_a)
        ims = [cv2.imread(str(Path(root, k, i + ".jpg"))) for k, i in (("real", real), ("fake", fake))]
        cells.append(np.hstack(ims))
        print("labelled real %s p=%.2f | labelled fake %s p=%.2f | inside %.2f outside %.2f" % (
            real[:8], c.prob_a if c.label_a == 0 else c.prob_b, fake[:8], c.prob_b if c.label_a == 0 else c.prob_a, c.hp_in, c.hp_out))
    while len(cells) % 3:
        cells.append(np.zeros_like(cells[0]))
    sheet = np.vstack([np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)])
    cv2.imwrite("runs/audit/exact_copy_conflicts.jpg", cv2.resize(sheet, None, fx=0.66, fy=0.66, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
