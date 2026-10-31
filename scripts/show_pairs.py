"""Contact sheet of training photos that exist as both real and fake: real, fake, amplified difference.
Usage: python scripts/show_pairs.py --n 8"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset/hackathon_dataset/train")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="runs/audit/real_fake_pairs.jpg")
    a = ap.parse_args()
    d = pd.read_csv("configs/folds.csv", dtype={"image_id": str})
    rows = []
    for _, g in d.groupby("group"):
        if g.label.nunique() != 2 or len(g) > 3:
            continue
        r = cv2.imread(str(Path(a.root) / "real" / (g[g.label == 0].image_id.iloc[0] + ".jpg")))
        f = cv2.imread(str(Path(a.root) / "fake" / (g[g.label == 1].image_id.iloc[0] + ".jpg")))
        # Least-corrupted pairs first: small pixel difference means both copies stayed close to the source photo.
        rows.append((float(np.abs(r.astype(float) - f).mean()), r, f))
    rows.sort(key=lambda x: x[0])
    print("pairs", len(rows), "mean abs diff, lowest 10:", [round(x[0], 1) for x in rows[:10]])
    out = []
    for _, r, f in rows[:a.n]:
        diff = np.clip(np.abs(r.astype(np.int16) - f) * 4, 0, 255).astype(np.uint8)
        out.append(np.hstack([r, f, diff]))
    cv2.imwrite(a.out, np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
