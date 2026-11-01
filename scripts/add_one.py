"""Would one more family lift the ensemble? Scores base + candidate on exactly the images the candidate has scored.
Usage: python scripts/add_one.py --base cfm dflipmix a_rn50 --candidates v_hp v_face ..."""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oof import family  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    a = ap.parse_args()
    base = [family("runs", n) for n in a.base]
    rows = []
    for name in a.candidates:
        c = family("runs", name)
        ids = sorted(set(c.index) & set(base[0].index))
        y = base[0].loc[ids, "label"].values
        b = np.mean([f.loc[ids, "z"].values for f in base], axis=0)
        z = c.loc[ids, "z"].values
        r = np.corrcoef(b, z)[0, 1]
        k = len(base)
        rows.append((roc_auc_score(y, (k * b + z) / (k + 1)) - roc_auc_score(y, b), name, roc_auc_score(y, z), roc_auc_score(y, b), r, len(ids)))
    print("candidate      alone   base    gain with it  correlation  images")
    for gain, name, alone, b, r, n in sorted(rows, reverse=True):
        print("%-14s %.4f  %.4f  %+.4f       %.3f        %d" % (name, alone, b, gain, r, n))


if __name__ == "__main__":
    main()
