"""Which ensemble is better on an outer fold nobody trained on: models on subsets of the data, or seeds on all of it?
Stands in for the final choice between the 5-fold models and retraining on all 7,000 images.
Subset models: nest<k>_cx, nest<k>_d. Seed models: <family>_f<k> plus bag<c|d><seed>_f<k>. Same labels, no flips.
Usage: python scripts/bag_test.py --outer 0 1"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canon_test import score  # noqa: E402
from ens_eval import member  # noqa: E402


def standard(p):
    z = np.log(p.clip(1e-6, 1 - 1e-6) / (1 - p.clip(1e-6, 1 - 1e-6)))
    return ((z - z.mean()) / z.std()).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--root", default="data/dataset/hackathon_dataset/train")
    a = ap.parse_args()
    for k in a.outer:
        val = pd.read_csv("runs/deit_f%d/val_preds.csv" % k, dtype={"image_id": str})
        y = val.label.values
        paths = [str(Path(a.root, "fake" if l else "real", i + ".jpg")) for i, l in zip(val.image_id, y)]
        inner = [j for j in range(5) if j != k]
        print("outer fold", k)
        out = {}
        for tag, nest, seed_base, seeds in (("ConvNeXt-T", "cx", "cxd", "c"), ("DeiT3-S", "d", "deit", "d")):
            sub = [score([member("runs/nest%d_%s_f%d" % (k, nest, j))], paths, lambda i: i) for j in inner]
            full = [standard(pd.read_csv("runs/%s_f%d/val_preds.csv" % (seed_base, k)).prob)]
            full += [standard(pd.read_csv("runs/bag%s%d_f%d/val_preds.csv" % (seeds, s, k)).prob) for s in (1, 2, 3)]
            for name, models in (("subsets", sub), ("seeds", full)):
                single = np.mean([roc_auc_score(y, m) for m in models])
                pair = np.mean([roc_auc_score(y, models[i] + models[i + 1]) for i in range(3)])
                print("  %-11s %-8s mean single %.4f  mean of 2 %.4f  all 4 %.4f" % (
                    tag, name, single, pair, roc_auc_score(y, np.mean(models, axis=0))))
                out.setdefault(name, []).extend(models)
        for name, models in out.items():
            print("  both families %-8s all 8 %.4f" % (name, roc_auc_score(y, np.mean(models, axis=0))))
        print("  both kinds together     all 16 %.4f" % roc_auc_score(y, np.mean(out["subsets"] + out["seeds"], axis=0)))


if __name__ == "__main__":
    main()
