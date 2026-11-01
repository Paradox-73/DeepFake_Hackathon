"""Can we undo the damage before scoring? Finds images of one kind, repairs them, and rescores with models that never trained on them.
Usage: python scripts/canon_test.py --families cfm dflipmix a_rn50"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ens_eval import member  # noqa: E402
from oof import family  # noqa: E402


def stretch(img):
    lo, hi = np.percentile(img, (1, 99))
    return np.clip((img.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)


def gamma(img, g=0.6):
    return (255 * (img.astype(np.float32) / 255) ** g).astype(np.uint8)


REPAIRS = {"invert": lambda i: 255 - i, "stretch": stretch, "gamma": gamma, "median": lambda i: cv2.medianBlur(i, 3),
           "bilateral": lambda i: cv2.bilateralFilter(i, 5, 40, 3), "denoise": lambda i: cv2.GaussianBlur(i, (0, 0), 0.8)}
TRIALS = [("blue", "invert"), ("blue", "stretch"), ("dark", "stretch"), ("dark", "gamma"), ("noisy", "median"), ("noisy", "bilateral"),
          ("noisy", "denoise"), ("lowcontrast", "stretch"), ("small", "stretch"), ("all", "stretch")]


class Images(Dataset):
    def __init__(self, paths, repair):
        self.paths, self.repair = paths, repair

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i])[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(self.repair(np.ascontiguousarray(img)))).permute(2, 0, 1)


@torch.inference_mode()
def score(models, paths, repair):
    out = []
    for x in DataLoader(Images(paths, repair), batch_size=64, num_workers=6):
        x = x.cuda(non_blocking=True).float().div_(255)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.stack([m(x) for m in models]).float().mean(0).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--root", default="data/dataset/hackathon_dataset/train")
    a = ap.parse_args()
    fams = [family("runs", n) for n in a.families]
    ids = sorted(fams[0].index)
    y = pd.Series(fams[0].loc[ids, "label"].values, index=ids)
    base = pd.Series(np.mean([f.loc[ids, "z"].values for f in fams], axis=0), index=ids)
    folds = pd.read_csv("configs/folds.csv", dtype={"image_id": str}).set_index("image_id").fold.reindex(ids)
    q = pd.read_csv("runs/audit/quality.csv", index_col=0, dtype={0: str}).loc[ids]
    paths = pd.Series([str(Path(a.root, "fake" if l else "real", i + ".jpg")) for i, l in y.items()], index=ids)
    groups = {"blue": q.redness < -5, "dark": q.bright < 50, "noisy": q.speckle > q.speckle.quantile(0.9),
              "lowcontrast": q.contrast < 30, "small": q["bytes"] < q["bytes"].quantile(0.1), "all": q.bright > -1}

    sample = paths[groups["blue"]].sample(32, random_state=1).tolist()
    tiles = [cv2.resize(cv2.imread(p), (112, 112)) for p in sample]
    cv2.imwrite("runs/audit/blue.jpg", np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 32, 8)]))

    pooled = roc_auc_score(y, base)
    print("baseline pooled AUC %.4f" % pooled)
    models_by_fold = {}
    print("group         repair      n   group AUC before  after  sum of both  pooled replace  pooled average")
    for group, repair in TRIALS:
        ids_g = base.index[groups[group].values]
        new = pd.Series(0.0, index=ids_g)
        for k in range(5):
            part = ids_g[folds.loc[ids_g] == k]
            if k not in models_by_fold:
                models_by_fold[k] = [member("runs/%s_f%d" % (f, k)) for f in a.families]
            new.loc[part] = score(models_by_fold[k], paths.loc[part].tolist(), REPAIRS[repair])
        rep, avg = base.copy(), base.copy()
        rep.loc[ids_g] = new
        avg.loc[ids_g] = (base.loc[ids_g] + new) / 2
        print("%-12s %-9s %5d   %.4f          %.4f   %.4f       %.4f          %.4f" % (
            group, repair, len(ids_g), roc_auc_score(y[ids_g], base[ids_g]), roc_auc_score(y[ids_g], new),
            roc_auc_score(y[ids_g], base[ids_g] + new), roc_auc_score(y, rep), roc_auc_score(y, avg)), flush=True)


if __name__ == "__main__":
    main()
