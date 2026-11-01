"""Does scoring extra views of each image lift the ensemble? Each image is scored only by fold models that never trained on it.
Views: slight zooms, small shifts, and for convolutional models a larger input than training used (DeiT keeps 224 there).
Usage: python scripts/tta_test.py --bundle runs/final/model_weights.pt"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_script import Member  # noqa: E402


def crop(x, top, left, size):
    return F.interpolate(x[..., top:top + size, left:left + size], size=224, mode="bilinear", align_corners=False)


VIEWS = {
    "base": lambda x: x,
    "zoom95": lambda x: crop(x, 6, 6, 212),
    "zoom90": lambda x: crop(x, 11, 11, 202),
    "zoom85": lambda x: crop(x, 17, 17, 190),
    "shift": lambda x: crop(x, 0, 0, 212),
    "up256": lambda x: F.interpolate(x, size=256, mode="bilinear", align_corners=False),
    "up288": lambda x: F.interpolate(x, size=288, mode="bilinear", align_corners=False),
}


def logit(m, x):
    return (m.backbone((x - m.mean) / m.std).squeeze(1).float() - m.logit_mean) / m.logit_std


@torch.inference_mode()
def score(m, paths, view, cnn):
    out = []
    for i in range(0, len(paths), 64):
        x = torch.from_numpy(np.stack([cv2.imread(p)[:, :, ::-1].copy() for p in paths[i:i + 64]])).permute(0, 3, 1, 2)
        x = x.cuda().float().div_(255)
        v = VIEWS[view if cnn or not view.startswith("up") else "base"](x)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(logit(m, v).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="runs/final/model_weights.pt")
    ap.add_argument("--folds_csv", default="configs/folds.csv")
    ap.add_argument("--root", default="data/dataset/hackathon_dataset/train")
    a = ap.parse_args()
    b = torch.load(a.bundle, map_location="cpu", weights_only=False)
    folds = pd.read_csv(a.folds_csv, dtype={"image_id": str})
    print(folds.columns.tolist(), len(folds), flush=True)
    keep = {v: [] for v in VIEWS}
    labels = []
    for k in range(5):
        val = folds[folds.fold == k]
        paths = [str(Path(a.root, "fake" if l else "real", i + ".jpg")) for i, l in zip(val.image_id, val.label)]
        labels.append(val.label.values)
        specs = [s for s in b["members"] if s["run"].endswith("_f%d" % k)]
        assert len(specs) == 3, [s["run"] for s in specs]
        per_view = {v: [] for v in VIEWS}
        for s in specs:
            m = Member(s).cuda().eval()
            cnn = "deit" not in s["arch"]
            for v in VIEWS:
                per_view[v].append(score(m, paths, v, cnn))
            del m
        for v in VIEWS:
            keep[v].append(np.stack(per_view[v]))
        print("fold", k, "done", flush=True)
    y = np.concatenate(labels)
    z = {v: np.concatenate(keep[v], 1) for v in VIEWS}
    np.savez("runs/tta_test.npz", y=y, **z)
    fold = np.concatenate([np.full(len(l), i) for i, l in enumerate(labels)])
    base = z["base"].mean(0)
    print("base %.4f" % roc_auc_score(y, base))
    for v in VIEWS:
        if v == "base":
            continue
        s, both = z[v].mean(0), (z[v].mean(0) + base) / 2
        gains = [roc_auc_score(y[fold == k], both[fold == k]) - roc_auc_score(y[fold == k], base[fold == k]) for k in range(5)]
        print("%-7s alone %.4f  with base %.4f  gain %+.4f  per fold %s" % (
            v, roc_auc_score(y, s), roc_auc_score(y, both), roc_auc_score(y, both) - roc_auc_score(y, base),
            " ".join("%+.4f" % g for g in gains)))


if __name__ == "__main__":
    main()
