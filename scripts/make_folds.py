"""Find near-duplicate training images and assign them to the same fold, so validation scores are not inflated
by the same photo (or its re-corrupted copy) sitting on both sides of a split.

Two signals, joined by union-find:
  - ImageNet-backbone embedding cosine similarity (survives colour changes and noise)
  - correlation of 32x32 grayscale thumbnails, absolute value so inverted copies still match

Only training images are read. Output: configs/folds.csv (image_id, label, group, fold).
Usage: python scripts/make_folds.py --root data/dataset/hackathon_dataset --out runs/audit"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfd.adapters import Sample  # noqa: E402
from dfd.data import ImageDataset, read_image  # noqa: E402


def find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return x


def union_pairs(n, pairs):
    p = list(range(n))
    for a, b in pairs:
        ra, rb = find(p, a), find(p, b)
        if ra != rb:
            p[rb] = ra
    return np.array([find(p, i) for i in range(n)])


@torch.no_grad()
def embed(samples, device, batch=64):
    m = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True, num_classes=0).to(device).eval()
    mean = torch.tensor(m.pretrained_cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(m.pretrained_cfg["std"], device=device).view(1, 3, 1, 1)
    dl = DataLoader(ImageDataset(samples, 224), batch_size=batch, num_workers=4)
    out = []
    for x, _, _ in dl:
        x = (x.to(device).float() / 255 - mean) / std
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.nn.functional.normalize(m(x).float(), dim=1).cpu())
    return torch.cat(out)


def thumbs(samples):
    t = np.stack([cv2.resize(cv2.cvtColor(read_image(s.path), cv2.COLOR_RGB2GRAY), (32, 32),
                             interpolation=cv2.INTER_AREA).astype(np.float32).ravel() for s in samples])
    t -= t.mean(1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-6
    return torch.from_numpy(t)


def top_pairs(v, thr, device, absolute=False, chunk=1024):
    v = v.to(device)
    pairs, sims = [], []
    for i in range(0, len(v), chunk):
        s = v[i:i + chunk] @ v.T
        s = s.abs() if absolute else s
        a, b = torch.where(s >= thr)
        a = a + i
        keep = a < b
        pairs.append(torch.stack([a[keep], b[keep]], 1).cpu())
        sims.append(s[a[keep] - i, b[keep]].cpu())
    return torch.cat(pairs).numpy(), torch.cat(sims).numpy()


def contact(samples, pairs, sims, out, n=32, size=112):
    order = np.argsort(sims)[:n]  # weakest matches above the threshold: the ones most likely to be wrong
    rows = max(1, int(np.ceil(len(order) / 4)))
    canvas = np.zeros((rows * size, 8 * size, 3), np.uint8)
    for k, idx in enumerate(order):
        r, c = k // 4, (k % 4) * 2
        for j, s in enumerate(pairs[idx]):
            canvas[r * size:(r + 1) * size, (c + j) * size:(c + j + 1) * size] = cv2.resize(
                read_image(samples[s].path), (size, size))[:, :, ::-1]
    cv2.imwrite(str(out), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset/hackathon_dataset")
    ap.add_argument("--out", default="runs/audit")
    # Deliberately loose: a false merge only keeps extra images in one fold, a miss leaks a photo across folds.
    ap.add_argument("--emb_thr", type=float, default=0.87)
    ap.add_argument("--thumb_thr", type=float, default=0.87)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(root / "train_metadata.csv", dtype={"image_id": str})
    meta["path"] = [str(root / "train" / ("fake" if l else "real") / (i + ".jpg")) for i, l in zip(meta.image_id, meta.label)]
    samples = [Sample(p, l, p) for p, l in zip(meta.path, meta.label)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    e, t = embed(samples, device), thumbs(samples)
    torch.save({"emb": e, "ids": meta.image_id.tolist()}, out / "train_embeddings.pt")
    for thr in (0.80, 0.85, 0.90, 0.95):
        print("embedding cos >= %.2f pairs: %d" % (thr, len(top_pairs(e, thr, device)[0])))
    for thr in (0.85, 0.90, 0.93, 0.96):
        print("thumbnail |corr| >= %.2f pairs: %d" % (thr, len(top_pairs(t, thr, device, absolute=True)[0])))

    pe, se = top_pairs(e, a.emb_thr, device)
    pt, st = top_pairs(t, a.thumb_thr, device, absolute=True)
    contact(samples, pe, se, out / "pairs_embedding.jpg")
    contact(samples, pt, st, out / "pairs_thumbnail.jpg")
    pairs = np.concatenate([pe, pt])
    meta["group"] = union_pairs(len(meta), pairs.tolist())
    sizes = meta.groupby("group").size()
    print("groups: %d, biggest %d, images in a group of 2+: %d" % (len(sizes), sizes.max(), sizes[sizes > 1].sum()))
    mixed = meta.groupby("group").label.nunique()
    print("groups containing both real and fake: %d" % (mixed > 1).sum())

    meta["fold"] = -1
    skf = StratifiedGroupKFold(a.folds, shuffle=True, random_state=a.seed)
    for k, (_, va) in enumerate(skf.split(meta, meta.label, meta.group)):
        meta.loc[va, "fold"] = k
    assert meta.groupby("group").fold.nunique().max() == 1, "a group straddles folds"
    print(meta.groupby("fold").label.agg(["size", "mean"]))
    meta[["image_id", "label", "group", "fold"]].to_csv(Path(__file__).resolve().parents[1] / "configs/folds.csv", index=False)


if __name__ == "__main__":
    main()
