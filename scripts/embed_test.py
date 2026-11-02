"""Two offline ideas on the leak-free fold models. 1: average the labels of the nearest training images in model
feature space. 2: a fitted score offset per image-quality bucket. Each fold is scored by things fitted without it.
Usage: python scripts/embed_test.py"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dfd.data import read_image  # noqa: E402
from dfd.models import load_checkpoint  # noqa: E402
from oof import family  # noqa: E402

ROOT = Path("data/dataset/hackathon_dataset/train")
rows = list(csv.DictReader(open("configs/folds.csv", newline="")))
ids = [r["image_id"] for r in rows]
y = np.array([int(r["label"]) for r in rows])
fold = np.array([int(r["fold"]) for r in rows])
path = {p.stem: p for p in ROOT.rglob("*.jpg")}
imgs = np.stack([read_image(str(path[i])) for i in ids]).transpose(0, 3, 1, 2)
print("images", imgs.shape, flush=True)


@torch.no_grad()
def embed(run):
    m, _ = load_checkpoint("runs/%s/best.pt" % run, "cuda")
    out = []
    for i in range(0, len(imgs), 256):
        x = torch.from_numpy(imgs[i:i + 256]).cuda().float().div_(255)
        with torch.autocast("cuda", dtype=torch.float16):
            b = m.backbone
            f = b.forward_head(b.forward_features((x - m.mean) / m.std), pre_logits=True)
        out.append(torch.nn.functional.normalize(f.float().flatten(1), dim=1))
    return torch.cat(out)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


fams = {n: family("runs", n) for n in ("ncx", "nd", "nr")}
base = np.mean([fams[n].loc[ids, "z"].values for n in fams], axis=0)
print("base %.4f" % roc_auc_score(y, base), flush=True)
ks = (10, 50, 200)
knn = {q: np.zeros(len(ids)) for q in ks}
for k in range(5):
    t = fold != k
    v = fold == k
    new = {r["image_id"]: float(r["new_label"]) for r in csv.DictReader(open("configs/relabel_nest%d.csv" % k, newline=""))}
    lab = torch.tensor([new.get(i, l) for i, l in zip(ids, y)], device="cuda").float()
    keep = (lab >= 0).cpu().numpy() & t
    votes = {q: np.zeros(int(v.sum())) for q in ks}
    for n in fams:
        e = embed("%s_f%d" % (n, k))
        sim = e[torch.from_numpy(v).cuda()] @ e[torch.from_numpy(keep).cuda()].T
        top = sim.topk(max(ks), dim=1).indices
        tl = lab[torch.from_numpy(keep).cuda()][top]
        for q in ks:
            votes[q] += logit(tl[:, :q].mean(1).cpu().numpy())
    for q in ks:
        z = votes[q] / 3
        knn[q][v] = (z - z.mean()) / z.std()
    print("fold", k, "base %.4f" % roc_auc_score(y[v], base[v]),
          " ".join("k%d %.4f +0.5 %.4f" % (q, roc_auc_score(y[v], knn[q][v]), roc_auc_score(y[v], base[v] + .5 * knn[q][v])) for q in ks), flush=True)
for q in ks:
    print("k=%d alone %.4f" % (q, roc_auc_score(y, knn[q])), " ".join(
        "w%.2f %+.4f" % (w, roc_auc_score(y, base + w * knn[q]) - roc_auc_score(y, base)) for w in (.1, .25, .5, 1)),
        "corr %.3f" % np.corrcoef(base, knn[q])[0, 1], flush=True)


def quality(a):
    a = a.transpose(1, 2, 0)
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    return [g.mean(), float(a[..., 2].mean()) - float(a[..., 0].mean()), np.log1p(cv2.Laplacian(g, cv2.CV_64F).var()), g.std()]


q = np.array([quality(a) for a in imgs])
one = np.concatenate([np.eye(5)[np.searchsorted(np.quantile(c, [.2, .4, .6, .8]), c)][:, 1:] for c in q.T], axis=1)
adj = np.zeros(len(ids))
for k in range(5):
    t = fold != k
    X = np.column_stack([base, one])
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(X[t], y[t])
    adj[~t] = lr.decision_function(X[~t])
print("quality offsets: base %.4f adjusted %.4f" % (roc_auc_score(y, base), roc_auc_score(y, adj)))
print("per fold", " ".join("%+.4f" % (roc_auc_score(y[fold == k], adj[fold == k]) - roc_auc_score(y[fold == k], base[fold == k])) for k in range(5)))
