"""Score an ensemble out of fold: each image is scored only by the fold models that never trained on it.
Reports clean and perturbed AUC, with and without the horizontal flip, using the same members as inference_script.py.
Usage: python scripts/ens_eval.py --families cfm dflipmix a_rn50 --out runs/ens_eval [--clean_only | --extra]
--extra scores the damage types in robust_eval.EXTRA instead of the default 20."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfd.perturb import apply_chain  # noqa: E402
from inference_script import Member  # noqa: E402
from robust_eval import DEFAULT, EXTRA  # noqa: E402


class Val(Dataset):
    def __init__(self, frame, root, chain):
        self.frame, self.root, self.chain = frame, root, chain

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        name, label = self.frame.image_id.iat[i], self.frame.label.iat[i]
        img = cv2.imread(str(Path(self.root, "fake" if label else "real", name + ".jpg")))[:, :, ::-1]
        img = apply_chain(np.ascontiguousarray(img), self.chain, np.random.default_rng([42, i]))
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)


def member(run):
    ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    p = pd.read_csv(Path(run) / "val_preds.csv").prob.clip(1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    spec = dict(arch=ck["arch"], img_size=ck["img_size"], mean=ck["mean"], std=ck["std"], state_dict=ck["state_dict"],
                view=ck.get("view", "rgb"), global_pool=ck.get("global_pool"), logit_mean=float(z.mean()), logit_std=float(z.std()))
    return Member(spec).cuda().eval()


@torch.inference_mode()
def score(models, frame, root, chain, workers):
    """Returns plain and flipped standardised logits, shape (models, images)."""
    plain, flipped = [], []
    for x in DataLoader(Val(frame, root, chain), batch_size=64, num_workers=workers):
        x = x.cuda(non_blocking=True).float().div_(255)
        with torch.autocast("cuda", dtype=torch.float16):
            plain.append(torch.stack([m(x) for m in models]).float().cpu())
            flipped.append(torch.stack([m(x.flip(3)) for m in models]).float().cpu())
    return torch.cat(plain, 1).numpy(), torch.cat(flipped, 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--root", default="data/dataset/hackathon_dataset/train")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="runs/ens_eval")
    ap.add_argument("--clean_only", action="store_true")
    ap.add_argument("--extra", action="store_true")
    a = ap.parse_args()
    perts = DEFAULT[:1] if a.clean_only else DEFAULT[:1] + EXTRA if a.extra else DEFAULT
    names = [p["name"] for p in perts]
    keep = {n: {"y": [], "plain": [], "flip": []} for n in names}
    for k in a.folds:
        runs = ["runs/%s_f%d" % (f, k) for f in a.families]
        frame = pd.read_csv(Path(runs[0]) / "val_preds.csv", dtype={"image_id": str})
        models = [member(r) for r in runs]
        for p in perts:
            pl, fl = score(models, frame, a.root, p["chain"], a.workers)
            keep[p["name"]]["y"].append(frame.label.values)
            keep[p["name"]]["plain"].append(pl)
            keep[p["name"]]["flip"].append(fl)
        print("fold", k, "done", flush=True)
        del models

    rows = []
    for n in names:
        y = np.concatenate(keep[n]["y"])
        pl, fl = np.concatenate(keep[n]["plain"], 1), np.concatenate(keep[n]["flip"], 1)
        row = {"name": n, "ensemble": roc_auc_score(y, pl.mean(0)), "ensemble_flip": roc_auc_score(y, (pl + fl).mean(0))}
        row.update({f: roc_auc_score(y, pl[i]) for i, f in enumerate(a.families)})
        rows.append(row)
    d = pd.DataFrame(rows)
    print(d.round(4).to_string(index=False))
    if len(d) > 1:
        rest = d[d.name != "clean"].drop(columns="name")
        print("mean over perturbed:", rest.mean().round(4).to_dict())
        print("worst over perturbed:", rest.min().round(4).to_dict())
    Path(a.out + ".json").write_text(json.dumps(rows, indent=1))
    # per-image standardised scores (families x images) for each damage type, so ensembles of any subset can be scored later
    np.savez_compressed(a.out + "_scores.npz", families=np.array(a.families), y=np.concatenate(keep[names[0]]["y"]),
                        **{n: np.concatenate(keep[n]["plain"], 1) for n in names})


if __name__ == "__main__":
    main()
