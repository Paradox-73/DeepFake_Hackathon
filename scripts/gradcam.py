"""Grad-CAM figures: where the detector looks, on out-of-fold images.
One run: each tile is input and heatmap.   python scripts/gradcam.py --run runs/cfm_f0 --n 12 --out docs/figures/gradcam.jpg
Several runs of the same fold: each row is one image, then one heatmap per run.
    python scripts/gradcam.py --run runs/cfm_f0 runs/dflipmix_f0 runs/a_rn50_f0 --n 4 --out docs/figures/gradcam_families.jpg"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfd.data import read_image  # noqa: E402
from dfd.models import load_checkpoint  # noqa: E402


def target_layer(net):
    b = net.backbone
    if hasattr(b, "blocks"):  # ViT: the class token alone feeds the head, so take the last block's input to attention
        return b.blocks[-1].norm1
    for name in ("stages", "layers"):
        if hasattr(b, name):
            return getattr(b, name)[-1]
    return b.conv4 if hasattr(b, "conv4") else list(b.children())[-3]


def cam(net, x, layer, sign):
    """sign=1 shows evidence for fake, sign=-1 shows evidence for real."""
    grabbed = {}
    h1 = layer.register_forward_hook(lambda m, i, o: grabbed.update(a=o))
    h2 = layer.register_full_backward_hook(lambda m, i, o: grabbed.update(g=o[0]))
    net.zero_grad()
    (sign * net(x)).sum().backward()
    h1.remove()
    h2.remove()
    a, g = grabbed["a"][0], grabbed["g"][0]
    if a.dim() == 2:  # ViT tokens: drop the class token, lay the patches out on their grid
        side = int(a.shape[0] ** 0.5)
        a, g = (t[t.shape[0] - side * side:].T.reshape(t.shape[1], side, side) for t in (a, g))
    elif a.shape[-1] > a.shape[0]:  # Swin keeps channels last
        a, g = a.permute(2, 0, 1), g.permute(2, 0, 1)
    m = (g.mean((1, 2), keepdim=True) * a).sum(0).relu().detach().cpu().numpy()
    return m / max(float(m.max()), 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--root", default="data/dataset/hackathon_dataset/train")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="docs/figures/gradcam.jpg")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(a.run) > 1:
        return families(a, device)
    a.run = a.run[0]
    net, ck = load_checkpoint(Path(a.run) / "best.pt", device)
    layer = target_layer(net)
    with open(Path(a.run) / "val_preds.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    fakes = sorted((r for r in rows if r["label"] == "1"), key=lambda r: -float(r["prob"]))[:a.n // 2]
    reals = sorted((r for r in rows if r["label"] == "0"), key=lambda r: float(r["prob"]))[:a.n // 2]
    tiles = []
    for r in fakes + reals:
        img = read_image(str(Path(a.root) / ("fake" if r["label"] == "1" else "real") / (r["image_id"] + ".jpg")))
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().to(device) / 255
        heat = cv2.resize(cam(net, x, layer, 1 if r["label"] == "1" else -1), (img.shape[1], img.shape[0]))
        color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        over = cv2.addWeighted(img[:, :, ::-1].copy(), 0.55, color, 0.45, 0)
        cv2.putText(over, "%s p=%.2f" % ("fake" if r["label"] == "1" else "real", float(r["prob"])), (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.hstack([img[:, :, ::-1], over]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rows_img = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
    cv2.imwrite(a.out, np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", a.out)


def overlay(img, heat, text):
    color = cv2.applyColorMap((cv2.resize(heat, (img.shape[1], img.shape[0])) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(img[:, :, ::-1].copy(), 0.55, color, 0.45, 0)
    cv2.putText(over, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return over


def families(a, device):
    """Same images through each run: the most confident correct fakes and reals by the runs' mean probability."""
    nets = [load_checkpoint(Path(r) / "best.pt", device) for r in a.run]
    preds = []
    for r in a.run:
        with open(Path(r) / "val_preds.csv", newline="") as f:
            preds.append({row["image_id"]: (int(row["label"]), float(row["prob"])) for row in csv.DictReader(f)})
    ids = sorted(set.intersection(*[set(p) for p in preds]))
    mean = {i: float(np.mean([p[i][1] for p in preds])) for i in ids}
    label = {i: preds[0][i][0] for i in ids}
    fakes = sorted((i for i in ids if label[i] == 1), key=lambda i: -mean[i])[:a.n // 2]
    reals = sorted((i for i in ids if label[i] == 0), key=lambda i: mean[i])[:a.n // 2]
    rows = []
    for i in fakes + reals:
        img = read_image(str(Path(a.root) / ("fake" if label[i] else "real") / (i + ".jpg")))
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().to(device) / 255
        tiles = [img[:, :, ::-1].copy()]
        cv2.putText(tiles[0], "%s, mean p=%.2f" % ("fake" if label[i] else "real", mean[i]), (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for (net, ck), p in zip(nets, preds):
            heat = cam(net, x, target_layer(net), 1 if label[i] else -1)
            tiles.append(overlay(img, heat, "%s p=%.2f" % (ck["arch"].split(".")[0].split("_patch")[0], p[i][1])))
        rows.append(np.hstack(tiles))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
