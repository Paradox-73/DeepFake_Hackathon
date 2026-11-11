"""Deepfake detector inference for the IIIT Bangalore Deepfake ML Hackathon.

    python inference_script.py --test_dir ./test/images/ --weights ./model_weights.pt --output predictions.csv

Writes image_id,label,confidence for every image in --test_dir. confidence is P(fake). The run is deterministic:
no random crops, fixed member order, sorted output. --weights is the single ensemble file from scripts/build_bundle.py.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INPUT_SIZE = 224


def apply_view(x, view):
    """What one ensemble family sees of an RGB batch in [0, 1]. A second view of the image gives the ensemble new errors."""
    if view == "rgb":
        return x
    if view == "face":  # swaps sit in the inner face, so drop the background and enlarge the face
        h, w = x.shape[-2:]
        top, left = int(0.12 * h), int(0.15 * w)
        return F.interpolate(x[..., top:h - top, left:w - left], size=(h, w), mode="bilinear", align_corners=False)
    if view == "highpass":  # fine texture only: no colour, no identity
        g = torch.exp(-(torch.arange(-6.0, 7.0, device=x.device) ** 2) / 8).to(x.dtype)
        kernel = (g[None] * g[:, None])[None, None]
        blur = F.conv2d(F.pad(x, (6, 6, 6, 6), mode="replicate"), (kernel / kernel.sum()).expand(3, -1, -1, -1), groups=3)
        return (4 * (x - blur) + 0.5).clamp(0, 1)
    if view == "ycbcr":  # swaps often show in chroma before they show in brightness
        y = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        return torch.stack([y, 0.564 * (x[:, 2] - y) + 0.5, 0.713 * (x[:, 0] - y) + 0.5], 1)
    if view == "fft":  # log magnitude spectrum, the classic place to look for resampling traces
        with torch.autocast(x.device.type, enabled=False):
            spec = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
            return (torch.log1p(spec.abs()) / 8).clamp(0, 1).to(x.dtype)
    raise ValueError("unknown view: %s" % view)


class TestImages(Dataset):
    def __init__(self, test_dir):
        self.paths = sorted(p for p in Path(test_dir).iterdir() if p.suffix.lower() in EXTENSIONS)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = None
        try:
            img = cv2.imdecode(np.fromfile(str(self.paths[i]), np.uint8), cv2.IMREAD_COLOR)
        except (OSError, ValueError, cv2.error):  # cv2.error: an empty file
            pass
        ok = img is not None
        if not ok:
            img = np.full((INPUT_SIZE, INPUT_SIZE, 3), 127, np.uint8)
        if img.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            smaller = img.shape[0] > INPUT_SIZE or img.shape[1] > INPUT_SIZE
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA if smaller else cv2.INTER_LINEAR)
        return torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1), i, ok


class Member(nn.Module):
    """One ensemble member: timm backbone with its own normalisation, output is a standardised logit."""

    def __init__(self, spec):
        super().__init__()
        pool = {"global_pool": spec["global_pool"]} if spec.get("global_pool") else {}
        self.backbone = timm.create_model(spec["arch"], pretrained=False, num_classes=1, drop_rate=0.0, **pool)
        self.view_name = spec.get("view", "rgb")
        self.register_buffer("mean", torch.tensor(spec["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(spec["std"]).view(1, 3, 1, 1))
        self.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in spec["state_dict"].items()})
        self.logit_mean, self.logit_std, self.img_size = spec["logit_mean"], spec["logit_std"], spec["img_size"]

    def forward(self, x):
        if x.shape[-1] != self.img_size:
            x = nn.functional.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        x = apply_view(x, self.view_name)
        return (self.backbone((x - self.mean) / self.std).squeeze(1).float() - self.logit_mean) / self.logit_std


@torch.inference_mode()
def predict(bundle, test_dir, device, batch_size, workers):
    ds = TestImages(test_dir)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda",
                    persistent_workers=workers > 0)
    print("[INFO] %d images, %d ensemble members, flip TTA: %s" % (len(ds), len(bundle["members"]), bundle["flip_tta"]))
    scores = np.zeros(len(ds), dtype=np.float64)
    failed = []
    # shortcut: holds every member on the GPU at once. All fit on a 16 GB T4; a far larger ensemble would need swapping.
    members = [Member(s).to(device).eval().to(memory_format=torch.channels_last) for s in bundle["members"]]
    for x, idx, ok in dl:
        x = x.to(device, non_blocking=True).float().div_(255).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            total = sum(m(x) for m in members)
            if bundle["flip_tta"]:
                flipped = torch.flip(x, dims=[3])
                total = (total + sum(m(flipped) for m in members)) / 2
        scores[idx.numpy()] = (total / len(members)).float().cpu().numpy()
        failed += idx[~ok].tolist()
    conf = 1.0 / (1.0 + np.exp(-scores * bundle["temperature"]))
    failed += np.flatnonzero(np.isnan(conf)).tolist()
    for i in failed:
        print("[WARN] unreadable image scored 0.5: %s" % ds.paths[i].name)
        conf[i] = 0.5
    return [p.stem for p in ds.paths], conf


def main():
    parser = argparse.ArgumentParser(description="Deepfake ML Hackathon inference")
    parser.add_argument("--test_dir", required=True, help="Path to test images folder")
    parser.add_argument("--weights", required=True, help="Path to model weight file")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=64)
    # 0: decoding is a tiny share of the run, and worker processes can die in containers with little shared memory
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None,
                        help="label = confidence >= threshold (default: the cut-off stored in --weights, 0.475)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    t0 = time.time()
    bundle = torch.load(args.weights, map_location="cpu", weights_only=False)
    ids, conf = predict(bundle, args.test_dir, device, args.batch_size, args.workers)
    threshold = bundle["threshold"] if args.threshold is None else args.threshold

    assert len(ids) == len(set(ids)), "duplicate image ids"
    assert ((conf >= 0) & (conf <= 1)).all(), "confidence outside [0, 1]"
    order = np.argsort(ids)
    Path(args.output).resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")  # same line ends as sample_submission.csv
        w.writerow(["image_id", "label", "confidence"])
        for i in order:
            w.writerow([ids[i], int(conf[i] >= threshold), "%.8f" % conf[i]])

    peak = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
    print("[DONE] %d rows in %s | %.1f s on %s | peak GPU memory %.0f MB | predicted fake %.1f%%" % (
        len(ids), args.output, time.time() - t0, device, peak, 100 * (conf >= threshold).mean()))


if __name__ == "__main__":
    main()
