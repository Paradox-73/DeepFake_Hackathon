"""Dataset, splitting, aggregation."""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .adapters import Sample
from .augment import resize_to, self_blend, train_augment
from .utils import get_logger

log = get_logger()
_warned = set()


def read_image(path):
    """RGB uint8 HxWx3. Works with unicode paths on Windows."""
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        try:
            from PIL import Image
            return np.array(Image.open(path).convert("RGB"))
        except Exception:
            if path not in _warned:
                _warned.add(path)
                log.warning("Unreadable image, using blank: %s", path)
            return np.full((224, 224, 3), 127, np.uint8)
    return img[:, :, ::-1]


class ImageDataset(Dataset):
    """Returns (uint8 CHW tensor, label float, index). Normalisation happens on the GPU inside the model.
    Augmentation randomness depends only on (seed, epoch, index) -> reproducible for any num_workers."""

    def __init__(self, samples, img_size, augment=None, seed=0, perturb=None):
        self.samples, self.size, self.aug, self.seed, self.perturb = samples, img_size, augment, seed, perturb
        self.epoch = 0
        self.blend = (augment or {}).get("blend", {})
        self.reals = [j for j, s in enumerate(samples) if s.label == 0]

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        rng = np.random.default_rng([self.seed, self.epoch, i])
        if self.aug is not None and s.label == 1 and rng.random() < self.blend.get("p", 0.0):
            # Slot of a real fake is taken by a pseudo-fake, so the class balance does not change.
            img = read_image(self.samples[self.reals[int(rng.integers(len(self.reals)))]].path)
            donor = None
            if rng.random() < self.blend.get("donor_p", 0.5):
                donor = read_image(self.samples[self.reals[int(rng.integers(len(self.reals)))]].path)
            img = self_blend(np.ascontiguousarray(img), rng, donor)
        else:
            img = read_image(s.path)
        if self.aug is not None:
            img = train_augment(img, self.size, self.aug, rng)
        else:
            if self.perturb is not None:
                img = self.perturb(np.ascontiguousarray(img), rng)
            img = resize_to(np.ascontiguousarray(img), self.size)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous()
        return x, float(s.label), i


def split_by_src(samples, val_fraction, seed):
    """Stratified split on original items (src) so frames of one video never straddle train/val."""
    srcs, labels = [], []
    seen = {}
    for s in samples:
        if s.src not in seen:
            seen[s.src] = len(srcs)
            srcs.append(s.src)
            labels.append(s.label)
    try:
        tr, va = train_test_split(srcs, test_size=val_fraction, random_state=seed, stratify=labels)
    except ValueError:
        log.warning("Stratified split failed, using plain random split")
        tr, va = train_test_split(srcs, test_size=val_fraction, random_state=seed)
    vs = set(va)
    return [s for s in samples if s.src not in vs], [s for s in samples if s.src in vs], sorted(tr), sorted(va)


def relabel(samples, relabel_csv):
    """Apply scripts/clean_labels.py output to TRAINING samples only. new_label -1 drops the image.
    A fractional new_label is a soft target."""
    with open(relabel_csv, newline="") as f:
        new = {r["image_id"]: float(r["new_label"]) for r in csv.DictReader(f)}
    out = [Sample(s.path, new.get(Path(s.src).stem, s.label), s.src) for s in samples]
    log.info("Relabel: %d of %d training labels changed or dropped", sum(Path(s.src).stem in new for s in samples), len(samples))
    return [s for s in out if s.label >= 0]


def split_by_fold(samples, folds_csv, fold):
    """Group-aware split from configs/folds.csv: near-duplicate photos always share a fold."""
    with open(folds_csv, newline="") as f:
        fold_of = {r["image_id"]: int(r["fold"]) for r in csv.DictReader(f)}
    samples = [s for s in samples if Path(s.src).stem in fold_of]  # a partial CSV holds images out, see nested_check.sh
    tr = [s for s in samples if fold_of[Path(s.src).stem] != fold]
    va = [s for s in samples if fold_of[Path(s.src).stem] == fold]
    return tr, va, sorted(s.src for s in tr), sorted(s.src for s in va)


def select_by_src(samples, srcs):
    keep = set(srcs)
    return [s for s in samples if s.src in keep]


def aggregate(samples, probs, mode="mean"):
    """Combine frame-level probabilities into one per src. Returns (srcs, probs, labels)."""
    order, groups, labels = [], {}, {}
    for s, p in zip(samples, probs):
        if s.src not in groups:
            order.append(s.src)
            groups[s.src] = []
            labels[s.src] = s.label
        groups[s.src].append(p)
    fn = {"mean": np.mean, "max": np.max, "median": np.median}[mode]
    return order, np.array([fn(groups[k]) for k in order]), np.array([labels[k] for k in order])
