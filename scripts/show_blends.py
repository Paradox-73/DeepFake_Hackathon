"""Contact sheet of pseudo-fakes: real image, self-blend, donor blend, one row per image. Visual check only.
Usage: python scripts/show_blends.py"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfd.augment import self_blend  # noqa: E402
from dfd.data import read_image  # noqa: E402

paths = sorted(Path("data/dataset/hackathon_dataset/train/real").iterdir())
rng = np.random.default_rng(3)
idx = rng.permutation(len(paths))[:12]
rows = []
for a, b in zip(idx[:6], idx[6:]):
    img, other = read_image(str(paths[a])), read_image(str(paths[b]))
    rows.append(np.hstack([img, self_blend(img, rng), self_blend(img, rng, other)]))
cv2.imwrite("runs/audit/blends.jpg", np.vstack(rows)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
