"""Training augmentation (numpy/cv2, deterministic given an rng)."""
import cv2
import numpy as np

from . import perturb


def resize_to(img, size):
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def _u(rng, rngrange):
    return float(rng.uniform(rngrange[0], rngrange[1]))


# Severity ranges for the corruption families visible in the official training images.
CORRUPTIONS = {
    "noise": lambda r: {"std": r.uniform(4, 35)},
    "speckle": lambda r: {"std": r.uniform(0.05, 0.35)},
    "salt_pepper": lambda r: {"amount": r.uniform(0.005, 0.05)},
    "jpeg": lambda r: {"quality": int(r.integers(15, 90))},
    "blur": lambda r: {"sigma": r.uniform(0.5, 3.0)},
    "motion_blur": lambda r: {"size": int(r.choice([5, 7, 9, 13]))},
    "pixelate": lambda r: {"block": int(r.integers(2, 9))},
    "rescale": lambda r: {"scale": r.uniform(0.25, 0.8)},
    "brightness": lambda r: {"factor": r.uniform(0.4, 1.7)},
    "contrast": lambda r: {"factor": r.uniform(0.4, 1.8)},
    "gamma": lambda r: {"g": r.uniform(0.4, 2.5)},
    "posterize": lambda r: {"bits": int(r.integers(2, 6))},
    "solarize": lambda r: {"thr": int(r.integers(64, 220))},
    "invert": lambda r: {},
    "grayscale": lambda r: {},
    "snow": lambda r: {"n": int(r.integers(20, 150))},
    "cutout": lambda r: {"boxes": int(r.integers(1, 3)), "frac": r.uniform(0.08, 0.25)},
    "dropout_squares": lambda r: {"n": int(r.integers(15, 80)), "size": int(r.integers(3, 10))},
}


# Colour damage found later, in the weakest group of the error analysis. Off by default so earlier runs stay repeatable.
COLOUR = {
    "hue": lambda r: {"shift": int(r.integers(10, 170))},
    "tint": lambda r: {"gains": tuple(r.uniform(0.6, 1.4, 3))},
    "channel_swap": lambda r: {"order": tuple(int(i) for i in r.permutation(3))},
    "sharpen": lambda r: {"amount": r.uniform(0.5, 2.5)},
}


def random_corruption(img, rng, colour=False):
    table = {**CORRUPTIONS, **COLOUR} if colour else CORRUPTIONS
    names = list(table)
    name = names[int(rng.integers(len(names)))]
    return perturb.apply_op(img, name, table[name](rng), rng)


def face_mask(h, w, rng):
    """Soft, slightly irregular oval over the inner face. Official crops are aligned, so no landmark model is needed."""
    cx, cy = w * rng.uniform(0.42, 0.58), h * rng.uniform(0.44, 0.60)
    ax, ay = w * rng.uniform(0.20, 0.34), h * rng.uniform(0.26, 0.42)
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), float(rng.uniform(-20, 20)), 0, 360, 1.0, -1)
    # Low-resolution random field added before re-thresholding gives the oval an uneven outline.
    wob = cv2.resize(rng.normal(0, 1, (6, 6)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    m = (cv2.GaussianBlur(m, (0, 0), 7) + 0.08 * wob > 0.5).astype(np.float32)
    sigma = float(rng.uniform(1, 11))
    m = cv2.GaussianBlur(m, (0, 0), sigma)
    return m / max(float(m.max()), 1e-6) * float(rng.choice([1.0, 1.0, 0.85, 0.7, 0.5]))


def _shift_source(src, rng):
    """Colour, sharpness and geometry change so the pasted region no longer matches its surroundings."""
    h, w = src.shape[:2]
    f = src.astype(np.float32)
    f = f * rng.uniform(0.85, 1.15, 3).astype(np.float32) + rng.uniform(-18, 18, 3).astype(np.float32)
    out = np.clip(f, 0, 255).astype(np.uint8)
    mode = rng.random()
    if mode < 0.33:
        out = perturb.blur(out, float(rng.uniform(0.5, 1.8)))
    elif mode < 0.66:
        b = perturb.blur(out, 1.0)
        out = np.clip(out.astype(np.float32) * 1.6 - b.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
    t = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-4, 4)), float(rng.uniform(0.94, 1.06)))
    t[:, 2] += rng.uniform(-5, 5, 2)
    return cv2.warpAffine(out, t, (w, h), borderMode=cv2.BORDER_REFLECT)


def self_blend(img, rng, donor=None):
    """Pseudo-fake from real training images. donor=None pastes a shifted copy of the same face (self-blended images);
    a donor pastes another real face, closer to a true swap."""
    src = _shift_source(img if donor is None else resize_to(donor, img.shape[0]), rng).astype(np.float32)
    m = face_mask(*img.shape[:2], rng)[:, :, None]
    tgt = img.astype(np.float32)
    if donor is not None:  # real swap tools colour-correct the pasted face, so a raw paste would be too easy to spot
        core = m[:, :, 0] > 0.5
        src = (src - src[core].mean(0)) / (src[core].std(0) + 1e-3) * tgt[core].std(0) + tgt[core].mean(0)
    return np.clip(tgt * (1 - m) + src * m, 0, 255).astype(np.uint8)


def train_augment(img, size, cfg, rng):
    cfg = cfg or {}
    if rng.random() < cfg.get("hflip", 0.5):
        img = img[:, ::-1]
    c = cfg.get("crop", {})
    if rng.random() < c.get("p", 0.5):
        h, w = img.shape[:2]
        area = _u(rng, c.get("scale", [0.6, 1.0])) * h * w
        ratio = float(np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3))))
        cw = int(min(w, round((area * ratio) ** 0.5)))
        ch = int(min(h, round((area / ratio) ** 0.5)))
        x0 = int(rng.integers(0, w - cw + 1))
        y0 = int(rng.integers(0, h - ch + 1))
        img = img[y0:y0 + ch, x0:x0 + cw]
    img = np.ascontiguousarray(img)
    img = resize_to(img, size)

    cj = cfg.get("color_jitter", {})
    if rng.random() < cj.get("p", 0.5):
        b = cj.get("strength", 0.2)
        img = perturb.brightness(img, 1 + rng.uniform(-b, b))
        img = perturb.contrast(img, 1 + rng.uniform(-b, b))
        f = img.astype(np.float32)
        gray = f.mean(axis=2, keepdims=True)
        s = 1 + rng.uniform(-b, b)
        img = np.clip(gray + (f - gray) * s, 0, 255).astype(np.uint8)

    j = cfg.get("jpeg", {})
    if rng.random() < j.get("p", 0.4):
        img = perturb.jpeg(img, int(rng.integers(*j.get("quality", [30, 95]))))
    bl = cfg.get("blur", {})
    if rng.random() < bl.get("p", 0.25):
        img = perturb.blur(img, _u(rng, bl.get("sigma", [0.3, 2.0])))
    n = cfg.get("noise", {})
    if rng.random() < n.get("p", 0.25):
        img = perturb.noise(img, _u(rng, n.get("std", [2, 15])), rng)
    r = cfg.get("rescale", {})
    if rng.random() < r.get("p", 0.25):
        img = perturb.rescale(img, _u(rng, r.get("scale", [0.4, 0.9])))
    lr = cfg.get("lowres")  # heavy shrink, log-uniform scale; draws nothing unless set, so earlier runs stay repeatable
    if lr and rng.random() < lr.get("p", 0.0):
        img = perturb.rescale(img, float(np.exp(rng.uniform(*np.log(lr.get("scale", [0.1, 0.35]))))))
    cp = cfg.get("corrupt", {})
    if rng.random() < cp.get("p", 0.0):
        for _ in range(int(rng.integers(1, cp.get("max_ops", 2) + 1))):
            img = random_corruption(np.ascontiguousarray(img), rng, cp.get("colour", False))
    return np.ascontiguousarray(img)
