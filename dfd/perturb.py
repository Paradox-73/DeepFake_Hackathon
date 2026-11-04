"""Image perturbations on uint8 RGB HxWx3 arrays. Used by training augmentation and robustness eval."""
import cv2
import numpy as np


def jpeg(img, quality=50, rng=None):
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def blur(img, sigma=1.0, rng=None):
    return cv2.GaussianBlur(img, (0, 0), float(sigma))


def noise(img, std=10.0, rng=None):
    rng = rng or np.random.default_rng(0)
    out = img.astype(np.float32) + rng.normal(0.0, float(std), img.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def rescale(img, scale=0.5, rng=None):
    h, w = img.shape[:2]
    sh, sw = max(8, int(h * scale)), max(8, int(w * scale))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def brightness(img, factor=1.0, rng=None):
    return np.clip(img.astype(np.float32) * float(factor), 0, 255).astype(np.uint8)


def contrast(img, factor=1.0, rng=None):
    f = img.astype(np.float32)
    m = f.mean()
    return np.clip((f - m) * float(factor) + m, 0, 255).astype(np.uint8)


def identity(img, rng=None):
    return img


def gamma(img, g=1.0, rng=None):
    return np.clip(255.0 * (img.astype(np.float32) / 255.0) ** float(g), 0, 255).astype(np.uint8)


def invert(img, rng=None):
    return 255 - img


def solarize(img, thr=128, rng=None):
    return np.where(img >= int(thr), 255 - img, img).astype(np.uint8)


def posterize(img, bits=3, rng=None):
    shift = 8 - int(bits)
    return ((img >> shift) << shift).astype(np.uint8)


def grayscale(img, rng=None):
    return np.repeat(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, None], 3, axis=2)


def pixelate(img, block=4, rng=None):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // int(block)), max(1, h // int(block))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def motion_blur(img, size=9, rng=None):
    rng = rng or np.random.default_rng(0)
    k = np.zeros((int(size), int(size)), np.float32)
    k[int(size) // 2, :] = 1.0 / int(size)
    r = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    k = cv2.warpAffine(k, r, (int(size), int(size)))
    return cv2.filter2D(img, -1, k / max(k.sum(), 1e-6))


def salt_pepper(img, amount=0.02, rng=None):
    rng = rng or np.random.default_rng(0)
    m = rng.random(img.shape[:2])
    out = img.copy()
    out[m < amount / 2] = 0
    out[m > 1 - amount / 2] = 255
    return out


def speckle(img, std=0.2, rng=None):
    rng = rng or np.random.default_rng(0)
    f = img.astype(np.float32) * (1 + rng.normal(0, float(std), img.shape).astype(np.float32))
    return np.clip(f, 0, 255).astype(np.uint8)


def snow(img, n=60, rng=None):
    rng = rng or np.random.default_rng(0)
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(int(n)):
        cv2.circle(out, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(1, 4)),
                   (255, 255, 255), -1, cv2.LINE_AA)
    return out


def cutout(img, boxes=1, frac=0.15, rng=None):
    """Solid rectangles of random position and colour."""
    rng = rng or np.random.default_rng(0)
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(int(boxes)):
        bh, bw = max(1, int(h * frac * rng.uniform(0.5, 1.5))), max(1, int(w * frac * rng.uniform(0.5, 1.5)))
        y, x = int(rng.integers(0, max(1, h - bh))), int(rng.integers(0, max(1, w - bw)))
        out[y:y + bh, x:x + bw] = rng.integers(0, 256, 3) if rng.random() < 0.3 else int(rng.integers(0, 256))
    return out


def dropout_squares(img, n=40, size=6, rng=None):
    """Many small squares: the scattered black and coloured dots seen in the official images."""
    rng = rng or np.random.default_rng(0)
    out = img.copy()
    h, w = img.shape[:2]
    colour = rng.random() < 0.5
    for _ in range(int(n)):
        y, x = int(rng.integers(0, h - size)), int(rng.integers(0, w - size))
        out[y:y + size, x:x + size] = rng.integers(0, 256, 3) if colour else 0
    return out


def hue(img, shift=30, rng=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(np.int32) + int(shift)) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def tint(img, gains=(0.8, 1.0, 1.25), rng=None):
    return np.clip(img.astype(np.float32) * np.array(gains, np.float32), 0, 255).astype(np.uint8)


def channel_swap(img, order=(2, 1, 0), rng=None):
    return img[..., list(order)]


def sharpen(img, amount=1.5, rng=None):
    return np.clip(img.astype(np.float32) * (1 + amount) - cv2.GaussianBlur(img, (0, 0), 1.0).astype(np.float32) * amount,
                   0, 255).astype(np.uint8)


def rotate(img, degrees=10, rng=None):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(degrees), 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def crop(img, frac=0.9, rng=None):
    """Centre crop keeping frac of each side, resized back: a zoom-in."""
    h, w = img.shape[:2]
    ch, cw = int(round(h * frac)), int(round(w * frac))
    top, left = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(img[top:top + ch, left:left + cw], (w, h), interpolation=cv2.INTER_LINEAR)


OPS = {"jpeg": jpeg, "blur": blur, "noise": noise, "rescale": rescale, "brightness": brightness, "contrast": contrast,
       "identity": identity, "gamma": gamma, "invert": invert, "solarize": solarize, "posterize": posterize,
       "grayscale": grayscale, "pixelate": pixelate, "motion_blur": motion_blur, "salt_pepper": salt_pepper,
       "speckle": speckle, "snow": snow, "cutout": cutout, "dropout_squares": dropout_squares, "hue": hue, "tint": tint,
       "channel_swap": channel_swap, "sharpen": sharpen, "rotate": rotate, "crop": crop}


def apply_op(img, op, params=None, rng=None):
    return OPS[op](img, rng=rng, **(params or {}))


def apply_chain(img, chain, rng=None):
    """chain: list of {op:..., params:{...}}; single dict also accepted."""
    if isinstance(chain, dict):
        chain = [chain]
    for step in chain:
        img = apply_op(img, step["op"], step.get("params"), rng)
    return img
