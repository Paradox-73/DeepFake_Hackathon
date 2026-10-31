"""Dataset adapters. An adapter turns a config section into a list of Sample.

To plug in the official format, write ONE function  fn(root, params) -> list  and point to it:
    data:
      adapter: custom
      root: /path/to/data
      params: {function: "my_adapter.py:load"}
The function may return Sample objects, dicts {path,label[,src]} or tuples (path,label[,src]).
label is 1 = fake, 0 = real, -1 = unknown (test set).
"""
import csv
import os
from dataclasses import dataclass
from pathlib import Path

from .utils import get_logger, load_callable

log = get_logger()

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
REAL_NAMES = {"real", "authentic", "original", "genuine", "pristine", "0"}
FAKE_NAMES = {"fake", "deepfake", "synthetic", "manipulated", "forged", "generated", "ai", "1"}


@dataclass
class Sample:
    path: str          # file actually read by the model (image or cached video frame)
    label: int         # 1 fake, 0 real, -1 unknown
    src: str           # id of the original item (image path / video path); predictions are per src


REGISTRY = {}


def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def parse_label(v, real_values=None, fake_values=None):
    s = str(v).strip().lower()
    real = {str(x).lower() for x in (real_values or REAL_NAMES)}
    fake = {str(x).lower() for x in (fake_values or FAKE_NAMES)}
    if s in fake:
        return 1
    if s in real:
        return 0
    try:
        return 1 if float(s) > 0.5 else 0
    except ValueError:
        raise ValueError("Cannot interpret label '%s'. Set params.real_values / params.fake_values." % v)


def _norm(items):
    out = []
    for it in items:
        if isinstance(it, Sample):
            out.append(it)
        elif isinstance(it, dict):
            out.append(Sample(str(it["path"]), int(it.get("label", -1)), str(it.get("src", it["path"]))))
        else:
            t = tuple(it)
            out.append(Sample(str(t[0]), int(t[1]) if len(t) > 1 else -1, str(t[2] if len(t) > 2 else t[0])))
    return out


@register("image_folder")
def image_folder(root, params):
    """root/<anything>/real/*.jpg and root/<anything>/fake/*.jpg (class = nearest folder whose
    name is in real_names / fake_names). Videos in the same tree are accepted (frames extracted later).
    params: extensions, real_names, fake_names, labeled (false for an unlabeled test folder)."""
    exts = {e.lower() for e in params.get("extensions", sorted(IMG_EXT | VID_EXT))}
    real = {str(x).lower() for x in params.get("real_names", REAL_NAMES)}
    fake = {str(x).lower() for x in params.get("fake_names", FAKE_NAMES)}
    labeled = params.get("labeled", True)
    out, skipped = [], 0
    for dp, dn, fns in os.walk(root):
        dn.sort()
        for fn in sorted(fns):
            if os.path.splitext(fn)[1].lower() not in exts:
                continue
            full = os.path.join(dp, fn)
            rel = Path(os.path.relpath(full, root)).as_posix()
            label = -1
            if labeled:
                for part in reversed(Path(rel).parts[:-1]):
                    if part.lower() in fake:
                        label = 1
                        break
                    if part.lower() in real:
                        label = 0
                        break
                if label < 0:
                    skipped += 1
                    continue
            out.append(Sample(full, label, rel))
    if skipped:
        log.warning("image_folder: skipped %d files not under a real/fake folder", skipped)
    return out


@register("video")
def video_folder(root, params):
    p = dict(params)
    p.setdefault("extensions", sorted(VID_EXT))
    return image_folder(root, p)


@register("csv")
def csv_adapter(root, params):
    """CSV with a path column and (optionally) a label column.
    params: csv (file, absolute or relative to root), path_col='path', label_col='label' (null = unlabeled),
    image_root (default root), path_prefix, path_suffix, sep=',', header=true, real_values, fake_values,
    skip_missing=false."""
    csv_path = params["csv"]
    if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        csv_path = os.path.join(root, csv_path)
    img_root = params.get("image_root", root)
    pc, lc = params.get("path_col", "path"), params.get("label_col", "label")
    header = params.get("header", True)
    pre, suf = params.get("path_prefix", ""), params.get("path_suffix", "")
    out = []
    with open(csv_path, newline="", encoding=params.get("encoding", "utf-8-sig")) as f:
        rows = list(csv.reader(f, delimiter=params.get("sep", ",")))
    if header:
        names, rows = [c.strip() for c in rows[0]], rows[1:]
        ip = names.index(pc)
        il = names.index(lc) if lc is not None else None
    else:
        ip, il = int(pc), (int(lc) if lc is not None else None)
    for r in rows:
        if not r:
            continue
        raw = r[ip].strip()
        full = pre + raw + suf
        if not os.path.isabs(full):
            full = os.path.join(img_root, full)
        if not os.path.exists(full):
            if params.get("skip_missing", False):
                continue
            raise FileNotFoundError(
                "CSV row points to missing file: %s (set params.skip_missing or path_prefix/path_suffix)" % full)
        label = parse_label(r[il], params.get("real_values"), params.get("fake_values")) if il is not None else -1
        out.append(Sample(full, label, raw))
    return out


@register("custom")
def custom(root, params):
    return _norm(load_callable(params["function"])(root, params))


def build_samples(section):
    """section: {adapter, root, params}"""
    name = section.get("adapter", "image_folder")
    if name not in REGISTRY:
        raise KeyError("Unknown adapter '%s'. Available: %s" % (name, sorted(REGISTRY)))
    samples = REGISTRY[name](section["root"], section.get("params") or {})
    if not samples:
        raise RuntimeError("Adapter '%s' found 0 samples under %s" % (name, section["root"]))
    return samples
