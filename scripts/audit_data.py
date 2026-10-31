"""Audit the official dataset: labels, formats, duplicates, per-class low-level statistics, contact sheets.
Usage: python scripts/audit_data.py --root data/dataset/hackathon_dataset --out runs/audit
Test images are read only for format statistics, never for labels or fitting."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def jpeg_quality(q):
    """Estimate libjpeg quality from the luma quantisation table (IJG scaling)."""
    base = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22,
            29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103,
            121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
    if not q:
        return -1
    t = np.array(q.get(0, []), dtype=float)
    if t.size != 64:
        return -1
    best, err = -1, 1e18
    for s_q in range(1, 101):
        s = 5000 / s_q if s_q < 50 else 200 - 2 * s_q
        c = np.clip(np.floor((np.array(sorted(base)) * s + 50) / 100), 1, 255)
        e = np.abs(np.sort(t) - c).sum()
        if e < err:
            best, err = s_q, e
    return best


def ahash(gray):
    """64-bit difference hash on an 8x9 downscale."""
    s = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((s[:, 1:] > s[:, :-1]).ravel()).tobytes().hex()


def one(path):
    raw = Path(path).read_bytes()
    im = Image.open(path)
    q = jpeg_quality(getattr(im, "quantization", None))
    info = {"path": path, "bytes": len(raw), "md5": hashlib.md5(raw).hexdigest(), "w": im.width, "h": im.height,
            "mode": im.mode, "format": im.format, "quality": q,
            "subsampling": getattr(im, "layer", [(0, 1, 1)])[0][1:3].__repr__() if hasattr(im, "layer") else ""}
    a = np.array(im.convert("RGB"))
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    f = g.astype(np.float32)
    hp = f - cv2.GaussianBlur(f, (0, 0), 1.0)
    fft = np.abs(np.fft.fftshift(np.fft.fft2(f - f.mean()))) ** 2
    yy, xx = np.indices(fft.shape)
    r = np.hypot(yy - fft.shape[0] / 2, xx - fft.shape[1] / 2)
    rmax = r.max()
    hf = fft[r > 0.5 * rmax].sum() / max(fft.sum(), 1e-9)
    # 8x8 grid blockiness: step across block borders relative to steps elsewhere
    dx = np.abs(np.diff(f, axis=1)).mean(axis=0)
    idx = np.arange(dx.size)
    block = dx[idx % 8 == 7].mean() / max(dx[idx % 8 != 7].mean(), 1e-6)
    info.update(mean=float(a.mean()), std=float(a.std()), lap_var=float(cv2.Laplacian(g, cv2.CV_64F).var()),
                hp_std=float(hp.std()), hf_ratio=float(hf), blockiness=float(block),
                sat=float(cv2.cvtColor(a, cv2.COLOR_RGB2HSV)[..., 1].mean()), dhash=ahash(g))
    return info


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def sheet(paths, out, n=64, size=112):
    cols = int(n ** 0.5)
    canvas = np.zeros((size * cols, size * cols, 3), np.uint8)
    for i, p in enumerate(paths[:n]):
        a = cv2.resize(cv2.imread(str(p)), (size, size), interpolation=cv2.INTER_AREA)
        canvas[(i // cols) * size:(i // cols + 1) * size, (i % cols) * size:(i % cols + 1) * size] = a
    cv2.imwrite(str(out), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dataset/hackathon_dataset")
    ap.add_argument("--out", default="runs/audit")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(root / "train_metadata.csv", dtype={"image_id": str})
    files = [(p, "real") for p in sorted((root / "train/real").iterdir())] + \
            [(p, "fake") for p in sorted((root / "train/fake").iterdir())] + \
            [(p, "test") for p in sorted((root / "test/images").iterdir())]
    split = {str(p): s for p, s in files}
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, list(split), chunksize=64))
    df = pd.DataFrame(rows)
    df["split"] = df.path.map(split)
    df["image_id"] = df.path.map(lambda p: Path(p).stem)
    df.to_csv(out / "stats.csv", index=False)

    rep = {"counts": df.split.value_counts().to_dict(), "metadata_rows": len(meta),
           "metadata_label_counts": meta.label.value_counts().to_dict(),
           "metadata_duplicates": int(meta.image_id.duplicated().sum())}
    tr = df[df.split != "test"].merge(meta, on="image_id", how="outer", indicator=True)
    rep["in_folder_not_metadata"] = int((tr._merge == "left_only").sum())
    rep["in_metadata_not_folder"] = int((tr._merge == "right_only").sum())
    both = tr[tr._merge == "both"]
    rep["folder_label_disagrees_with_metadata"] = int(((both.split == "fake").astype(int) != both.label).sum())

    for col in ("format", "mode", "subsampling"):
        rep[col] = {s: g[col].value_counts().to_dict() for s, g in df.groupby("split")}
    rep["size"] = {s: Counter(zip(g.w, g.h)).most_common(5).__repr__() for s, g in df.groupby("split")}
    rep["quality"] = {s: g.quality.value_counts().head(8).to_dict() for s, g in df.groupby("split")}

    tt = df[df.split != "test"].copy()
    y = (tt.split == "fake").astype(int).values
    sep = {}
    for c in ("bytes", "quality", "mean", "std", "lap_var", "hp_std", "hf_ratio", "blockiness", "sat"):
        v = tt[c].values
        x = auc(y, v)
        sep[c] = {"auc_fake_vs_real": round(max(x, 1 - x), 4), "real_mean": round(float(v[y == 0].mean()), 4),
                  "fake_mean": round(float(v[y == 1].mean()), 4),
                  "test_mean": round(float(df[df.split == "test"][c].mean()), 4)}
    rep["single_feature_separability"] = sep

    def dup_report(key):
        g = defaultdict(list)
        for k, s in zip(df[key], df.split):
            g[k].append(s)
        groups = [v for v in g.values() if len(v) > 1]
        kinds = Counter("+".join(sorted(set(v))) for v in groups)
        return {"groups": len(groups), "images": sum(map(len, groups)), "kinds": dict(kinds)}

    rep["exact_duplicates_md5"] = dup_report("md5")
    rep["perceptual_duplicates_dhash"] = dup_report("dhash")
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))

    rng = np.random.default_rng(0)
    for s in ("real", "fake", "test"):
        p = df[df.split == s].path.values
        sheet(rng.permutation(p), out / f"sheet_{s}.jpg")


if __name__ == "__main__":
    main()
