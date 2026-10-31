"""Turn videos into cached frame images and optionally face-crop images. Deterministic and cached."""
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .adapters import IMG_EXT, VID_EXT, Sample
from .faces import crop_face, get_detector
from .utils import get_logger

log = get_logger()
_DET = {}


def _detector(face_cfg):
    key = json.dumps(face_cfg, sort_keys=True)
    if key not in _DET:
        _DET[key] = get_detector(face_cfg)
    return _DET[key]


def _write(path, bgr, q):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, q] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, bgr, params)
    buf.tofile(path)


def _maybe_face(bgr, face_cfg):
    det = _detector(face_cfg)
    if det is None:
        return bgr
    box = det.detect(bgr)
    if box is None:
        return bgr  # fallback: full frame
    return crop_face(bgr, box, face_cfg.get("margin", 0.3))


def _read_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        if total > n:
            idxs = np.linspace(0, total - 1, n + 2)[1:-1].round().astype(int)
        else:
            idxs = np.arange(total)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
    if not frames:  # unknown frame count: read sequentially, keep n evenly spaced
        cap.release()
        cap = cv2.VideoCapture(path)
        allf = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            allf.append(fr)
        if allf:
            frames = [allf[int(i)] for i in np.linspace(0, len(allf) - 1, min(n, len(allf)))]
    cap.release()
    return frames


def _job(args):
    path, src, label, cache, vcfg, face_cfg = args
    ext = os.path.splitext(path)[1].lower()
    q = int(vcfg.get("jpeg_quality", 95))
    key = hashlib.sha1((os.path.abspath(path) + json.dumps([vcfg, face_cfg], sort_keys=True)).encode()).hexdigest()[:16]
    if ext in VID_EXT:
        n = int(vcfg.get("num_frames", 8))
        fext = vcfg.get("frame_ext", "jpg")
        outs = [os.path.join(cache, key[:2], "%s_%02d.%s" % (key, i, fext)) for i in range(n)]
        if all(os.path.exists(o) for o in outs):
            return [Sample(o, label, src) for o in outs]
        res = []
        for i, fr in enumerate(_read_frames(path, n)):
            _write(outs[i], _maybe_face(fr, face_cfg), q)
            res.append(Sample(outs[i], label, src))
        return res
    face_on = face_cfg and face_cfg.get("detector", "none") not in (None, "none")
    if ext in IMG_EXT and face_on and face_cfg.get("apply_to_images", True):
        o = os.path.join(cache, key[:2], key + ".png")
        if not os.path.exists(o):
            bgr = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return []
            _write(o, _maybe_face(bgr, face_cfg), q)
        return [Sample(o, label, src)]
    return [Sample(path, label, src)]


def prepare_samples(samples, data_cfg, cache_dir):
    """Returns (samples, failed_srcs). Videos -> frame samples (same src); images pass through
    unless a face detector is configured, then they are face-cropped into the cache."""
    vcfg = data_cfg.get("video", {}) or {}
    face_cfg = data_cfg.get("face", {}) or {}
    face_on = face_cfg.get("detector", "none") not in (None, "none")
    if not (face_on or any(os.path.splitext(s.path)[1].lower() in VID_EXT for s in samples)):
        return samples, []
    jobs = [(s.path, s.src, s.label, cache_dir, vcfg, face_cfg) for s in samples]
    workers = int(data_cfg.get("prepare_workers", 1))
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(tqdm(ex.map(_job, jobs, chunksize=4), total=len(jobs), desc="prepare"))
    else:
        results = [_job(j) for j in tqdm(jobs, desc="prepare")]
    out, failed = [], []
    for s, r in zip(samples, results):
        if r:
            out.extend(r)
        else:
            failed.append(s.src)
    if failed:
        log.warning("prepare: %d items could not be decoded (e.g. %s)", len(failed), failed[:3])
    return out, failed
