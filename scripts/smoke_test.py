"""End-to-end smoke test on synthetic data. Run from deepfake/:  python scripts/smoke_test.py
Generates tiny fake data (real = smooth images, fake = same + an inverted square patch), then runs train,
predict, benchmark, robust_eval, ONNX export, plus the CSV and video adapters through the real CLIs."""
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "runs" / "smoke"
PY = sys.executable


def smooth(rng, s=160):
    base = rng.integers(0, 255, (6, 6, 3), dtype=np.uint8)
    return cv2.resize(base, (s, s), interpolation=cv2.INTER_CUBIC)


def make_fake(img, rng):
    f = img.copy()
    y, x = rng.integers(0, 100, 2)
    f[y:y + 48, x:x + 48] = 255 - f[y:y + 48, x:x + 48]  # inverted patch = learnable artifact
    return f


def gen():
    if D.exists():
        shutil.rmtree(D)
    rng = np.random.default_rng(0)
    for split, n in (("train", 48), ("test", 16)):
        for cls in ("real", "fake"):
            sub = D / "data" / split / (cls if split == "train" else "")
            sub.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                im = smooth(rng)
                im = make_fake(im, rng) if cls == "fake" else im
                cv2.imwrite(str(sub / ("%s_%03d.png" % (cls, i))), im)
    # csv version of the test set (no labels)
    with open(D / "test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "x"])
        for p in sorted((D / "data" / "test").glob("*.png")):
            w.writerow([p.name, 0])
    # tiny videos
    for cls in ("real", "fake"):
        vd = D / "videos" / "train" / cls
        vd.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            wr = cv2.VideoWriter(str(vd / ("v%d.mp4" % i)), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 160))
            base = smooth(rng)
            for t in range(12):
                fr = np.roll(base, t * 2, axis=1)
                wr.write(make_fake(fr, rng) if cls == "fake" else fr)
            wr.release()


def run(*args):
    print("\n$ python " + " ".join(str(a) for a in args), flush=True)
    t = time.time()
    r = subprocess.run([PY, *map(str, args)], cwd=ROOT)
    if r.returncode != 0:
        sys.exit("FAILED: %s" % " ".join(map(str, args)))
    print("(%.1fs)" % (time.time() - t))


def main():
    t0 = time.time()
    gen()
    cfg = "configs/smoke.yaml"
    ck = "runs/smoke/exp/best.pt"
    run("train.py", "--config", cfg)
    run("predict.py", "--config", cfg, "--ckpt", ck, "--out", "runs/smoke/predictions.csv")
    # CSV adapter on test set, custom column names
    run("predict.py", "--config", cfg, "--ckpt", ck, "--out", "runs/smoke/predictions_csv.csv", "--set",
        "test.adapter=csv", "test.root=runs/smoke/data/test", "test.params.csv=runs/smoke/test.csv",
        "test.params.path_col=file", "test.params.label_col=null", "predict.columns.id=image_name",
        "predict.columns.label=prediction", "predict.columns.prob=null", "predict.id_mode=src")
    run("benchmark.py", "--ckpt", ck, "--iters", "30", "--warmup", "5", "--batch", "16")
    run("robust_eval.py", "--config", cfg, "--ckpt", ck)
    run("export_onnx.py", "--ckpt", ck, "--out", "runs/smoke/model.onnx")
    # video adapter: frames -> train 1 epoch -> per-video predictions
    vset = ["data.adapter=video", "data.root=runs/smoke/videos/train", "data.video.num_frames=4",
            "data.val_fraction=0.25", "output_dir=runs/smoke/exp_video", "train.epochs=1",
            "test.adapter=video", "test.root=runs/smoke/videos/train", "test.params.labeled=false"]
    run("train.py", "--config", cfg, "--set", *vset)
    run("predict.py", "--config", cfg, "--ckpt", "runs/smoke/exp_video/best.pt", "--out",
        "runs/smoke/predictions_video.csv", "--set", *vset)

    rows = list(csv.reader(open(D / "predictions.csv")))
    vrows = list(csv.reader(open(D / "predictions_video.csv")))
    assert rows[0] == ["id", "label", "probability"] and len(rows) == 33, rows[:2]
    assert len(vrows) == 13, len(vrows)  # 12 videos, one row each
    print("\nHead of predictions.csv:", rows[:3])
    print("SMOKE TEST PASSED in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
