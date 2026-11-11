"""Run the TensorFlow export of the ensemble on a folder of images. PyTorch is not needed.

    python tf_inference.py --test_dir ./test/images/ --weights ./tf_savedmodel.zip --output predictions_tf.csv

--weights takes tf_savedmodel.zip or the folder it unzips to. Image loading, output format and the label cut-off are the
same as in inference_script.py, so confidence matches predictions.csv to about 1e-4 (PyTorch runs in fp16 on a GPU).
"""
import argparse
import csv
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INPUT_SIZE = 224
THRESHOLD = 0.475  # the cut-off stored in model_weights.pt


def load_image(path):
    """RGB float32 in [0, 1] at 224x224, or None if the file can't be decoded."""
    try:
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):  # cv2.error: an empty file
        return None
    if img is None:
        return None
    if img.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        smaller = img.shape[0] > INPUT_SIZE or img.shape[1] > INPUT_SIZE
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA if smaller else cv2.INTER_LINEAR)
    return np.ascontiguousarray(img[:, :, ::-1]).astype(np.float32) / 255


def main():
    parser = argparse.ArgumentParser(description="Deepfake ML Hackathon inference, TensorFlow export")
    parser.add_argument("--test_dir", required=True, help="Path to test images folder")
    parser.add_argument("--weights", required=True, help="tf_savedmodel.zip or its unzipped folder")
    parser.add_argument("--output", default="predictions_tf.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="label = confidence >= threshold")
    args = parser.parse_args()

    t0 = time.time()
    weights = Path(args.weights)
    if weights.suffix.lower() == ".zip":
        unzipped = Path(tempfile.mkdtemp())
        zipfile.ZipFile(weights).extractall(unzipped)
        weights = unzipped
    model = tf.saved_model.load(str(weights)).signatures["serving_default"]

    paths = sorted((p for p in Path(args.test_dir).iterdir() if p.suffix.lower() in EXTENSIONS), key=lambda p: p.stem)
    conf = np.full(len(paths), np.nan)
    for start in range(0, len(paths), args.batch_size):
        imgs = [load_image(p) for p in paths[start:start + args.batch_size]]
        ok = [i for i, im in enumerate(imgs) if im is not None]
        if ok:
            out = model(image=tf.constant(np.stack([imgs[i] for i in ok])))["confidence"].numpy()
            conf[[start + i for i in ok]] = out
    for i in np.flatnonzero(np.isnan(conf)):
        print("[WARN] unreadable image scored 0.5: %s" % paths[i].name)
        conf[i] = 0.5

    ids = [p.stem for p in paths]
    assert len(ids) == len(set(ids)), "duplicate image ids"
    Path(args.output).resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image_id", "label", "confidence"])
        for i, c in zip(ids, conf):
            w.writerow([i, int(c >= args.threshold), "%.8f" % c])
    print("[DONE] %d rows in %s | %.1f s" % (len(ids), args.output, time.time() - t0))


if __name__ == "__main__":
    main()
