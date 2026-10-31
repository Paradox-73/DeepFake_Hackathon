"""Optional ONNX export. Input: float32 RGB in [0,1], NCHW (normalisation is inside the graph).
Output: P(fake) per image. python export_onnx.py --ckpt runs/exp/best.pt --out runs/exp/model.onnx"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dfd.models import load_checkpoint


class WithSigmoid(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.sigmoid(self.m(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    out = a.out or str(Path(a.ckpt).with_suffix(".onnx"))
    model, ck = load_checkpoint(a.ckpt, "cpu")
    w = WithSigmoid(model).eval()
    s = ck["img_size"]
    x = torch.rand(2, 3, s, s)
    torch.onnx.export(w, x, out, input_names=["image"], output_names=["prob_fake"], opset_version=a.opset,
                      dynamic_axes={"image": {0: "batch"}, "prob_fake": {0: "batch"}}, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x.numpy()})[0]
    with torch.no_grad():
        ref = w(x).numpy()
    err = float(np.abs(got - ref).max())
    print("Exported %s (%.1f MB), max |onnx - torch| = %.2e" % (out, Path(out).stat().st_size / 2**20, err))
    if err > 1e-3:
        raise SystemExit("ONNX output differs from PyTorch more than expected")


if __name__ == "__main__":
    main()
