"""Export the whole ensemble to one TensorFlow SavedModel, written by hand because onnx2tf fails on ConvNeXt and DeiT.
Step 1 (needs PyTorch): python scripts/export_tf_ensemble.py dump --weights model_weights.pt --images <dir> --out tf_dump.npz
Step 2 (needs TensorFlow): python scripts/export_tf_ensemble.py build --dump tf_dump.npz --out tf_savedmodel
Step 2 fails unless every member matches PyTorch on the images from step 1."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
MAX_ERROR = 1e-3


def dump(a):
    import torch
    from inference_script import Member, TestImages

    bundle = torch.load(a.weights, map_location="cpu", weights_only=False)
    ds = TestImages(a.images)
    x = torch.stack([ds[i][0] for i in range(min(16, len(ds)))]).float() / 255
    arrays, meta = {"images": x.permute(0, 2, 3, 1).numpy()}, []
    with torch.inference_mode():
        scores = torch.stack([Member(s).eval()(x) for s in bundle["members"]])
    arrays["member_scores"] = scores.numpy()
    arrays["confidence"] = torch.sigmoid(scores.mean(0) * bundle["temperature"]).numpy()
    for i, s in enumerate(bundle["members"]):
        assert s.get("view", "rgb") == "rgb" and s["img_size"] == 224, "only plain 224 px members are ported"
        meta.append({k: s[k] for k in ("arch", "mean", "std", "logit_mean", "logit_std")})
        for k, v in s["state_dict"].items():
            if k.startswith("backbone.") and v.is_floating_point():
                arrays["m%d|%s" % (i, k[len("backbone."):])] = v.numpy()
    arrays["meta"] = np.array(json.dumps({"members": meta, "temperature": bundle["temperature"], "flip_tta": bundle["flip_tta"]}))
    np.savez(a.out, **arrays)


def build(a):
    import tensorflow as tf

    def linear(x, w):
        return tf.tensordot(x, w["weight"], [[-1], [1]]) + w["bias"]

    def layer_norm(x, w):
        mean, var = tf.nn.moments(x, [-1], keepdims=True)
        return (x - mean) * tf.sqrt(1.0 / (var + 1e-6)) * w["weight"] + w["bias"]

    def conv(x, w, stride=1, pad=0, bias=True):
        kernel = tf.transpose(w["weight"], [2, 3, 1, 0])
        x = tf.nn.conv2d(x, kernel, stride, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        return x + w["bias"] if bias else x

    def batch_norm(x, w):
        return tf.nn.batch_normalization(x, w["running_mean"], w["running_var"], w["bias"], w["weight"], 1e-5)

    def mlp(x, w):
        return linear(tf.nn.gelu(linear(x, w.sub("fc1")), approximate=False), w.sub("fc2"))

    def convnext(x, w):
        x = layer_norm(conv(x, w.sub("stem.0"), 4), w.sub("stem.1"))
        for s, depth in enumerate((3, 3, 9, 3)):
            if s:
                x = conv(layer_norm(x, w.sub("stages.%d.downsample.0" % s)), w.sub("stages.%d.downsample.1" % s), 2)
            for b in range(depth):
                k = w.sub("stages.%d.blocks.%d" % (s, b))
                dw = k.sub("conv_dw")
                # pad first: TF 2.21's CPU graph optimiser crashes when it fuses explicit-padding depthwise conv with the bias add
                y = tf.nn.depthwise_conv2d(tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]]), tf.transpose(dw["weight"], [2, 3, 0, 1]),
                                           [1, 1, 1, 1], "VALID") + dw["bias"]
                x = x + mlp(layer_norm(y, k.sub("norm")), k.sub("mlp")) * k["gamma"]
        return linear(layer_norm(tf.reduce_mean(x, [1, 2]), w.sub("head.norm")), w.sub("head.fc"))

    def attention(x, w):
        n, dim = tf.shape(x)[0], x.shape[-1]
        q, k, v = tf.unstack(tf.reshape(linear(x, w.sub("qkv")), [n, x.shape[1], 3, 6, dim // 6]), axis=2)
        q, k, v = [tf.transpose(t, [0, 2, 1, 3]) for t in (q, k, v)]
        weights = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(dim // 6)))
        return linear(tf.reshape(tf.transpose(tf.matmul(weights, v), [0, 2, 1, 3]), [n, x.shape[1], dim]), w.sub("proj"))

    def deit3(x, w):
        x = conv(x, w.sub("patch_embed.proj"), 16)
        x = tf.reshape(x, [tf.shape(x)[0], 196, 384]) + w["pos_embed"]
        x = tf.concat([tf.tile(w["cls_token"], [tf.shape(x)[0], 1, 1]), x], 1)
        for b in range(12):
            k = w.sub("blocks.%d" % b)
            x = x + attention(layer_norm(x, k.sub("norm1")), k.sub("attn")) * k["ls1.gamma"]
            x = x + mlp(layer_norm(x, k.sub("norm2")), k.sub("mlp")) * k["ls2.gamma"]
        return linear(layer_norm(x, w.sub("norm"))[:, 0], w.sub("head"))

    def bottleneck(x, w, stride):
        y = tf.nn.relu(batch_norm(conv(x, w.sub("conv1"), bias=False), w.sub("bn1")))
        y = tf.nn.relu(batch_norm(conv(y, w.sub("conv2"), stride, 1, False), w.sub("bn2")))
        y = batch_norm(conv(y, w.sub("conv3"), bias=False), w.sub("bn3"))
        if w.has("downsample.1.weight"):  # ResNet-D shrinks the shortcut with average pooling, then a 1x1 convolution
            s = tf.nn.avg_pool2d(x, stride, stride, "VALID") if stride > 1 else x
            x = batch_norm(conv(s, w.sub("downsample.1"), bias=False), w.sub("downsample.2"))
        return tf.nn.relu(x + y)

    def resnet50d(x, w):
        x = tf.nn.relu(batch_norm(conv(x, w.sub("conv1.0"), 2, 1, False), w.sub("conv1.1")))
        x = tf.nn.relu(batch_norm(conv(x, w.sub("conv1.3"), 1, 1, False), w.sub("conv1.4")))
        x = tf.nn.relu(batch_norm(conv(x, w.sub("conv1.6"), 1, 1, False), w.sub("bn1")))
        x = tf.nn.max_pool2d(x, 3, 2, [[0, 0], [1, 1], [1, 1], [0, 0]])
        for s, depth in enumerate((3, 4, 6, 3)):
            for b in range(depth):
                x = bottleneck(x, w.sub("layer%d.%d" % (s + 1, b)), 2 if s and not b else 1)
        return linear(tf.reduce_mean(x, [1, 2]), w.sub("fc"))

    class Weights:
        """Reads one member's weights by their PyTorch names, stored as 16-bit floats to keep the export small."""

        def __init__(self, store, prefix=""):
            self.store, self.prefix = store, prefix

        def __getitem__(self, key):
            return tf.cast(self.store[(self.prefix + key).replace(".", "__")], tf.float32)

        def sub(self, key):
            return Weights(self.store, self.prefix + key + ".")

        def has(self, key):
            return (self.prefix + key).replace(".", "__") in self.store

    class Ensemble(tf.Module):
        def __init__(self, npz, meta):
            super().__init__()
            self.meta, self.stores = meta, []
            for i in range(len(meta["members"])):
                head = "m%d|" % i
                self.stores.append({k[len(head):].replace(".", "__"): tf.Variable(npz[k], trainable=False)
                                    for k in npz.files if k.startswith(head)})

        def member(self, image, i):
            m = self.meta["members"][i]
            net = next(f for name, f in (("convnext", convnext), ("deit3", deit3), ("resnet50d", resnet50d)) if m["arch"].startswith(name))
            x = (image - tf.constant(m["mean"], tf.float32)) / tf.constant(m["std"], tf.float32)
            return (tf.squeeze(net(x, Weights(self.stores[i])), 1) - m["logit_mean"]) / m["logit_std"]

        def scores(self, image):
            return tf.stack([self.member(image, i) for i in range(len(self.meta["members"]))])

        @tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32, name="image")])
        def serving(self, image):
            """image: RGB, 224x224, values 0 to 1. Returns the same confidence as inference_script.py."""
            total = tf.reduce_mean(self.scores(image), 0)
            if self.meta["flip_tta"]:
                total = (total + tf.reduce_mean(self.scores(tf.reverse(image, [2])), 0)) / 2
            return {"confidence": tf.sigmoid(total * self.meta["temperature"]), "members": self.scores(image)}

    npz = np.load(a.dump)
    meta = json.loads(str(npz["meta"]))
    model = Ensemble(npz, meta)
    tf.saved_model.save(model, a.out, signatures={"serving_default": model.serving})

    loaded = tf.saved_model.load(a.out).signatures["serving_default"]
    out = loaded(tf.constant(npz["images"]))
    errors = np.abs(out["members"].numpy() - npz["member_scores"]).max(1)
    for m, e in zip(meta["members"], errors):
        print("%-42s largest score difference %.2e" % (m["arch"], e))
    err = float(np.abs(out["confidence"].numpy() - npz["confidence"]).max())
    print("images %d, members %d, largest confidence difference %.2e" % (len(npz["images"]), len(meta["members"]), err))
    assert err < MAX_ERROR and errors.max() < 10 * MAX_ERROR, "TensorFlow output differs from PyTorch"
    print("TF EXPORT OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["dump", "build"])
    ap.add_argument("--weights", default="model_weights.pt")
    ap.add_argument("--images", default="data/dataset/hackathon_dataset/train/fake")
    ap.add_argument("--dump", default="tf_dump.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    (dump if args.step == "dump" else build)(args)
