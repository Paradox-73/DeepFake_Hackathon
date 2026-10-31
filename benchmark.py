"""Inference speed / memory report.
python benchmark.py --ckpt runs/exp/best.pt            (model from checkpoint)
python benchmark.py --arch efficientnet_b0 --img-size 224   (random weights, for choosing a backbone)
Reports: params, model size, latency (batch 1) mean/median/p95, throughput at --batch, peak GPU memory,
peak CPU RSS, preprocessing time per image, CPU-only latency."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from dfd.augment import resize_to
from dfd.models import build_model, count_params, load_checkpoint
from dfd.utils import env_info, save_json, seed_everything


def peak_rss_mb():
    p = psutil.Process(os.getpid())
    mi = p.memory_info()
    if hasattr(mi, "peak_wset"):  # Windows
        return mi.peak_wset / 2**20
    import resource  # Linux: KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def timeit(fn, warmup, iters, cuda):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return np.array(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--fp32", action="store_true", help="disable fp16 autocast on GPU")
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--out", default=None, help="json output path")
    a = ap.parse_args()
    seed_everything(0, deterministic=False)

    if a.ckpt:
        model, ck = load_checkpoint(a.ckpt, "cpu")
        size, name = ck["img_size"], ck["arch"]
        size_mb = Path(a.ckpt).stat().st_size / 2**20
    else:
        model = build_model({"arch": a.arch, "pretrained": False}).eval()
        size, name = a.img_size, a.arch
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
    res = {"arch": name, "img_size": size, "params_M": count_params(model) / 1e6, "model_file_MB": size_mb,
           "weights_fp32_MB": count_params(model) * 4 / 2**20, "env": env_info()}

    # preprocessing cost (decode-free): resize 1280x720 -> size, to tensor
    img = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    t = timeit(lambda: torch.from_numpy(resize_to(img, size)).permute(2, 0, 1).contiguous(), 5, 50, False)
    res["preprocess_ms_per_image_720p"] = float(np.median(t))

    cuda = torch.cuda.is_available()
    if cuda:
        dev = torch.device("cuda")
        m = model.to(dev).eval().to(memory_format=torch.channels_last)
        amp = not a.fp32
        torch.cuda.reset_peak_memory_stats()
        x1 = torch.rand(1, 3, size, size, device=dev).contiguous(memory_format=torch.channels_last)
        xb = torch.rand(a.batch, 3, size, size, device=dev).contiguous(memory_format=torch.channels_last)

        def run(x):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                return m(x)

        lat = timeit(lambda: run(x1), a.warmup, a.iters, True)
        tb = timeit(lambda: run(xb), 5, max(10, a.iters // 5), True)
        res["gpu"] = {"device": torch.cuda.get_device_name(0), "precision": "fp16" if amp else "fp32",
                      "latency_ms_bs1": {"mean": float(lat.mean()), "median": float(np.median(lat)),
                                         "p95": float(np.percentile(lat, 95))},
                      "throughput_img_per_s_bs%d" % a.batch: float(a.batch / (tb.mean() / 1000)),
                      "peak_gpu_mem_MB": torch.cuda.max_memory_allocated() / 2**20}
        m.cpu()
        del x1, xb
        torch.cuda.empty_cache()
    if not a.no_cpu:
        m = model.cpu().eval().float()
        x1 = torch.rand(1, 3, size, size)
        xb = torch.rand(min(a.batch, 16), 3, size, size)
        with torch.no_grad():
            lat = timeit(lambda: m(x1), 5, max(10, a.iters // 4), False)
            tb = timeit(lambda: m(xb), 2, 5, False)
        res["cpu"] = {"threads": torch.get_num_threads(),
                      "latency_ms_bs1": {"mean": float(lat.mean()), "median": float(np.median(lat)),
                                         "p95": float(np.percentile(lat, 95))},
                      "throughput_img_per_s": float(xb.shape[0] / (tb.mean() / 1000))}
    res["peak_cpu_rss_MB"] = peak_rss_mb()

    out = a.out or (str(Path(a.ckpt).parent / "benchmark.json") if a.ckpt else "benchmark.json")
    save_json(res, out)
    print("\n=== Benchmark: %s @ %dpx ===" % (name, size))
    print("params %.2fM | model file %.1f MB" % (res["params_M"], size_mb))
    if "gpu" in res:
        g = res["gpu"]
        print("GPU %s (%s): bs1 latency median %.2f ms (p95 %.2f) | throughput %.0f img/s | peak GPU mem %.0f MB" % (
            g["device"], g["precision"], g["latency_ms_bs1"]["median"], g["latency_ms_bs1"]["p95"],
            g["throughput_img_per_s_bs%d" % a.batch], g["peak_gpu_mem_MB"]))
    if "cpu" in res:
        c = res["cpu"]
        print("CPU (%d threads): bs1 latency median %.2f ms | throughput %.1f img/s" % (
            c["threads"], c["latency_ms_bs1"]["median"], c["throughput_img_per_s"]))
    print("preprocess (720p resize) %.2f ms | peak process RSS %.0f MB" % (
        res["preprocess_ms_per_image_720p"], res["peak_cpu_rss_MB"]))
    print("saved", out)


if __name__ == "__main__":
    main()
