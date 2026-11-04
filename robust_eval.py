"""Re-score the validation split (or a labeled test set) under perturbations.
python robust_eval.py --config CFG --ckpt runs/exp/best.pt [--on val|test]"""
import argparse
from pathlib import Path

import numpy as np

from dfd.config import load_config
from dfd.data import select_by_src
from dfd.engine import evaluate
from dfd.models import load_checkpoint
from dfd.perturb import apply_chain
from dfd.pipeline import load_section
from dfd.utils import get_logger, load_json, pick_device, save_json, seed_everything

DEFAULT = (
    [{"name": "clean", "chain": [{"op": "identity"}]}]
    + [{"name": "jpeg_q%d" % q, "chain": [{"op": "jpeg", "params": {"quality": q}}]} for q in (90, 70, 50, 30)]
    + [{"name": "blur_s%g" % s, "chain": [{"op": "blur", "params": {"sigma": s}}]} for s in (1, 2, 3)]
    + [{"name": "noise_std%d" % s, "chain": [{"op": "noise", "params": {"std": s}}]} for s in (5, 10, 20)]
    + [{"name": "downscale_%g" % s, "chain": [{"op": "rescale", "params": {"scale": s}}]} for s in (0.5, 0.25)]
    + [{"name": "darker", "chain": [{"op": "brightness", "params": {"factor": 0.6}}]},
       {"name": "brighter", "chain": [{"op": "brightness", "params": {"factor": 1.4}}]},
       {"name": "low_contrast", "chain": [{"op": "contrast", "params": {"factor": 0.6}}]},
       {"name": "grayscale", "chain": [{"op": "grayscale"}]},
       {"name": "pixelate_4", "chain": [{"op": "pixelate", "params": {"block": 4}}]},
       {"name": "salt_pepper_2pct", "chain": [{"op": "salt_pepper", "params": {"amount": 0.02}}]},
       {"name": "cutout", "chain": [{"op": "cutout", "params": {"boxes": 1, "frac": 0.2}}]},
       {"name": "combo_jpeg50_noise10_blur1",
        "chain": [{"op": "jpeg", "params": {"quality": 50}}, {"op": "noise", "params": {"std": 10}},
                  {"op": "blur", "params": {"sigma": 1}}]}])

# Damage the 20 above leave out: colour shifts, geometry, harsher compression and resizing. Used by ens_eval.py --extra.
EXTRA = (
    [{"name": "jpeg_q%d" % q, "chain": [{"op": "jpeg", "params": {"quality": q}}]} for q in (20, 10)]
    + [{"name": "downscale_0.125", "chain": [{"op": "rescale", "params": {"scale": 0.125}}]}]
    + [{"name": "hue_%d" % s, "chain": [{"op": "hue", "params": {"shift": s}}]} for s in (15, 45)]
    + [{"name": "tint_blue", "chain": [{"op": "tint", "params": {"gains": (0.8, 1.0, 1.25)}}]},
       {"name": "tint_warm", "chain": [{"op": "tint", "params": {"gains": (1.25, 1.0, 0.8)}}]},
       {"name": "rgb_to_bgr", "chain": [{"op": "channel_swap", "params": {"order": (2, 1, 0)}}]},
       {"name": "invert", "chain": [{"op": "invert"}]},
       {"name": "gamma_0.6", "chain": [{"op": "gamma", "params": {"g": 0.6}}]},
       {"name": "gamma_1.6", "chain": [{"op": "gamma", "params": {"g": 1.6}}]},
       {"name": "sharpen", "chain": [{"op": "sharpen", "params": {"amount": 1.5}}]},
       {"name": "posterize_3", "chain": [{"op": "posterize", "params": {"bits": 3}}]},
       {"name": "motion_blur_9", "chain": [{"op": "motion_blur", "params": {"size": 9}}]},
       {"name": "speckle", "chain": [{"op": "speckle", "params": {"std": 0.2}}]}]
    + [{"name": "rotate_%d" % d, "chain": [{"op": "rotate", "params": {"degrees": d}}]} for d in (5, -10, 15)]
    + [{"name": "crop_%d" % round(100 * f), "chain": [{"op": "crop", "params": {"frac": f}}]} for f in (0.9, 0.8)]
    + [{"name": "combo_jpeg30_down0.5_noise10",
        "chain": [{"op": "jpeg", "params": {"quality": 30}}, {"op": "rescale", "params": {"scale": 0.5}},
                  {"op": "noise", "params": {"std": 10}}]}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--on", default="val", choices=["val", "test"])
    ap.add_argument("--out", default=None, help="output prefix (default: next to checkpoint)")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    log = get_logger()
    seed = cfg.get("seed", 42)
    seed_everything(seed, cfg.get("deterministic", True))
    device = pick_device(cfg.get("device", "auto"))
    model, ck = load_checkpoint(a.ckpt, device)
    if a.on == "val":
        samples, _ = load_section(cfg, "data")
        split = load_json(Path(a.ckpt).parent / "split.json")
        samples = select_by_src(samples, split["val"])
    else:
        samples, _ = load_section(cfg, "test")
        if any(s.label < 0 for s in samples):
            raise SystemExit("Test set has no labels; use --on val")
    rc = cfg.get("robust", {})
    perts = rc.get("perturbations") or DEFAULT
    ev = {"batch_size": rc.get("batch_size", 64), "num_workers": rc.get("num_workers", 0), "amp": True,
          "aggregate": cfg.get("predict", {}).get("aggregate", "mean")}
    tta = tuple(cfg.get("predict", {}).get("tta", []))
    rows = []
    for p in perts:
        chain = p["chain"]

        def fn(img, rng, chain=chain):
            return apply_chain(img, chain, rng)

        _, _, _, m = evaluate(model, samples, ck["img_size"], device, ev, ck["threshold"], tta, fn, seed)
        rows.append(dict(name=p["name"], **m))
        log.info("%-28s acc %.4f auc %.4f f1 %.4f logloss %.4f", p["name"], m["acc"], m["auc"], m["f1"], m["logloss"])
    clean = next((r for r in rows if r["name"] == "clean"), rows[0])
    pref = a.out or str(Path(a.ckpt).parent / ("robust_" + a.on))
    save_json(rows, pref + ".json")
    with open(pref + ".md", "w") as f:
        f.write("| perturbation | auc | acc | f1 | logloss | auc drop vs clean |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| %s | %.4f | %.4f | %.4f | %.4f | %+.4f |\n" % (
                r["name"], r["auc"], r["acc"], r["f1"], r["logloss"], r["auc"] - clean["auc"]))
        f.write("\nMean AUC over perturbed (non-clean) settings: %.4f\n" %
                np.mean([r["auc"] for r in rows if r["name"] != "clean"]))
    log.info("Wrote %s.json/.md", pref)


if __name__ == "__main__":
    main()
