"""Compare ensembles built from any subset of families, using the per-image scores that ens_eval.py saves.
Each mix is the plain mean of its families' standardised scores, as in inference_script.py.
Usage: python scripts/compare_mixes.py runs/e20/eval_default_scores.npz runs/e20/eval_extra_scores.npz
       --mix old=cfm,dflipmix,a_rn50 new=lrc10,lrd10,lrr10 both=cfm,dflipmix,a_rn50,lrc10,lrd10,lrr10 [--base old]"""
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", nargs="+")
    ap.add_argument("--mix", nargs="+", required=True, help="name=family,family,...")
    ap.add_argument("--base", default=None, help="mix the others are paired against (default: the first)")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    files = [np.load(f) for f in a.scores]
    fams = list(files[0]["families"])
    y = files[0]["y"]
    variants = {}
    for f in files:
        assert list(f["families"]) == fams and np.array_equal(f["y"], y), "score files must cover the same families and images"
        variants.update({k: f[k] for k in f.files if k not in ("families", "y")})
    mixes = {}
    for m in a.mix:
        name, members = m.split("=")
        mixes[name] = [fams.index(x) for x in members.split(",")]
    base = a.base or next(iter(mixes))
    names = ["clean"] + sorted(v for v in variants if v != "clean")
    auc = {m: {v: roc_auc_score(y, variants[v][idx].mean(0)) for v in names} for m, idx in mixes.items()}

    rng = np.random.default_rng(0)
    clean = variants["clean"]
    boots = [rng.integers(0, len(y), len(y)) for _ in range(a.boot)]
    print("%-10s %8s %10s %9s %9s %9s %12s" % ("mix", "clean", "clean diff", "+/- se", "mean dmg", "worst", "shrink 1/8"))
    for m, idx in mixes.items():
        s, b = clean[idx].mean(0), clean[mixes[base]].mean(0)
        d = [roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], b[i]) for i in boots] if m != base else [0.0]
        dmg = [auc[m][v] for v in names[1:]]
        print("%-10s %8.4f %+10.4f %9.4f %9.4f %9.4f %12.4f" % (m, auc[m]["clean"], auc[m]["clean"] - auc[base]["clean"],
              float(np.std(d)), float(np.mean(dmg)), min(dmg), auc[m].get("downscale_0.125", float("nan"))))
    print("\nAUC per damage type, minus the %s mix:" % base)
    print("%-30s" % "damage" + "".join("%10s" % m for m in mixes))
    for v in names:
        print("%-30s" % v + "".join("%+10.4f" % (auc[m][v] - auc[base][v]) for m in mixes))


if __name__ == "__main__":
    main()
