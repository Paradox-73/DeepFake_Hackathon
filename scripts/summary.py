"""One line per finished run: best epoch, validation AUC, mean AUC under perturbation, worst perturbation.
Usage: python scripts/summary.py [name prefix]"""
import json
import sys
from pathlib import Path

prefix = sys.argv[1] if len(sys.argv) > 1 else ""
for d in sorted(Path("runs").iterdir()):
    m = d / "metrics.json"
    if not d.name.startswith(prefix) or not m.exists():
        continue
    r = json.loads(m.read_text())
    h = r["history"]
    line = "%-34s best_ep %2d/%2d val %.4f last %.4f %3.0fs/ep" % (
        d.name, r["best_epoch"], len(h), r["val"]["auc"], h[-1]["val_auc"], h[-1]["sec"])
    rob = d / "robust_val.json"
    if rob.exists():
        rows = json.loads(rob.read_text())
        pert = [x for x in rows if x["name"] != "clean"]
        worst = min(pert, key=lambda x: x["auc"])
        line += " | robust mean %.4f worst %.4f (%s)" % (sum(x["auc"] for x in pert) / len(pert), worst["auc"], worst["name"])
    print(line)
