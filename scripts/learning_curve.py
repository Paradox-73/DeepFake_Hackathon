"""How much does more training data help? Compare the same validation fold scored by models trained on 60% and 80% of the data.
The 60% models come from scripts/nested_check.sh, trained with one more fold held out.
Usage: python scripts/learning_curve.py"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


def auc(run):
    d = pd.read_csv("runs/%s/val_preds.csv" % run)
    return roc_auc_score(d.label, d.prob)


gains = []
for outer in range(5):
    for small, large in (("nest%d_cx" % outer, "cxd"), ("nest%d_d" % outer, "deit")):
        for k in range(5):
            if Path("runs/%s_f%d/val_preds.csv" % (small, k)).exists():
                a, b = auc("%s_f%d" % (small, k)), auc("%s_f%d" % (large, k))
                gains.append(b - a)
                print("%-9s fold %d: 60%% data %.4f, 80%% data %.4f, gain %+.4f" % (small, k, a, b, b - a))
print("mean gain from 60%% to 80%% of the data: %+.4f over %d pairs" % (sum(gains) / len(gains), len(gains)))
