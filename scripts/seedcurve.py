"""How much does adding a second seed per family lift the ensemble, paired by fold? Run from the project root on the server."""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "scripts")
from oof import family
one = ["cfm", "dflipmix", "a_rn50"]
two = ["s1c", "s1d", "s1r"]
nest = ["ncx", "nd", "nr"]
f = {n: family("runs", n) for n in one + two + nest}
ids = sorted(set.intersection(*[set(f[n].index) for n in f]))
y = f["cfm"].loc[ids, "label"].values
z = {n: f[n].loc[ids, "z"].values for n in f}
fold = np.array([int(l.split(",")[3]) for l in open("configs/folds.csv").read().split("\n")[1:] if l])
ids0 = [l.split(",")[0] for l in open("configs/folds.csv").read().split("\n")[1:] if l]
print(open("configs/folds.csv").readline().strip(), len(ids), len(ids0))
m = dict(zip(ids0, fold))
fo = np.array([m.get(str(i), -1) for i in ids])
print("unmatched", (fo < 0).sum())
def mean(ns): return np.mean([z[n] for n in ns], axis=0)
A, B, C = mean(one), mean(two), mean(one + two)
print("pooled: first seeds %.4f second seeds %.4f both %.4f" % (roc_auc_score(y, A), roc_auc_score(y, B), roc_auc_score(y, C)))
print("nested %.4f" % roc_auc_score(y, mean(nest)))
print("pair gains both minus mean of (A,B) per fold:")
g = []
for k in range(5):
    s = fo == k
    a, b, c = roc_auc_score(y[s], A[s]), roc_auc_score(y[s], B[s]), roc_auc_score(y[s], C[s])
    g.append(c - (a + b) / 2)
    print(k, "%.4f %.4f %.4f %+.4f" % (a, b, c, g[-1]))
print("mean gain %+.4f se %.4f" % (np.mean(g), np.std(g, ddof=1) / 5 ** .5))
for n in one:
    print(n, "%.4f" % roc_auc_score(y, z[n]))
print("corr seeds within family", [round(np.corrcoef(z[a], z[b])[0, 1], 3) for a, b in zip(one, two)])
