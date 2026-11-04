"""Report figures: training curves, AUC under each perturbation, and the ROC curve with its confusion matrix.
Usage: python scripts/make_figures.py   (reads runs/*/history.csv and the runs/final/ens_eval*.json and ens_eval_scores.npz
that scripts/ens_eval.py writes for the three shipped families)"""
import json
import math
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FAMILIES = [("lrc25", "ConvNeXt-T", "#2a78d6"), ("lrd25", "DeiT3-S", "#eb6834"), ("lrr25", "ResNet-50D", "#1baf7a")]
OUT = Path("docs/figures")
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "savefig.dpi": 200, "axes.titlelocation": "left", "axes.titlesize": 11, "axes.titleweight": "bold"})


def curves():
    fig, (loss, auc) = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, label, colour in FAMILIES:
        h = pd.concat(pd.read_csv(f) for f in sorted(Path("runs").glob(name + "_f[0-9]/history.csv")))
        m = h.groupby("epoch")[["train_loss", "val_auc"]].mean()
        loss.plot(m.index, m.train_loss, color=colour, lw=2, label=label)
        auc.plot(m.index, m.val_auc, color=colour, lw=2, label=label)
    loss.set(title="Training loss, mean of 5 folds", xlabel="Epoch", ylabel="Binary cross-entropy")
    auc.set(title="Validation AUC, mean of 5 folds", xlabel="Epoch", ylabel="AUC against noisy labels", ylim=(0.85, 0.92))
    auc.legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png")


def robust(src="runs/final/ens_eval.json", out="robustness.png", title="Ensemble AUC under each perturbation, 7,000 out-of-fold images"):
    d = pd.DataFrame(json.loads(Path(src).read_text()))
    order = d[d.name != "clean"].sort_values("ensemble")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    clean = float(d[d.name == "clean"].ensemble.iloc[0])
    lo = min(0.904, math.floor((float(order.ensemble.min()) - 0.001) * 1000) / 1000)  # widen the axis only for lower scores
    ax.axvline(clean, color=MUTED, lw=1.2)
    ax.text(clean - 0.0002, len(order) - 0.4, "clean %.4f" % clean, ha="right", va="bottom", color=INK)
    ax.plot(order.ensemble, order.name, "o", ms=8, color="#2a78d6", mec=SURFACE, mew=2)
    for y, v in zip(order.name, order.ensemble):
        ax.text(0.9175, y, "%.4f" % v, va="center", color=MUTED, fontsize=9)
    ax.set(title=title, xlabel="AUC against noisy labels", xlim=(lo, 0.919), ylim=(-0.7, len(order) + 0.3))
    fig.tight_layout()
    fig.savefig(OUT / out)


def roc(cutoff, src="runs/final/ens_eval_scores.npz"):
    """ROC curve and confusion matrix from the undamaged out-of-fold family scores that scripts/ens_eval.py saves."""
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

    d = np.load(src)
    order = list(d["families"])
    y, fams = d["y"], d["clean"][[order.index(name) for name, _, _ in FAMILIES]]
    ens = fams.mean(0)
    fig, (ax, cm_ax) = plt.subplots(1, 2, figsize=(9, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls="--")
    for (_, label, colour), z in zip(FAMILIES, fams):
        fpr, tpr, _ = roc_curve(y, z)
        ax.plot(fpr, tpr, color=colour, lw=1.2, alpha=0.8, label="%s %.4f" % (label, roc_auc_score(y, z)))
    fpr, tpr, _ = roc_curve(y, ens)
    ax.plot(fpr, tpr, color=INK, lw=2.2, label="Ensemble %.4f" % roc_auc_score(y, ens))
    pred = (ens >= np.log(cutoff / (1 - cutoff))).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    ax.plot(fp / (fp + tn), tp / (tp + fn), "o", ms=8, color=INK, mec=SURFACE, mew=2)
    ax.annotate("cut-off %.3f" % cutoff, (fp / (fp + tn), tp / (tp + fn)), xytext=(14, -16), textcoords="offset points", color=INK)
    ax.set(title="ROC, 7,000 out-of-fold images", xlabel="False positive rate (real called fake)",
           ylabel="True positive rate (fake caught)", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(frameon=False, loc="lower right", labelcolor=INK, title="AUC", title_fontsize=9, fontsize=9)
    m = np.array([[tn, fp], [fn, tp]])
    cm_ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.6)
    for (i, j), v in np.ndenumerate(m):
        cm_ax.text(j, i, "%d\n%.1f%%" % (v, 100 * v / m[i].sum()), ha="center", va="center", color=INK, fontsize=11)
    cm_ax.set(title="Labels at cut-off %.3f" % cutoff, xticks=[0, 1], yticks=[0, 1], xticklabels=["Real", "Fake"],
              yticklabels=["Real", "Fake"], xlabel="Predicted", ylabel="Label as given")
    cm_ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUT / "roc_confusion.png")


if __name__ == "__main__":
    import torch

    curves()
    if Path("runs/final/ens_eval.json").exists():
        robust()
    if Path("runs/final/ens_eval_extra.json").exists():
        robust("runs/final/ens_eval_extra.json", "robustness_extra.png",
               "Ensemble AUC under 21 more kinds of damage, 7,000 out-of-fold images")
    if Path("runs/final/ens_eval_scores.npz").exists() and Path("runs/final/model_weights.pt").exists():
        roc(torch.load("runs/final/model_weights.pt", map_location="cpu", mmap=True, weights_only=False)["threshold"])
