import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


def compute_metrics(y, p, threshold=0.5):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    out = {"acc": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred, zero_division=0)),
           "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
           "auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan"),
           "threshold": float(threshold), "n": int(len(y))}
    return out


def best_threshold(y, p):
    """Threshold maximising accuracy on the given set (ties -> closest to 0.5)."""
    y = np.asarray(y).astype(int)
    grid = np.linspace(0.05, 0.95, 91)
    accs = np.array([((p >= t).astype(int) == y).mean() for t in grid])
    best = accs.max()
    cands = grid[accs >= best - 1e-12]
    return float(cands[np.argmin(np.abs(cands - 0.5))])
