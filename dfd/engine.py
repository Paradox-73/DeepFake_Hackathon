"""Inference helper and training loop."""
import csv
import math
import time
from pathlib import Path

import numpy as np
import timm.optim
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import ImageDataset, aggregate
from .metrics import best_threshold, compute_metrics
from .models import build_model, count_params, save_checkpoint
from .utils import get_logger

log = get_logger()


def _autocast(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp) and device.type == "cuda")


@torch.no_grad()
def infer(model, samples, img_size, device, batch_size=64, num_workers=0, amp=True, tta=(), perturb=None, seed=0,
          desc="infer"):
    """Returns P(fake) per sample, in order. tta: subset of ['hflip']."""
    model.eval()
    ds = ImageDataset(samples, img_size, augment=None, seed=seed, perturb=perturb)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    out = np.zeros(len(samples), dtype=np.float64)
    for x, _, idx in tqdm(dl, desc=desc, leave=False):
        x = x.to(device, non_blocking=True).float().div_(255).contiguous(memory_format=torch.channels_last)
        with _autocast(device, amp):
            lg = model(x).float()
            if "hflip" in tta:
                lg = (lg + model(torch.flip(x, dims=[3])).float()) / 2
        out[idx.numpy()] = torch.sigmoid(lg).cpu().numpy()
    return out


def evaluate(model, samples, img_size, device, cfg_eval, threshold=0.5, tta=(), perturb=None, seed=0):
    """Frame/image-level inference -> per-src aggregation -> metrics."""
    p = infer(model, samples, img_size, device, cfg_eval.get("batch_size", 64), cfg_eval.get("num_workers", 0),
              cfg_eval.get("amp", True), tta, perturb, seed)
    srcs, ap, al = aggregate(samples, p, cfg_eval.get("aggregate", "mean"))
    return srcs, ap, al, compute_metrics(al, ap, threshold)


def _make_loader(train_samples, cfg, seed):
    tcfg = cfg["train"]
    ds = ImageDataset(train_samples, cfg["model"]["img_size"], augment=cfg.get("augment", {}), seed=seed)
    labels = np.array([s.label for s in train_samples])
    n_pos, n_neg = max(1, int((labels == 1).sum())), max(1, int((labels == 0).sum()))
    g = torch.Generator().manual_seed(seed)
    mode = tcfg.get("imbalance", "pos_weight")
    sampler = None
    if mode == "sampler":
        w = np.where(labels == 1, 0.5 / n_pos, 0.5 / n_neg)
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(labels), replacement=True, generator=g)
    dl = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=sampler is None, sampler=sampler, generator=g,
                    num_workers=tcfg.get("num_workers", 0), pin_memory=True, drop_last=len(ds) > tcfg["batch_size"])
    pos_weight = n_neg / n_pos if mode == "pos_weight" else 1.0
    return ds, dl, pos_weight, (n_neg, n_pos)


def train_loss(lg, yt, pw, gce_q=None):
    """Binary cross-entropy, or generalised cross-entropy when gce_q is set (less pull from wrong labels)."""
    if not gce_q:
        return F.binary_cross_entropy_with_logits(lg, yt, pos_weight=pw)
    p = torch.sigmoid(lg).clamp(1e-6, 1 - 1e-6)
    return (yt * (1 - p ** gce_q) + (1 - yt) * (1 - (1 - p) ** gce_q)).mean() / gce_q


def pairwise_loss(lg, yt):
    """Soft AUC: every pair where one target is above the other pushes its logits apart, weighted by the target gap."""
    w = (yt.unsqueeze(0) - yt.unsqueeze(1)).clamp(min=0)
    return (w * F.softplus(lg.unsqueeze(1) - lg.unsqueeze(0))).sum() / w.sum().clamp(min=1e-6)


def sam_step(model, opt, x, yt, device, tcfg, pw, rho):
    """Sharpness-aware step: move to the worst nearby weights, take the gradient there, step from the original weights.
    bfloat16 needs no gradient scaler, which would otherwise refuse the second unscale of one step."""
    def forward():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return train_loss(model(x).float(), yt, pw, tcfg.get("gce"))

    first = forward()
    first.backward()
    params = [p for p in model.parameters() if p.grad is not None]
    norm = torch.norm(torch.stack([p.grad.norm() for p in params])) + 1e-12
    with torch.no_grad():
        steps = [p.grad * (rho / norm) for p in params]
        for p, e in zip(params, steps):
            p.add_(e)
    opt.zero_grad(set_to_none=True)
    forward().backward()
    with torch.no_grad():
        for p, e in zip(params, steps):
            p.sub_(e)
    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
    opt.step()
    opt.zero_grad(set_to_none=True)
    return first.item()


def train(cfg, train_samples, val_samples, out_dir, device):
    out_dir = Path(out_dir)
    tcfg, mcfg = cfg["train"], cfg["model"]
    seed = cfg.get("seed", 42)
    img_size = mcfg["img_size"]
    model = build_model(mcfg).to(device).to(memory_format=torch.channels_last)
    log.info("Model %s: %.2fM params", mcfg["arch"], count_params(model) / 1e6)
    ds, dl, pos_weight, (n_neg, n_pos) = _make_loader(train_samples, cfg, seed)
    log.info("Train: %d real / %d fake samples, imbalance=%s (pos_weight=%.3f)", n_neg, n_pos,
             tcfg.get("imbalance", "pos_weight"), pos_weight)
    pw = torch.tensor(pos_weight, device=device)
    ema = None
    if tcfg.get("ema"):
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(tcfg["ema"]), use_buffers=True)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.05))
    if tcfg.get("timm_opt"):  # no weight decay on norm layers and, with layer_decay set, slower early layers
        opt = timm.optim.create_optimizer_v2(model.backbone, "adamw", lr=tcfg["lr"], layer_decay=tcfg.get("layer_decay"),
                                             weight_decay=tcfg.get("weight_decay", 0.05))
        for g in opt.param_groups:  # LambdaLR below ignores timm's lr_scale key
            g["lr"] = tcfg["lr"] * g.get("lr_scale", 1.0)
    accum = int(tcfg.get("grad_accum", 1))
    steps_per_epoch = max(1, len(dl) // accum)
    total = steps_per_epoch * tcfg["epochs"]
    warm = int(steps_per_epoch * tcfg.get("warmup_epochs", 1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))) * (1 - 0.01) + 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(tcfg.get("amp", True)) and device.type == "cuda")
    ls = tcfg.get("label_smoothing", 0.0)
    mix_rng = np.random.default_rng(seed)
    rho = tcfg.get("sam")
    assert not (rho and accum > 1), "SAM steps every batch, so it cannot run with grad_accum"

    monitor = tcfg.get("monitor", "auc")  # auc | acc | f1 | logloss
    sign = -1.0 if monitor == "logloss" else 1.0
    best, best_epoch, bad = -1e9, -1, 0
    hist = []
    ev = {"batch_size": tcfg.get("eval_batch_size", tcfg["batch_size"] * 2), "num_workers": tcfg.get("num_workers", 0),
          "amp": tcfg.get("amp", True), "aggregate": cfg.get("predict", {}).get("aggregate", "mean")}
    t_start = time.time()
    for epoch in range(tcfg["epochs"]):
        model.train()
        ds.set_epoch(epoch)
        run, nb, t0 = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(tqdm(dl, desc="epoch %d" % (epoch + 1), leave=False)):
            x = x.to(device, non_blocking=True).float().div_(255).contiguous(memory_format=torch.channels_last)
            y = y.to(device).float()
            yt = y * (1 - ls) + 0.5 * ls if ls else y
            if tcfg.get("boot") and epoch >= tcfg.get("boot_from", 5):  # blend targets with the averaged model's own view of the clean image
                with torch.no_grad(), _autocast(device, tcfg.get("amp", True)):
                    yt = (1 - tcfg["boot"]) * yt + tcfg["boot"] * torch.sigmoid(ema.module(x).float())
            if tcfg.get("cutmix") and mix_rng.random() < 0.5:  # half the batches paste a box from another image instead of mixup
                perm = torch.randperm(x.size(0), device=device)
                h, w = x.shape[-2:]
                r = (1 - float(mix_rng.beta(tcfg["cutmix"], tcfg["cutmix"]))) ** 0.5
                cy, cx = int(mix_rng.integers(h)), int(mix_rng.integers(w))
                y0, y1 = max(0, cy - int(h * r / 2)), min(h, cy + int(h * r / 2))
                x0, x1 = max(0, cx - int(w * r / 2)), min(w, cx + int(w * r / 2))
                x[..., y0:y1, x0:x1] = x[perm][..., y0:y1, x0:x1]
                lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)
                yt = lam * yt + (1 - lam) * yt[perm]
            elif tcfg.get("mixup") and epoch < tcfg["epochs"] - tcfg.get("mixup_off", 0):  # mixed targets resist memorising wrong labels
                lam = float(mix_rng.beta(tcfg["mixup"], tcfg["mixup"]))
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[perm]
                yt = lam * yt + (1 - lam) * yt[perm]
            if rho:
                loss = sam_step(model, opt, x, yt, device, tcfg, pw, rho)
                sched.step()
                if ema is not None:
                    ema.update_parameters(model)
                run += loss
                nb += 1
                continue
            with _autocast(device, tcfg.get("amp", True)):
                lg = model(x)
            loss = train_loss(lg.float(), yt, pw, tcfg.get("gce"))
            if tcfg.get("pairwise"):
                loss = loss + tcfg["pairwise"] * pairwise_loss(lg.float(), yt)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema is not None:
                    ema.update_parameters(model)
            run += loss.item()
            nb += 1
        ev_model = ema.module if ema is not None else model
        srcs, vp, vl, m = evaluate(ev_model, val_samples, img_size, device, ev, 0.5)
        row = {"epoch": epoch + 1, "train_loss": run / max(1, nb), "lr": opt.param_groups[0]["lr"],
               **{"val_" + k: v for k, v in m.items() if k != "threshold"}, "sec": time.time() - t0}
        hist.append(row)
        score = sign * m[monitor] if not math.isnan(m[monitor]) else -1e9
        if tcfg.get("keep_last"):  # with data.full the validation fold is in training, so its AUC cannot pick an epoch
            score = epoch
        log.info("epoch %d/%d loss %.4f | val acc %.4f auc %.4f f1 %.4f logloss %.4f | %.0fs", epoch + 1,
                 tcfg["epochs"], row["train_loss"], m["acc"], m["auc"], m["f1"], m["logloss"], row["sec"])
        if score > best:
            best, best_epoch, bad = score, epoch + 1, 0
            thr = best_threshold(vl, vp) if tcfg.get("tune_threshold", True) else 0.5
            save_checkpoint(out_dir / "best.pt", ev_model, mcfg, img_size, thr, cfg, {"epoch": epoch + 1, **m})
        else:
            bad += 1
        save_checkpoint(out_dir / "last.pt", ev_model, mcfg, img_size, 0.5, cfg, {"epoch": epoch + 1})
        if epoch + 1 == tcfg.get("stop_epoch"):  # the learning-rate schedule still spans train.epochs
            break
        if bad >= tcfg.get("patience", 5):
            log.info("Early stopping at epoch %d (best epoch %d)", epoch + 1, best_epoch)
            break
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    return {"best_epoch": best_epoch, "train_seconds": time.time() - t_start, "history": hist}
