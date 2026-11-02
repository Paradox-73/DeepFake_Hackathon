"""timm backbone + built-in normalisation; single-logit output (>0 means fake)."""
import timm
import torch
import torch.nn as nn

from inference_script import apply_view  # one definition, so training and the deliverable can never disagree

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DeepfakeNet(nn.Module):
    """Input: float tensor in [0,1], shape (B,3,H,W), RGB. Output: logits (B,)."""

    def __init__(self, backbone, mean, std, view="rgb"):
        super().__init__()
        self.backbone = backbone
        self.view_name = view
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((apply_view(x, self.view_name) - self.mean) / self.std).squeeze(1)


class PatchNet(nn.Module):
    """Early stages of a CNN with one logit per location, averaged, so each score sees only a small patch."""

    def __init__(self, arch, stage, drop, pretrained, **extra):
        super().__init__()
        self.features = timm.create_model(arch, pretrained=pretrained, features_only=True, out_indices=(stage,), **extra)
        self.drop = nn.Dropout2d(drop)
        self.head = nn.Conv2d(self.features.feature_info.channels()[-1], 1, 1)

    def forward(self, x):
        return self.head(self.drop(self.features(x)[-1])).flatten(2).mean(2)


def build_model(mcfg):
    arch = mcfg.get("arch", "efficientnet_b0")
    pretrained_path = mcfg.get("pretrained_path")
    extra = {"drop_path_rate": mcfg["drop_path_rate"]} if mcfg.get("drop_path_rate") else {}  # Xception rejects this key
    if mcfg.get("global_pool"):
        extra["global_pool"] = mcfg["global_pool"]
    if mcfg.get("truncate") is not None:
        bb = PatchNet(arch, mcfg["truncate"], mcfg.get("drop_rate", 0.2), bool(mcfg.get("pretrained", True)), **extra)
        return DeepfakeNet(bb, IMAGENET_MEAN, IMAGENET_STD, mcfg.get("view", "rgb"))
    bb = timm.create_model(
        arch, pretrained=bool(mcfg.get("pretrained", True)) and not pretrained_path, num_classes=1,
        drop_rate=mcfg.get("drop_rate", 0.2), **extra)
    if pretrained_path:  # offline ImageNet weights (a timm state dict)
        sd = torch.load(pretrained_path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        own = bb.state_dict()
        sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        bb.load_state_dict(sd, strict=False)
    cfg = getattr(bb, "pretrained_cfg", {}) or {}
    mean, std = cfg.get("mean", IMAGENET_MEAN), cfg.get("std", IMAGENET_STD)
    return DeepfakeNet(bb, tuple(mean), tuple(std), mcfg.get("view", "rgb"))


def save_checkpoint(path, model, mcfg, img_size, threshold, cfg, extra=None):
    torch.save({"state_dict": model.state_dict(), "arch": mcfg.get("arch"), "img_size": img_size,
                "view": mcfg.get("view", "rgb"), "global_pool": mcfg.get("global_pool"), "truncate": mcfg.get("truncate"),
                "mean": model.mean.flatten().tolist(), "std": model.std.flatten().tolist(),
                "threshold": threshold, "config": cfg, "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model({"arch": ck["arch"], "pretrained": False, "drop_rate": 0.0, "view": ck.get("view", "rgb"),
                     "global_pool": ck.get("global_pool"), "truncate": ck.get("truncate")})
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck


def count_params(model):
    return sum(p.numel() for p in model.parameters())
