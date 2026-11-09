"""MODEL INTEGRATION POINT (real pipeline).

Classifier : soft-voting ensemble (equal-weight mean of softmax) of EfficientNet-B3, DenseNet201,
             ResNet50, ConvNeXt-Tiny, each with 8/4/2/1-view dihedral TTA.
             No metadata given  -> the 4 B2 (image-only) networks.
             Any metadata given -> the 4 B2 + 4 D2 (image + age/sex/site) networks.
Explainer  : Grad-CAM on ONE member (EfficientNet-B3 by default), target = ensemble's top class.
Segmenter  : U-Net (ResNet34 encoder), largest connected component.

Everything else in the app only depends on the `Prediction` dataclass.

Expected folder (MODEL_DIR, default ./models):
    ensemble_cfg.json   B2_*.pt   D2_*.pt   seg_unet_resnet34.pt
    seg_selection.json  (optional, overrides SEG defaults below)
    thresholds.json     (optional, {"akiec": 0.4, ...} per-class "unsure" thresholds)
Or set DERMASCAN_HF_REPO (and HF_TOKEN for a private repo) to download them at start-up.
"""
from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CLASS_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IOU_CAM_THRESHOLD = 0.5                       # heatmap binarisation level

MODEL_DIR = Path(os.environ.get("DERMASCAN_MODEL_DIR", Path(__file__).parent / "models"))
HF_REPO = os.environ.get("DERMASCAN_HF_REPO", "")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TTA_VIEWS = int(os.environ.get("DERMASCAN_TTA", 8 if DEV.type == "cuda" else 4))   # 1, 2, 4 or 8
CAM_MODEL = "efficientnet_b3"                 # member used for Grad-CAM (one network, not the whole ensemble)
VIEW = 384                                    # size of the square image shown with mask / heatmap / overlay

# ---- segmentation defaults. VERIFY against how the U-Net was trained (or provide seg_selection.json) ----
SEG = dict(file="seg_unet_resnet34.pt", encoder="resnet34", size=256, threshold=0.5,
           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# friendly site (sites.py) -> HAM10000 localization the D2 models were trained on
HAM_SITE = {
    "Scalp": "scalp", "Face": "face", "Ear": "ear", "Nose": "face", "Lip / mouth": "face", "Neck": "neck",
    "Chest (upper trunk)": "chest", "Abdomen (lower trunk)": "abdomen", "Upper back": "back",
    "Lower back": "back", "Side of trunk": "trunk", "Genital area": "genital",
    "Upper arm": "upper extremity", "Forearm": "upper extremity", "Hand (back of hand)": "hand",
    "Palm": "acral", "Thigh (any part)": "lower extremity", "Lower leg": "lower extremity",
    "Foot (top)": "foot", "Sole": "acral", "Unknown": "unknown",
}


@dataclass
class Prediction:
    probs: dict           # {code: prob}
    label: str | None     # class code, or None when "not sure"
    is_unsure: bool
    confidence: float     # top-class probability
    mask: Image.Image     # segmentation mask (L, white = lesion)
    cam: Image.Image      # Grad-CAM heatmap (RGB)
    overlay: Image.Image  # image + mask outline + heatmap
    iou: float            # attention-lesion overlap
    used_metadata: bool = False
    n_networks: int = 0
    cam_source: str = ""


# ----------------------------------------------------------------------------- visual helpers
def attention_lesion_iou(cam01: np.ndarray, mask01: np.ndarray, thr: float = IOU_CAM_THRESHOLD) -> float:
    """cam01 in [0,1], mask01 boolean/0-1, same HxW. IoU between thresholded heatmap and lesion mask."""
    a, b = cam01 >= thr, mask01 > 0.5
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def colorize(cam01: np.ndarray) -> Image.Image:
    x = np.clip(cam01, 0, 1)
    rgb = np.stack([np.clip(1.5 - abs(4 * x - 3), 0, 1), np.clip(1.5 - abs(4 * x - 2), 0, 1),
                    np.clip(1.5 - abs(4 * x - 1), 0, 1)], -1)
    return Image.fromarray((rgb * 255).astype("uint8"))


def make_overlay(img: Image.Image, mask01: np.ndarray, cam01: np.ndarray) -> Image.Image:
    base = np.asarray(img.convert("RGB"), dtype=float)
    heat = np.asarray(colorize(cam01), dtype=float)
    out = 0.6 * base + 0.4 * heat * (cam01[..., None] > 0.2)
    m = mask01 > 0.5
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    out[edge] = (255, 255, 255)
    return Image.fromarray(out.clip(0, 255).astype("uint8"))


# ----------------------------------------------------------------------------- network (same as training)
class Net(nn.Module):
    def __init__(self, name: str, use_meta: bool, meta_dim: int, C: int = 7):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=False, num_classes=0)
        din = self.backbone.num_features
        self.use_meta = use_meta
        if use_meta:
            self.mlp = nn.Sequential(nn.Linear(meta_dim, 64), nn.ReLU(True), nn.Dropout(0.2),
                                     nn.Linear(64, 32), nn.ReLU(True))
            din += 32
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(din, 512), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Dropout(0.4), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Dropout(0.3), nn.Linear(256, C))

    def forward(self, x, mt):
        f = self.backbone(x)
        if self.use_meta:
            f = torch.cat([f, self.mlp(mt)], 1)
        return self.head(f)


def _clean_sd(sd: dict) -> dict:
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {k.replace("backbone.m.", "backbone.", 1): v for k, v in sd.items()}
    return {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}


# ----------------------------------------------------------------------------- loading
def _ensure_files():
    if (MODEL_DIR / "ensemble_cfg.json").exists():
        return
    if HF_REPO:
        from huggingface_hub import snapshot_download
        snapshot_download(HF_REPO, local_dir=MODEL_DIR, token=os.environ.get("HF_TOKEN") or None)
    if not (MODEL_DIR / "ensemble_cfg.json").exists():
        raise FileNotFoundError(
            f"Model files not found in {MODEL_DIR}. Copy ensemble_deploy/* and the segmentation weights there, "
            "or set DERMASCAN_HF_REPO.")


def _build_seg(sd: dict):
    import segmentation_models_pytorch as smp
    sd = _clean_sd(sd)
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    n_out = sd["segmentation_head.0.weight"].shape[0]
    model = smp.Unet(SEG["encoder"], encoder_weights=None, in_channels=3, classes=n_out)
    model.load_state_dict(sd)          # strict: fails loudly if encoder / decoder settings differ from training
    return model, n_out


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    _ensure_files()
    cfg = json.loads((MODEL_DIR / "ensemble_cfg.json").read_text())

    nets = {}
    for tag, members in cfg["members"].items():
        for name, fname in members.items():
            net = Net(name, tag == "D2", cfg["meta_dim"])
            net.load_state_dict(_clean_sd(torch.load(MODEL_DIR / fname, map_location="cpu")))
            for p in net.parameters():
                p.requires_grad_(False)
            nets[(tag, name)] = net.to(DEV).eval()

    seg_cfg = dict(SEG)
    sel = MODEL_DIR / "seg_selection.json"
    if sel.exists():                                    # tolerant override; unknown keys are ignored
        raw = json.loads(sel.read_text())
        for dst, srcs in dict(size=("size", "img_size", "img", "input_size", "image_size"),
                              threshold=("threshold", "thr", "thresh"), encoder=("encoder",),
                              mean=("mean",), std=("std",)).items():
            for s in srcs:
                if s in raw:
                    seg_cfg[dst] = raw[s]
                    break
    seg_cfg["size"] = max(32, int(round(int(seg_cfg["size"]) / 32)) * 32)   # U-Net needs multiples of 32
    seg, n_out = _build_seg(torch.load(MODEL_DIR / seg_cfg["file"], map_location="cpu"))
    seg = seg.to(DEV).eval()
    for p in seg.parameters():
        p.requires_grad_(False)

    thr = {c: 0.35 for c in CLASS_CODES}               # placeholder until thresholds.json exists
    if (MODEL_DIR / "thresholds.json").exists():
        thr.update(json.loads((MODEL_DIR / "thresholds.json").read_text()))
    else:
        print("WARNING: thresholds.json not found, using placeholder 'unsure' thresholds (0.35).")

    print(f"[model_api] {len(nets)} classifiers, seg={seg_cfg}, seg_out={n_out}, TTA={TTA_VIEWS}, device={DEV}")
    return dict(cfg=cfg, nets=nets, seg=seg, seg_cfg=seg_cfg, seg_out=n_out, thr=thr)


# ----------------------------------------------------------------------------- pre-processing
def _center_crop(rgb: np.ndarray) -> np.ndarray:       # identical to training: centre square crop
    h, w = rgb.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return rgb[y0:y0 + s, x0:x0 + s]


def _encode_meta(cfg: dict, age: float | None, sex: str, site: str) -> torch.Tensor:
    d = cfg["missing_defaults"]
    a, b = cfg["age_scale"]
    age_v = ((d["age"] if age is None else float(age)) - a) / b
    sx = (sex or "unknown").lower()
    sx = sx if sx in cfg["sex_cols"] else "unknown"
    loc = HAM_SITE.get(site, "unknown")
    loc = loc if loc in cfg["localizations"] else "unknown"
    vec = [age_v] + [float(sx == c) for c in cfg["sex_cols"]] + [float(loc == c) for c in cfg["localizations"]]
    return torch.tensor([vec], dtype=torch.float32, device=DEV)


def _dihedral(x: torch.Tensor, n: int) -> torch.Tensor:
    views = [torch.rot90(x.flip(3) if flip else x, k, (2, 3)) for flip in (False, True) for k in range(4)]
    idx = {1: [0], 2: [0, 4], 4: [0, 1, 2, 3], 8: list(range(8))}.get(n, list(range(8)))
    return torch.cat([views[i] for i in idx], 0)


def _autocast():
    return torch.autocast("cuda") if DEV.type == "cuda" else contextlib.nullcontext()


# ----------------------------------------------------------------------------- classification
@torch.no_grad()
def _member_logits(net: Net, x: torch.Tensor, mt: torch.Tensor) -> torch.Tensor:
    xs = _dihedral(x, TTA_VIEWS)
    with _autocast():
        out = net(xs, mt.expand(len(xs), -1))
    return out.float().mean(0)                           # mean logits over views (as in training eval)


def _cam_layer(net: Net, name: str):
    b = net.backbone
    if name.startswith("convnext"):
        return b.stages[-1]
    if name.startswith("efficientnet"):
        return b.bn2
    if name.startswith("densenet"):
        return b.features
    return b.layer4                                      # resnet


def _grad_cam(net: Net, name: str, x: torch.Tensor, mt: torch.Tensor, cls: int) -> np.ndarray:
    store = {}

    def fwd(_m, _i, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = _cam_layer(net, name).register_forward_hook(fwd)
    try:
        with torch.enable_grad():
            xin = x.clone().requires_grad_(True)         # params are frozen, so the graph needs an input that requires grad
            net(xin, mt)[0, cls].backward()
    finally:
        h.remove()
    a, g = store["a"].detach(), store["g"]
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(VIEW, VIEW), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    cam -= cam.min()
    return cam / (cam.max() + 1e-8)


# ----------------------------------------------------------------------------- segmentation
@torch.no_grad()
def _segment(M: dict, view_rgb: np.ndarray) -> np.ndarray:
    sc = M["seg_cfg"]
    s = sc["size"]
    x = cv2.resize(view_rgb, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = (x - np.array(sc["mean"], np.float32)) / np.array(sc["std"], np.float32)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEV)
    out = M["seg"](x).float()
    p = torch.sigmoid(out)[0, 0] if M["seg_out"] == 1 else torch.softmax(out, 1)[0, 1]
    p = cv2.resize(p.cpu().numpy(), (VIEW, VIEW), interpolation=cv2.INTER_LINEAR)
    m = (p >= float(sc["threshold"])).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:                                            # keep the largest blob only
        m = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return m


# ----------------------------------------------------------------------------- public entry point
def analyze(image: Image.Image, age: int | None, sex: str, site: str) -> Prediction:
    """image: RGB PIL. age may be None, sex may be "Unknown", site is the friendly label from sites.SITES
    ("Unknown" allowed). Returns a Prediction."""
    M = load_models()
    cfg, nets = M["cfg"], M["nets"]

    crop = _center_crop(np.array(image.convert("RGB")))
    clf = cv2.resize(crop, (cfg["img"], cfg["img"]), interpolation=cv2.INTER_AREA)
    view = cv2.resize(crop, (VIEW, VIEW), interpolation=cv2.INTER_AREA if len(crop) > VIEW else cv2.INTER_CUBIC)

    mean = torch.tensor(cfg["mean"], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=DEV).view(1, 3, 1, 1)
    x = torch.from_numpy(clf).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255
    x = (x - mean) / std

    has_meta = age is not None or (sex or "Unknown") != "Unknown" or (site or "Unknown") != "Unknown"
    tags = cfg["routing"]["any_metadata" if has_meta else "no_metadata"]
    mt = _encode_meta(cfg, age, sex, site) if has_meta else torch.zeros(1, cfg["meta_dim"], device=DEV)

    members = [k for k in nets if k[0] in tags]
    P = np.mean([torch.softmax(_member_logits(nets[k], x, mt), 0).cpu().numpy() for k in members], 0)
    probs = dict(zip(CLASS_CODES, map(float, P)))
    top = max(probs, key=probs.get)
    unsure = probs[top] < M["thr"][top]

    cam_key = next((k for k in members if k[1] == CAM_MODEL and k[0] == ("D2" if has_meta else "B2")), members[0])
    cam = _grad_cam(nets[cam_key], cam_key[1], x, mt, CODE_IDX[top])
    mask = _segment(M, view)

    return Prediction(probs, None if unsure else top, unsure, probs[top],
                      Image.fromarray(mask * 255), colorize(cam),
                      make_overlay(Image.fromarray(view), mask.astype(float), cam),
                      attention_lesion_iou(cam, mask), has_meta, len(members), f"{cam_key[0]} {cam_key[1]}")


CODE_IDX = {c: i for i, c in enumerate(CLASS_CODES)}
