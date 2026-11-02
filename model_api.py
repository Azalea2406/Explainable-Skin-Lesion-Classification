"""MODEL INTEGRATION POINT.
Replace the body of `analyze()` with your real pipeline (classifier + segmentation + Grad-CAM).
Everything else in the app only depends on the `Prediction` dataclass below.
The IoU helper is real and can be reused as-is."""
from dataclasses import dataclass
import hashlib
import numpy as np
from PIL import Image
import streamlit as st

CLASS_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
THRESHOLDS = {c: 0.35 for c in CLASS_CODES}   # TODO: paste your tuned per-class thresholds (0.35 only so the dummy output shows both states)
IOU_CAM_THRESHOLD = 0.5                        # heatmap binarisation level

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

@st.cache_resource
def load_models():
    """TODO: load classifier / segmenter weights once (torch.load ... .eval())."""
    return None

def analyze(image: Image.Image, age: int, sex: str, site_vec: list[float]) -> Prediction:
    """image: RGB PIL. Returns a Prediction. DUMMY OUTPUT until the model is wired in."""
    _ = load_models()
    img = image.copy(); img.thumbnail((512, 512))
    seed = int(hashlib.sha256(img.tobytes()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.4, len(CLASS_CODES))
    p = np.exp(logits) / np.exp(logits).sum()
    probs = dict(zip(CLASS_CODES, map(float, p)))
    top = max(probs, key=probs.get)
    unsure = probs[top] < THRESHOLDS[top]
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h * 0.5, w * 0.5
    mask = (((yy - cy) / (h * 0.30)) ** 2 + ((xx - cx) / (w * 0.30)) ** 2) <= 1
    cam = np.exp(-(((yy - cy - h * 0.04) / (h * 0.22)) ** 2 + ((xx - cx + w * 0.03) / (w * 0.22)) ** 2))
    return Prediction(probs, None if unsure else top, unsure, probs[top],
                      Image.fromarray((mask * 255).astype("uint8")), colorize(cam),
                      make_overlay(img, mask.astype(float), cam), attention_lesion_iou(cam, mask))
