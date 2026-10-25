import io
import json

import streamlit as st
from PIL import Image, ImageOps

from inference import DISCLAIMER, Predictor

st.set_page_config(page_title="Skin lesion classifier", page_icon="🩺", layout="wide")

CLASS_INFO = {
    "akiec": ("Actinic keratosis / intraepithelial carcinoma", "pre-malignant or early malignant"),
    "bcc":   ("Basal cell carcinoma", "malignant"),
    "bkl":   ("Benign keratosis-like lesion", "benign"),
    "df":    ("Dermatofibroma", "benign"),
    "mel":   ("Melanoma", "malignant"),
    "nv":    ("Melanocytic nevus (common mole)", "benign"),
    "vasc":  ("Vascular lesion", "benign"),
}
RULES = {"group": "Group flag - refer anything that could be malignant (mel, bcc, akiec)",
         "melanoma": "Melanoma flag - refer anything that could be melanoma"}
THRESHOLDS = {0.75: "0.75 - answers more cases", 0.80: "0.80 - stricter, refers more cases"}


@st.cache_resource(show_spinner="Loading models...")
def get_predictor():
    return Predictor()


@st.cache_data(show_spinner=False, max_entries=32)
def run_models(img_bytes, age, sex, localization):
    """Heavy step (2 models x 8 views). Cached, so changing the rule/threshold does not re-run it."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
    p, per = get_predictor().probabilities(img, age, sex, localization)
    return p, per


pr = get_predictor()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Optional patient details")
    st.caption("Used by one of the two models. Leave blank if unknown.")
    age = st.number_input("Age (years)", min_value=0, max_value=110, value=None, step=1, placeholder="unknown")
    sex = st.selectbox("Sex", ["unknown", "male", "female"], format_func=str.capitalize)
    loc = st.selectbox("Body site", pr.localizations, index=pr.localizations.index("unknown"),
                       format_func=str.capitalize)
    st.header("Safety settings")
    rule = st.radio("Referral rule", list(RULES), format_func=RULES.get)
    thr = st.radio("Confidence threshold", list(THRESHOLDS), format_func=THRESHOLDS.get)
    st.caption("A case is answered only if the model is confident enough and the safety rule does not "
               "fire. Otherwise it is referred to a clinician.")

# ------------------------------------------------------------------ main
st.title("Skin lesion classifier")
st.warning(DISCLAIMER)

uploaded = st.file_uploader("Upload a dermoscopy image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a dermoscopy image (JPG or PNG) to begin.")
    st.stop()

img_bytes = uploaded.getvalue()
try:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
except Exception:
    st.error("This file could not be read as an image.")
    st.stop()

left, right = st.columns([1, 1.3], gap="large")
with left:
    st.image(image, caption=f"{uploaded.name} ({image.width}x{image.height})", width="stretch")
    if min(image.size) < 300:
        st.warning("Low resolution: the model works at 300x300, so small images may be less reliable.")

with st.spinner("Analysing (2 models x 8 views)..."):
    probs, per_model = run_models(img_bytes, None if age is None else float(age), sex, loc)

prof = pr.rules[rule][thr]
status, msg = pr.decide(probs, prof)
p_mal = float(probs[pr.mal].sum())
order = sorted(range(len(pr.classes)), key=lambda i: -probs[i])
top = pr.classes[order[0]]
agree = len({int(q.argmax()) for q in per_model}) == 1

with right:
    if status == "answered":
        name, group = CLASS_INFO[msg]
        st.success(f"### {name}\nCategory: {group}. Confidence {probs[order[0]] * 100:.1f}%.")
        st.caption("This is a model output, not a diagnosis.")
    else:
        st.error(f"### Refer to a clinician\n{msg}.")
        st.caption("The model's most likely class is shown below for information only.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Top class", CLASS_INFO[top][0].split(" (")[0].split(" /")[0], f"{probs[order[0]] * 100:.1f}%")
    c2.metric("P(malignant group)", f"{p_mal * 100:.1f}%", help="Sum of melanoma, basal cell carcinoma and actinic keratosis probabilities")
    c3.metric("Models agree", "Yes" if agree else "No", help="Whether the image-only and image+details models pick the same class")
    if not agree:
        st.caption("The two models disagree, which is a sign of an uncertain case.")

    st.subheader("Class probabilities")
    for i in order:
        st.progress(float(probs[i]), text=f"{CLASS_INFO[pr.classes[i]][0]} - {probs[i] * 100:.1f}%")

with st.expander("Model details"):
    for tag, q in zip(["Model B (image only)", "Model D (image + age, sex, body site)"], per_model):
        t3 = sorted(range(len(q)), key=lambda i: -q[i])[:3]
        st.write(f"**{tag}:** " + ", ".join(f"{pr.classes[i]} {q[i] * 100:.1f}%" for i in t3))
    st.write(f"Rule `{rule}`, threshold {thr:.2f}. Ensemble = average of the two models' calibrated probabilities, "
             "each from 8 flipped/rotated views.")

with st.expander("About and limits"):
    st.markdown(
        "- Trained and validated on **HAM10000** dermoscopy images (7 classes).\n"
        "- On the validation set the default setting answered about 89% of cases at about 97.5% accuracy, "
        "and referred the rest. These numbers were tuned on that set and have **not** been confirmed on independent data.\n"
        "- Phone photos, other devices, and darker skin tones are not well represented in the training data, "
        "so results there are unreliable.\n"
        "- Do not upload images that identify a person to a public deployment.")

st.download_button("Download result (JSON)", json.dumps(dict(
    status=status, message=msg, rule=rule, threshold=thr, p_malignant_group=p_mal,
    probs={c: float(probs[i]) for i, c in enumerate(pr.classes)}), indent=2),
    file_name="result.json", mime="application/json")
