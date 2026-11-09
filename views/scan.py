import io
from datetime import datetime
import streamlit as st
from PIL import Image
import auth, content, model_api, report, sites

S, user = st.session_state, auth.current_user()
S.setdefault("cases", {}); S.setdefault("active", None)
MAP = open("assets/body_map.svg").read()


@st.dialog("Where is the lesion?", width="large")
def body_map():
    st.markdown("<style>[data-testid='stDialog'] [data-testid='stImage']{justify-content:center}</style>", unsafe_allow_html=True)
    st.caption("Upper extremity = arm, forearm, hand · Lower extremity = thigh, leg, foot. "
               "Pick the closest region; use “Unknown” if unsure.")
    zoom = st.slider("Zoom", 60, 160, 100, 10, format="%d%%")
    st.image(MAP, width=int(620 * zoom / 100))

# ---------- sidebar: history ----------
with st.sidebar:
    if st.button("New analysis", type="primary", icon=":material/add:", width="stretch"):
        S.active = None; st.rerun()
    today = datetime.now().date()
    for title, pick in (("TODAY", lambda d: d == today), ("EARLIER", lambda d: d != today)):
        group = [(i, c) for i, c in reversed(S.cases.items()) if pick(c["created"].date())]
        if group: st.caption(title)
        for cid, c in group:
            if st.button(f"{c['case_no']} · {c['meta']['site']}", key=f"h_{cid}", width="stretch",
                         type="primary" if cid == S.active else "secondary"):
                S.active = cid; st.rerun()

# ---------- chat ----------
st.title("New scan")
with st.chat_message("assistant"):
    st.write("Hello, Doctor. Attach a lesion photo and send. "
             "I'll return a ranked list of possible classes and a downloadable report.")

case = S.cases.get(S.active)
if case:
    m = case["meta"]
    with st.chat_message("user"):
        c1, c2 = st.columns([1, 3])
        c1.image(case["image"], width="stretch")
        c2.markdown(f"**Age** {m['age'] if m['age'] is not None else 'not provided'} · **Sex** {m['sex']}  \n**Site** {m['site']}")
    p = case["pred"]
    with st.chat_message("assistant"):
        top = sorted(p.probs.items(), key=lambda kv: -kv[1])[:3]
        st.subheader("Analysis result")
        if p.is_unsure:
            st.warning("**Not sure.** Model confidence is below the decision threshold. Refer to a dermatologist.",
                       icon=":material/help:")
        else:
            info = content.CLASSES[p.label]
            st.markdown(f"#### {info['name']}  ·  `{p.label}`")
            st.caption(f"{content.RISK_MARK[info['risk']]} {info['risk']}")
        for code, pr in top:
            st.progress(pr, text=f"{content.CLASSES[code]['name']} — {pr:.0%}")
        k1, k2 = st.columns(2)
        k1.metric("Classification confidence", f"{p.confidence:.0%}")
        k2.metric("Attention–lesion overlap (IoU)", f"{p.iou:.2f}", help="How much of the model's attention falls on the segmented lesion. Not segmentation accuracy.")
        for col, (img, cap) in zip(st.columns(3), [(p.mask, "Segmentation mask"), (p.cam, "Grad-CAM heatmap"), (p.overlay, "Mask + Grad-CAM overlay")]):
            col.image(img, caption=cap, width="stretch")
        with st.expander("About this finding"):
            info = content.UNSURE if p.is_unsure else content.CLASSES[p.label]
            st.write(info["about"]); st.write("\n".join(f"- {s}" for s in info["steps"]))
        st.download_button("Download PDF report", case["pdf"], file_name=f"{case['report_id']}.pdf", mime="application/pdf",
                           icon=":material/download:", type="primary")
        st.caption("Decision support only, not a diagnosis. Please confirm clinically.")

# ---------- inputs ----------
with st.container(key="card_meta"):
    c1, c2, c3 = st.columns([1, 1, 2])
    age = c1.number_input("Age (optional)", 0, 120, value=None, step=1, placeholder="e.g. 54")
    sex = c2.selectbox("Sex", ["Unknown", "Male", "Female"], index=0)
    site = c3.selectbox("Lesion site", list(sites.SITES), index=list(sites.SITES).index("Unknown"))
    if c3.button("Where is this on the body?", icon=":material/accessibility_new:"):
        body_map()
prompt = st.chat_input("Attach a lesion image and send", accept_file=True, file_type=["jpg", "jpeg", "png"])
if prompt:
    if not prompt.files:
        st.info("Please attach a lesion image. Free-text questions aren't connected yet.")
    else:
        with st.spinner("Analysing…"):
            img = Image.open(io.BytesIO(prompt.files[0].getvalue())).convert("RGB")
            a = int(age) if age is not None else None
            pred = model_api.analyze(img, a, sex, site)
            now, n = datetime.now(), len(S.cases) + 1
            c = dict(created=now, case_no=f"Case {n:04d}", report_id=f"DS-{now:%y%m%d}-{n:03d}",
                     clinician=user["name"], image=img, pred=pred, meta=dict(age=a, sex=sex, site=site))
            c["pdf"] = report.build_report(c)
            S.cases[f"c{n}"] = c; S.active = f"c{n}"
        st.rerun()
