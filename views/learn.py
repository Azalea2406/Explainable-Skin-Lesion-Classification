import streamlit as st
import content

st.title("Skin lesion library")
st.caption("Plain-language guides to the lesion classes the model can recognise. Open to everyone.")
with st.container(key="card_featured"):
    st.subheader("The ABCDE rule: what clinicians look for in a mole")
    cols = st.columns(5)
    for col, (k, t, d) in zip(cols, content.ABCDE):
        col.markdown(f"**{k} · {t}**  \n{d}")
flt = st.segmented_control("Filter", ["All", "Malignant", "Pre-cancerous", "Benign"], default="All", label_visibility="collapsed")
items = [(c, i) for c, i in content.CLASSES.items() if flt in (None, "All", i["risk"])]
for row in range(0, len(items), 3):
    cols = st.columns(3)
    for col, (code, info) in zip(cols, items[row:row + 3]):
        with col.container(key=f"card_l_{code}"):
            st.caption(f"{content.RISK_MARK[info['risk']]} {info['risk'].upper()}")
            st.subheader(info["name"])
            st.write(info["about"])
            with st.expander("Typical next steps"):
                st.write("\n".join(f"- {s}" for s in info["steps"]))
st.caption("Educational information only. See a clinician for any skin concern.")
