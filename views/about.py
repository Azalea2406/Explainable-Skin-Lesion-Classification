import streamlit as st

st.title("About DermaScan")
with st.container(key="card_about1"):
    st.subheader("What it does")
    st.write("DermaScan ranks the most likely lesion classes for a skin image, shows where the model looked "
             "(Grad-CAM) and where the lesion is (segmentation), and produces a PDF report. "
             "When confidence is low it says “Not sure” and recommends dermatologist review.")
with st.container(key="card_about2"):
    st.subheader("Who is it for?")
    st.markdown("**For doctors’ assistance**  \n"
                "Helps primary care physicians analyze lesion images and provides structured classification "
                "reports to support their diagnosis.")
    st.markdown("**Teledermatology**  \n"
                "Allows rural clinics to upload images for preliminary risk screening, helping to prioritize "
                "urgent cases such as melanoma.")
    st.markdown("**Transparent clinical reports**  \n"
                "Explains why the AI reached its decision, with both qualitative and quantitative validation.")
    st.markdown("**For general users**  \n"
                "Intended strictly as a first step before visiting a dermatologist. It gives preliminary "
                "information about suspicious skin lesions through image-based screening, so you can decide "
                "when to seek medical advice. It is not a diagnosis and does not replace a doctor.")
