import streamlit as st

st.title("About DermaScan")
with st.container(key="card_about1"):
    st.subheader("What it does")
    st.write("DermaScan ranks the most likely lesion classes for a skin image, shows where the model looked "
             "(Grad-CAM) and where the lesion is (segmentation), and produces a PDF report. "
             "When confidence is low it says “Not sure” and recommends dermatologist review.")
with st.container(key="card_about2"):
    st.subheader("Who it is for")
    st.write("Analysis is currently restricted to clinicians as decision support. General users can use the Learn section. It does not provide a diagnosis.")
