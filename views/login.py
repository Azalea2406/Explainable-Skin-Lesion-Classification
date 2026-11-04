import streamlit as st
import auth

st.title("Skin lesion analysis for clinical decision support")
st.caption("Clinicians get the analysis tools. The Learn section is open to everyone.")
_, mid, _ = st.columns([1, 2, 1])
with mid.container(key="card_auth"):
    t_in, t_up = st.tabs(["Sign in", "Create account"])
    with t_in, st.form("signin"):
        email = st.text_input("Work email")
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary", width="stretch"):
            ok, msg = auth.login(email, pw)
            st.rerun() if ok else st.error(msg)
    with t_up, st.form("signup"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Full name")
        role = c2.selectbox("Role", auth.ROLES)
        email2 = st.text_input("Work email", key="email2")
        pw2 = st.text_input("Password (min. 10 characters)", type="password", key="pw2")
        agree = st.checkbox("I understand results are decision support, not a diagnosis.")
        if st.form_submit_button("Create account", type="primary", width="stretch"):
            if not agree: st.error("Please accept the statement above.")
            else:
                ok, msg = auth.register(name, email2, role, pw2)
                (st.success if ok else st.error)(msg)
st.caption("Images are processed securely and are not used for training without consent. "
           "Not a substitute for professional medical judgement.")
