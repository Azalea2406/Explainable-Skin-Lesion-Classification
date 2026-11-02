import streamlit as st
import auth

u = auth.current_user()
st.title("Account")
with st.container(key="card_acct"):
    st.markdown(f"**{u['name']}**  \n{u['role']} · {u['email']}")
    if st.button("Sign out", icon=":material/logout:"):
        auth.logout(); st.rerun()
