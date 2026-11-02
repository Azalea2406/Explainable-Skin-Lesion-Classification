import streamlit as st
import auth
from styles import inject_css

st.set_page_config(page_title="DermaScan", page_icon=":material/health_and_safety:", layout="wide")
inject_css()
st.logo("assets/logo.svg", size="large")

learn = st.Page("views/learn.py", title="Learn", icon=":material/menu_book:", url_path="learn")
about = st.Page("views/about.py", title="About", icon=":material/info:", url_path="about")

if auth.current_user():
    pages = [st.Page("views/scan.py", title="New scan", icon=":material/add_a_photo:", url_path="scan", default=True),
             learn, about, st.Page("views/account.py", title="Account", icon=":material/person:", url_path="account")]
else:
    pages = [st.Page("views/login.py", title="Sign in", icon=":material/login:", url_path="login", default=True), learn, about]

st.navigation(pages, position="top").run()
