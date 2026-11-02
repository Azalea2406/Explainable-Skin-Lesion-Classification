import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600&display=swap');
h1,h2,h3,h4{font-family:'Fraunces',Georgia,serif!important;color:#2F5D4A!important}
header[data-testid="stHeader"]{background:#FBF9EE;border-bottom:1px solid #D9D6C0;box-shadow:0 2px 8px rgba(47,93,74,.08)}
section[data-testid="stSidebar"]{background:#F3F7EC;border-right:1px solid #CFDDC8}
[class*="st-key-card"]{background:#FFFEFA;border-radius:16px;padding:1.25rem 1.4rem;
  box-shadow:0 10px 28px rgba(47,93,74,.15),0 1px 3px rgba(47,93,74,.12)}
[data-testid="stChatMessage"]{background:#FFFEFA;border-radius:16px;padding:1rem 1.2rem;
  box-shadow:0 10px 28px rgba(47,93,74,.15),0 1px 3px rgba(47,93,74,.12)}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){background:#D5E8D3}
[data-testid="stMainBlockContainer"]{max-width:1100px}
</style>
"""

def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
