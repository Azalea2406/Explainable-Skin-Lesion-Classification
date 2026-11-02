# DermaScan (Streamlit front end)
    pip install -r requirements.txt
    streamlit run app.py
- Model hook: `model_api.analyze()` (returns `Prediction`); set tuned `THRESHOLDS`.
- Site mapping: `sites.py`. Report: `report.py`. Class text: `content.py` (needs dermatologist review).
- Accounts: SQLite; `DERMASCAN_AUTO_APPROVE=0` then `python auth.py approve email`.
- History is per-session for now (st.session_state); persist to a DB next.
