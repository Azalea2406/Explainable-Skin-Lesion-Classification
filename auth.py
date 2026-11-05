"""Clinician-only auth (SQLite + PBKDF2). Prototype-grade: swap for st.login / OIDC in production.
New accounts are 'pending' until approved:  python auth.py approve user@clinic.org
Set DERMASCAN_AUTO_APPROVE=1 (default) to skip approval during development."""
import hashlib, hmac, os, re, secrets, sqlite3, sys
from pathlib import Path
import streamlit as st

DB = Path(os.getenv("DERMASCAN_DB", "data/users.db"))
AUTO_APPROVE = os.getenv("DERMASCAN_AUTO_APPROVE", "1") == "1"
ROLES = ["Dermatologist", "Physician / GP", "Resident", "Nurse practitioner"]

def _con():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS users(email TEXT PRIMARY KEY, name TEXT, role TEXT,
        reg_no TEXT, salt BLOB, pw BLOB, status TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP)""")
    return con

def _hash(pw: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000)

def register(name, email, role, reg_no, password):
    email = email.strip().lower()
    if not (name.strip() and reg_no.strip()): return False, "Please fill in all fields."
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email): return False, "Enter a valid email address."
    if len(password) < 10: return False, "Password must be at least 10 characters."
    salt = secrets.token_bytes(16)
    status = "approved" if AUTO_APPROVE else "pending"
    try:
        with _con() as con:
            con.execute("INSERT INTO users(email,name,role,reg_no,salt,pw,status) VALUES(?,?,?,?,?,?,?)",
                        (email, name.strip(), role, reg_no.strip(), salt, _hash(password, salt), status))
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    return True, ("Account created. You can sign in now." if AUTO_APPROVE
                  else "Account created. An administrator will verify your registration before you can sign in.")

def login(email, password):
    with _con() as con:
        row = con.execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    if not row or not hmac.compare_digest(row["pw"], _hash(password, row["salt"])):
        return False, "Incorrect email or password."
    if row["status"] != "approved":
        return False, "Your account is awaiting verification."
    st.session_state["user"] = {"email": row["email"], "name": row["name"], "role": row["role"]}
    return True, ""

def current_user(): return st.session_state.get("user")

def logout():
    for k in ("user", "cases", "active"): st.session_state.pop(k, None)

if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "approve":
    with _con() as con: con.execute("UPDATE users SET status='approved' WHERE email=?", (sys.argv[2].lower(),))
    print("approved", sys.argv[2])
