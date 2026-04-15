import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import hashlib
import datetime
import os

# ── Page config (must be FIRST Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="AI Malaria Lab Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Try to import TF / Keras (graceful fallback) ─────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# ── Inject custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   GLOBAL DARK BASE
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* App backgrounds */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.appview-container,
.main {
    background: linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#0f2027 100%) !important;
}

/* Block container padding */
.block-container { padding-top:1.5rem !important; }

/* All generic text */
p, span, div, li, small, td, th,
label, code, pre, .stMarkdown {
    color: #e2e8f0 !important;
}
h1,h2,h3,h4,h5,h6 { color:#e2e8f0 !important; }

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   SIDEBAR
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1b2a 0%,#1a2a3a 100%) !important;
    border-right: 1px solid rgba(0,168,150,0.25) !important;
}
section[data-testid="stSidebar"] * { color:#e2e8f0 !important; }

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   TEXT INPUTS
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stTextInput > div > div > input,
[data-testid="stTextInput"] input,
textarea {
    background-color: #0f172a !important;
    border: 1px solid rgba(0,168,150,0.40) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}
.stTextInput > div > div > input::placeholder,
textarea::placeholder { color:#64748b !important; }
.stTextInput > div > div > input:focus,
textarea:focus {
    border-color:#00A896 !important;
    box-shadow:0 0 0 3px rgba(0,168,150,.15) !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   NUMBER INPUT
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stNumberInput > div > div > input,
[data-testid="stNumberInput"] input {
    background-color: #0f172a !important;
    border: 1px solid rgba(0,168,150,0.40) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}
/* +/- stepper buttons */
.stNumberInput button,
[data-testid="stNumberInput"] button,
[data-testid="stNumberInput"] > div button {
    background-color: #1e293b !important;
    border: 1px solid rgba(0,168,150,0.30) !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
}
.stNumberInput button:hover,
[data-testid="stNumberInput"] button:hover {
    background-color: #00A896 !important;
    color: #fff !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   SELECTBOX
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stSelectbox > div > div,
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div,
[data-baseweb="select"] {
    background-color: #0f172a !important;
    border: 1px solid rgba(0,168,150,0.40) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] svg { color: #e2e8f0 !important; fill:#94a3b8; }
/* Dropdown popup */
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"] {
    background-color: #1e293b !important;
    border: 1px solid rgba(0,168,150,0.25) !important;
}
[role="option"],
[data-baseweb="option"] {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
}
[role="option"]:hover,
[data-baseweb="option"]:hover {
    background-color: rgba(0,168,150,0.18) !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   FILE UPLOADER
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploadDropzone"],
.stFileUploader,
.stFileUploader > div {
    background-color: #1e293b !important;
    border: 2px dashed rgba(0,168,150,0.45) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] small {
    color: #94a3b8 !important;
}
/* Browse files button */
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploader"] button {
    background-color: #0f172a !important;
    border: 1px solid rgba(0,168,150,0.5) !important;
    color: #00A896 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploadDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover {
    background-color: #00A896 !important;
    color: #fff !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   RADIO BUTTONS
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */
[data-testid="stRadio"] > div,
.stRadio > div { background-color:transparent !important; gap:4px; }
[data-testid="stRadio"] label,
.stRadio label {
    background-color: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    padding: 0.4rem 0.8rem !important;
    transition: all .2s;
}
[data-testid="stRadio"] label:has(input:checked),
.stRadio label:has(input:checked) {
    background: linear-gradient(135deg,#00A896,#028090) !important;
    color: #fff !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   ALERTS  (info / warning / success / error)
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */
[data-testid="stAlert"],
.stAlert,
div[role="alert"] {
    border-radius: 12px !important;
    background-color: #1e293b !important;
}
/* info */
[data-testid="stAlert"][data-type="info"],
.stInfo,
[data-baseweb="notification"][kind="info"] {
    background-color: rgba(6,182,212,0.10) !important;
    border-left: 4px solid #06B6D4 !important;
    color: #67e8f9 !important;
}
/* warning */
[data-testid="stAlert"][data-type="warning"],
.stWarning,
[data-baseweb="notification"][kind="warning"] {
    background-color: rgba(245,158,11,0.12) !important;
    border-left: 4px solid #F59E0B !important;
    color: #fcd34d !important;
}
/* success */
[data-testid="stAlert"][data-type="success"],
.stSuccess,
[data-baseweb="notification"][kind="positive"] {
    background-color: rgba(16,185,129,0.10) !important;
    border-left: 4px solid #10B981 !important;
    color: #6ee7b7 !important;
}
/* error */
[data-testid="stAlert"][data-type="error"],
.stError,
[data-baseweb="notification"][kind="negative"] {
    background-color: rgba(239,68,68,0.10) !important;
    border-left: 4px solid #EF4444 !important;
    color: #fca5a5 !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   DATAFRAME
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
.stDataFrame,
.dataframe {
    background-color: #1e293b !important;
    border-radius:12px !important;
}
.dataframe thead th { background:#00A896 !important; color:#fff !important; font-weight:600 !important; }
.dataframe tbody tr:hover { background:rgba(0,168,150,0.08) !important; }
/* Ag-grid / canvas iframe inside dataframe */
[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrame"] canvas { background:#1e293b !important; }

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   DOWNLOAD BUTTON
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg,#00A896,#028090) !important;
    color:#fff !important; border:none !important;
    border-radius:12px !important; font-weight:600 !important;
    box-shadow:0 4px 15px rgba(0,168,150,.35) !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   MAIN ACTION BUTTON
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stButton > button {
    background: linear-gradient(135deg,#00A896,#028090) !important;
    color:#fff !important; border:none !important;
    border-radius:12px !important; font-weight:600 !important;
    font-size:.95rem !important; padding:.6rem 1.8rem !important;
    box-shadow:0 4px 15px rgba(0,168,150,.35) !important;
    transition:all .2s !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 6px 20px rgba(0,168,150,.5) !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   TABS
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(0,168,150,0.08) !important;
    border-radius:12px !important; gap:4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius:10px !important;
    color:#94a3b8 !important; font-weight:500;
    background-color:transparent !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#00A896,#028090) !important;
    color:#fff !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   FORM BORDER OVERRIDE
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */
[data-testid="stForm"],
.stForm {
    background-color:transparent !important;
    border:none !important;
}

/* Divider */
hr { border-color:rgba(0,168,150,0.2) !important; }

/* Scrollbar */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#0f172a; }
::-webkit-scrollbar-thumb { background:#00A896; border-radius:3px; }

/* ━━━━━━━━━━━━━━━━━━━━━━━━━
   COMPONENT CARDS
   ━━━━━━━━━━━━━━━━━━━━━━━━━ */

/* Hero banner */
.hero-banner {
    background:linear-gradient(135deg,#00A896 0%,#028090 50%,#05668D 100%);
    border-radius:20px; padding:2.2rem 2.5rem; margin-bottom:1.8rem;
    display:flex; align-items:center; gap:1.5rem;
    box-shadow:0 8px 32px rgba(0,168,150,.35);
    position:relative; overflow:hidden;
}
.hero-banner::before {
    content:''; position:absolute; top:-40px; right:-40px;
    width:200px; height:200px; border-radius:50%;
    background:rgba(255,255,255,.07);
}
.hero-banner::after {
    content:''; position:absolute; bottom:-60px; left:30%;
    width:280px; height:280px; border-radius:50%;
    background:rgba(255,255,255,.04);
}
.hero-icon { font-size:4rem; filter:drop-shadow(0 4px 8px rgba(0,0,0,.3)); }
.hero-text h1 { color:#fff !important; font-size:2.1rem !important; font-weight:800 !important; margin:0 !important; line-height:1.15; }
.hero-text p  { color:rgba(255,255,255,.85) !important; margin:.3rem 0 0 !important; font-size:1rem; }
.hero-badge {
    margin-left:auto; z-index:1;
    background:rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.3);
    border-radius:50px; padding:.45rem 1.1rem;
    color:#fff !important; font-size:.8rem; font-weight:600;
    backdrop-filter:blur(10px);
}

/* Metric cards */
.metric-card {
    background:linear-gradient(135deg,#1e293b,#1a2332);
    border:1px solid rgba(0,168,150,.2); border-radius:16px;
    padding:1.4rem 1.6rem; text-align:center;
    box-shadow:0 4px 20px rgba(0,0,0,.35);
    transition:transform .2s, box-shadow .2s;
    height:140px; display:flex; flex-direction:column;
    align-items:center; justify-content:center;
}
.metric-card:hover { transform:translateY(-4px); box-shadow:0 8px 30px rgba(0,168,150,.25); }
.metric-icon  { font-size:1.9rem; margin-bottom:.3rem; }
.metric-value { font-size:2.2rem; font-weight:800; color:#00A896; margin:.1rem 0; }
.metric-label { font-size:.75rem; color:#94a3b8; font-weight:500; text-transform:uppercase; letter-spacing:.07em; }

/* Section cards */
.section-card {
    background:linear-gradient(135deg,#1e293b,#1a2332);
    border:1px solid rgba(0,168,150,.18); border-radius:20px;
    padding:1.8rem 2rem; margin-bottom:1.4rem;
    box-shadow:0 4px 24px rgba(0,0,0,.3);
}
.section-title {
    color:#00A896 !important; font-size:1.15rem !important; font-weight:700 !important;
    margin-bottom:1rem !important; display:flex; align-items:center; gap:.5rem;
    border-bottom:1px solid rgba(0,168,150,.2); padding-bottom:.7rem;
}

/* Result boxes */
.result-severe {
    background:linear-gradient(135deg,rgba(239,68,68,.15),rgba(220,38,38,.08));
    border:2px solid #EF4444; border-radius:16px; padding:1.5rem 2rem; text-align:center;
    box-shadow:0 0 30px rgba(239,68,68,.2);
}
.result-nonsevere {
    background:linear-gradient(135deg,rgba(16,185,129,.15),rgba(5,150,105,.08));
    border:2px solid #10B981; border-radius:16px; padding:1.5rem 2rem; text-align:center;
    box-shadow:0 0 30px rgba(16,185,129,.2);
}
.result-label { font-size:1.9rem; font-weight:800; margin:.5rem 0 .2rem; }
.result-conf  { font-size:1rem; color:#94a3b8; }
.result-emoji { font-size:3rem; }

/* Confidence bar */
.conf-bar-outer { background:#1a2332; border-radius:50px; height:14px; overflow:hidden; margin:.8rem 0; }
.conf-bar-inner { height:100%; border-radius:50px; transition:width .8s ease; }

/* Login card */
.login-card {
    background:linear-gradient(145deg,#1e293b,#0f172a);
    border:1px solid rgba(0,168,150,.3); border-radius:24px;
    padding:3rem 3.5rem; width:100%; max-width:440px;
    box-shadow:0 20px 60px rgba(0,0,0,.5);
}
.login-logo  { text-align:center; font-size:3.5rem; margin-bottom:.5rem; }
.login-title { text-align:center; color:#00A896 !important; font-size:1.6rem !important; font-weight:800 !important; margin:0 !important; }
.login-sub   { text-align:center; color:#94a3b8 !important; font-size:.9rem; margin-bottom:2rem !important; }

/* Sidebar user card */
.sidebar-user-card {
    background:rgba(0,168,150,.12); border:1px solid rgba(0,168,150,.3);
    border-radius:14px; padding:1rem 1.2rem; margin-bottom:1.2rem; text-align:center;
}
.sidebar-avatar { font-size:2.5rem; }
.sidebar-uname  { font-weight:700; font-size:1rem; color:#00A896 !important; margin-top:.3rem; }
.sidebar-role   { font-size:.75rem; color:#94a3b8 !important; }

/* Pills */
.pill { display:inline-block; padding:.25rem .75rem; border-radius:50px; font-size:.75rem; font-weight:600; }
.pill-green { background:rgba(16,185,129,.18); color:#10B981 !important; border:1px solid rgba(16,185,129,.35); }
.pill-red   { background:rgba(239,68,68,.15);  color:#EF4444 !important; border:1px solid rgba(239,68,68,.3); }
.pill-blue  { background:rgba(6,182,212,.15);  color:#06B6D4 !important; border:1px solid rgba(6,182,212,.3); }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#   CONSTANTS & USER STORE
# ════════════════════════════════════════════════════════════════
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def load_users() -> dict:
    """Load login credentials from Streamlit secrets.
    Expected secret format in secrets.toml:

    [users.admin]
    password = "admin123"
    role = "Administrator"

    [users.lab1]
    password = "lab1pass"
    role = "Lab Technician"
    """
    users = {}
    secret_users = st.secrets.get("users", {})

    if isinstance(secret_users, dict):
        for uname, creds in secret_users.items():
            if isinstance(creds, dict):
                password = creds.get("password")
                role = creds.get("role")
                if password and role:
                    users[uname] = {"hash": _hash(password), "role": role}
                elif creds.get("hash") and role:
                    users[uname] = {"hash": creds["hash"], "role": role}

    return users

USERS = load_users()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "malaria_severity_model.h5")
IMG_SIZE   = (224, 224)

# ════════════════════════════════════════════════════════════════
#   SESSION STATE INIT
# ════════════════════════════════════════════════════════════════
RECORDS_FILE = os.path.join(os.path.dirname(__file__), "records.json")

def load_records_from_disk():
    if os.path.exists(RECORDS_FILE):
        try:
            import json
            with open(RECORDS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_records_to_disk(records):
    import json
    try:
        with open(RECORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

_defaults = {
    "authenticated": False,
    "username": "",
    "user_role": "",
    "records": [],
    "last_result": None,
    "model": None,
    "model_loaded": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Load persisted records on first run
if "records_loaded" not in st.session_state:
    st.session_state["records"] = load_records_from_disk()
    st.session_state["records_loaded"] = True

# ════════════════════════════════════════════════════════════════
#   MODEL LOADING
# ════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not MODEL_AVAILABLE:
        return None
    try:
        m = keras.models.load_model(path, compile=False)
        return m
    except Exception:
        return None

def get_model():
    if not st.session_state["model_loaded"]:
        m = load_model(MODEL_PATH)
        st.session_state["model"] = m
        st.session_state["model_loaded"] = True
    return st.session_state["model"]

# ════════════════════════════════════════════════════════════════
#   PREDICTION
# ════════════════════════════════════════════════════════════════
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, img: Image.Image):
    arr = preprocess_image(img)
    raw = model.predict(arr, verbose=0)
    # Support both binary (sigmoid) and softmax output
    if raw.shape[-1] == 1:
        prob_severe = float(raw[0][0])
    else:
        prob_severe = float(raw[0][1])
    label      = "Severe" if prob_severe >= 0.5 else "Non-severe"
    confidence = prob_severe if prob_severe >= 0.5 else (1.0 - prob_severe)
    return label, round(confidence * 100, 1)

def demo_predict():
    """Return a plausible random result when no model is available."""
    import random
    label = random.choice(["Severe", "Non-severe"])
    conf  = round(random.uniform(72.0, 98.5), 1)
    return label, conf

# ════════════════════════════════════════════════════════════════
#   LOGIN PAGE
# ════════════════════════════════════════════════════════════════
def show_login():
    col_l, col_c, col_r = st.columns([1, 1.1, 1])
    with col_c:
        st.markdown("""
        <div class="login-card">
            <div class="login-logo">🩺</div>
            <p class="login-title">AI Malaria Lab Assistant</p>
            <p class="login-sub">Secure clinical dashboard · v2.0</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            uname = st.text_input("👤  Username", placeholder="Enter your username")
            pw    = st.text_input("🔑  Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if uname in USERS and USERS[uname]["hash"] == _hash(pw):
                st.session_state["authenticated"] = True
                st.session_state["username"]      = uname
                st.session_state["user_role"]     = USERS[uname]["role"]
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")

        st.markdown("---")
        if not USERS:
            st.warning(
                "No login credentials are configured. Add secure user entries under `users` in Streamlit secrets."
            )

# ════════════════════════════════════════════════════════════════
#   SIDEBAR
# ════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-user-card">
            <div class="sidebar-avatar">{'👨‍⚕️' if st.session_state['user_role']=='Administrator' else '🔬'}</div>
            <div class="sidebar-uname">{st.session_state['username'].upper()}</div>
            <div class="sidebar-role">{st.session_state['user_role']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🗂 Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Dashboard", "🔬 Image Analysis", "📋 Patient Records", "📈 Analytics"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            # Persist records before clearing session
            save_records_to_disk(st.session_state["records"])
            for k in _defaults:
                st.session_state[k] = _defaults[k]
            st.session_state["records_loaded"] = True  # keep flag so records reload correctly
            st.session_state["records"] = load_records_from_disk()
            st.rerun()

        st.markdown("""
        <div style='text-align:center;margin-top:1rem;'>
            <span style='font-size:.7rem;color:#475569;'>AI Malaria Lab Assistant v2.0<br/>© 2026 Ahmad Yahaya</span>
        </div>
        """, unsafe_allow_html=True)

    return page.split(" ", 1)[1].strip()  # strip emoji

# ════════════════════════════════════════════════════════════════
#   HERO BANNER
# ════════════════════════════════════════════════════════════════
def render_hero(title: str, subtitle: str, badge: str):
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-icon">🩺</div>
        <div class="hero-text">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        <div class="hero-badge">{badge}</div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#   PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════
def page_dashboard():
    render_hero(
        "AI Malaria Lab Assistant",
        "Clinical-grade malaria severity classification powered by deep learning.",
        "🟢 System Online",
    )

    records = st.session_state["records"]
    n       = len(records)
    severe  = sum(1 for r in records if r["Severity"] == "Severe")
    nonsev  = n - severe
    avg_conf = round(np.mean([r["Confidence (%)"] for r in records]), 1) if records else 0.0

    # ── Metric cards ──
    c1, c2, c3, c4 = st.columns(4)
    mets = [
        (c1, "🧪", str(n),          "Total Tests",       "#00A896"),
        (c2, "🔴", str(severe),      "Severe Cases",      "#EF4444"),
        (c3, "🟢", str(nonsev),      "Non-Severe Cases",  "#10B981"),
        (c4, "📊", f"{avg_conf}%",   "Avg Confidence",    "#06B6D4"),
    ]
    for col, icon, val, lbl, color in mets:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Microscope visual banner ──
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d1b2a 0%, #0f2027 50%, #1a2a1a 100%);
        border: 1px solid rgba(0,168,150,0.25);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        gap: 2.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
    ">
      <!-- glowing orbs -->
      <div style="position:absolute;top:-60px;right:-60px;width:220px;height:220px;border-radius:50%;background:radial-gradient(circle,rgba(0,168,150,0.18),transparent 70%);pointer-events:none;"></div>
      <div style="position:absolute;bottom:-80px;left:10%;width:280px;height:280px;border-radius:50%;background:radial-gradient(circle,rgba(5,102,141,0.15),transparent 70%);pointer-events:none;"></div>

      <!-- Microscope SVG artwork -->
      <div style="flex-shrink:0;width:140px;height:140px;display:flex;align-items:center;justify-content:center;">
        <svg viewBox="0 0 120 140" xmlns="http://www.w3.org/2000/svg" style="width:120px;height:140px;">
          <!-- base -->
          <rect x="20" y="118" width="80" height="12" rx="6" fill="#00A896" opacity="0.9"/>
          <!-- pillar -->
          <rect x="55" y="60" width="10" height="60" rx="5" fill="#028090"/>
          <!-- arm -->
          <rect x="35" y="58" width="50" height="10" rx="5" fill="#00A896"/>
          <!-- body/tube -->
          <rect x="50" y="18" width="20" height="45" rx="8" fill="#05668D"/>
          <!-- eyepiece -->
          <ellipse cx="60" cy="16" rx="13" ry="7" fill="#00A896"/>
          <!-- objective -->
          <rect x="54" y="60" width="12" height="20" rx="4" fill="#028090"/>
          <!-- lens glint -->
          <circle cx="60" cy="80" r="6" fill="#00A896" opacity="0.7"/>
          <circle cx="57" cy="77" r="2" fill="white" opacity="0.5"/>
          <!-- stage -->
          <rect x="28" y="86" width="64" height="8" rx="4" fill="#1e293b" stroke="#00A896" stroke-width="1"/>
          <!-- slide -->
          <rect x="36" y="84" width="48" height="4" rx="2" fill="#e2e8f0" opacity="0.7"/>
          <!-- blood cells dots -->
          <circle cx="46" cy="86" r="3" fill="#EF4444" opacity="0.8"/>
          <circle cx="58" cy="85" r="2.5" fill="#EF4444" opacity="0.6"/>
          <circle cx="70" cy="87" r="3" fill="#EF4444" opacity="0.8"/>
        </svg>
      </div>

      <!-- Text content -->
      <div style="flex:1;z-index:1;">
        <div style="font-size:1.55rem;font-weight:800;color:#e2e8f0;margin-bottom:.4rem;">
          🔬 Microscopic Blood Smear Analysis
        </div>
        <div style="color:rgba(226,232,240,0.72);font-size:.93rem;line-height:1.6;margin-bottom:1rem;">
          Our AI-powered CNN model examines blood smear images at the cellular level,
          detecting and classifying <b style="color:#00A896;">Plasmodium falciparum</b> infection severity
          with clinical-grade accuracy.
        </div>
        <div style="display:flex;gap:1rem;flex-wrap:wrap;">
          <span style="background:rgba(0,168,150,0.15);border:1px solid rgba(0,168,150,0.4);border-radius:50px;padding:.3rem .9rem;font-size:.78rem;color:#00A896;font-weight:600;">🧬 Deep Learning CNN</span>
          <span style="background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.35);border-radius:50px;padding:.3rem .9rem;font-size:.78rem;color:#06B6D4;font-weight:600;">⚡ &lt;3s Analysis</span>
          <span style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.35);border-radius:50px;padding:.3rem .9rem;font-size:.78rem;color:#10B981;font-weight:600;">✅ Clinical Grade</span>
        </div>
      </div>

      <!-- Ring visualization -->
      <div style="flex-shrink:0;width:120px;height:120px;position:relative;">
        <svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" style="width:120px;height:120px;">
          <circle cx="60" cy="60" r="54" fill="none" stroke="rgba(0,168,150,0.15)" stroke-width="8"/>
          <circle cx="60" cy="60" r="54" fill="none" stroke="#00A896" stroke-width="8"
            stroke-dasharray="226" stroke-dashoffset="60" stroke-linecap="round"
            transform="rotate(-90 60 60)"/>
          <circle cx="60" cy="60" r="38" fill="rgba(0,168,150,0.06)"/>
          <circle cx="60" cy="60" r="38" fill="none" stroke="rgba(0,168,150,0.2)" stroke-width="2"/>
          <!-- Parasites -->
          <circle cx="48" cy="52" r="5" fill="#EF4444" opacity="0.85"/>
          <circle cx="68" cy="48" r="4" fill="#EF4444" opacity="0.7"/>
          <circle cx="60" cy="68" r="5" fill="#10B981" opacity="0.85"/>
          <circle cx="75" cy="65" r="3.5" fill="#10B981" opacity="0.7"/>
          <circle cx="45" cy="68" r="4" fill="#EF4444" opacity="0.75"/>
          <text x="60" y="64" text-anchor="middle" fill="white" font-size="8" font-weight="700" opacity="0.9">SMEAR</text>
        </svg>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick guide ──
    st.markdown("""
    <div class="section-card">
    <div class="section-title">📘 Quick Workflow Guide</div>
    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
    """, unsafe_allow_html=True)

    steps = [
        ("1️⃣", "#00A896", "Upload Image", "Go to Image Analysis and upload a JPG/PNG blood smear."),
        ("2️⃣", "#06B6D4", "Enter Patient Info", "Fill in Patient ID, Name, Gender, Age, Facility."),
        ("3️⃣", "#8B5CF6", "Run AI Analysis", "Click Analyze — CNN classifies severity in <3s."),
        ("4️⃣", "#10B981", "Save & Export", "Save to records; export full CSV from Patient Records."),
    ]
    col_set = st.columns(4)
    for i, (num, c, h, body) in enumerate(steps):
        with col_set[i]:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border:1px solid {c}33;border-radius:14px;padding:1rem;height:100%;">
                <div style="font-size:1.6rem;">{num}</div>
                <div style="color:{c};font-weight:700;margin:.3rem 0;">{h}</div>
                <div style="color:#94a3b8;font-size:.82rem;">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#   PAGE: IMAGE ANALYSIS
# ════════════════════════════════════════════════════════════════
def page_analysis():
    render_hero(
        "Image Analysis",
        "Upload a blood smear image and classify malaria severity using AI.",
        "🔬 CNN Model Active",
    )

    with st.spinner("Loading AI model…"):
        model = get_model()

    model_ok = model is not None and MODEL_AVAILABLE

    # Model status banner
    if model_ok:
        st.success("✅ Model loaded successfully — `malaria_severity_model.h5`")
    else:
        st.warning(
            "⚠️ Model not loaded (TensorFlow not available or model file missing). "
            "**Demo mode** — results are simulated for UI testing. Install TensorFlow and ensure "
            "`malaria_severity_model.h5` is in the same folder to enable real predictions."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")

    # ── Left: Upload + Patient form ──
    with left:
        st.markdown("""<div class="section-card">
        <div class="section-title">📁 Upload Blood Smear Image</div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png"],
            help="Max file size: 10 MB",
            label_visibility="collapsed",
        )
        img = None
        if uploaded:
            if uploaded.size > 10 * 1024 * 1024:
                st.error("❌ File exceeds 10 MB limit. Please upload a smaller image.")
                uploaded = None
            else:
                img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Patient form ──
        st.markdown("""<div class="section-card">
        <div class="section-title">🧑‍⚕️ Patient Information</div>""", unsafe_allow_html=True)

        with st.form("patient_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                pid   = st.text_input("Patient ID *", placeholder="e.g. P001234")
                pname = st.text_input("Patient Name *", placeholder="e.g. Ahmad Musa")
            with c2:
                gender  = st.selectbox("Gender *", ["Male", "Female", "Other"])
                age     = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
            facility = st.text_input("Facility *", placeholder="e.g. Kano General Hospital")

            analyze_btn = st.form_submit_button("🔍 Analyze & Save", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Right: Results ──
    with right:
        st.markdown("""<div class="section-card">
        <div class="section-title">📊 Analysis Result</div>""", unsafe_allow_html=True)

        # TOP of result box: microscope placeholder OR uploaded blood smear image
        if img is not None and uploaded is not None:
            st.image(img, caption=f"📸 {uploaded.name}", use_container_width=True)
        else:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                        padding:1.8rem 1rem;gap:.8rem;
                        background:rgba(0,168,150,0.04);border-radius:14px;
                        border:1px dashed rgba(0,168,150,0.22);margin-bottom:.5rem;">
                <div style="font-size:3.4rem;filter:drop-shadow(0 4px 10px rgba(0,168,150,.3));">🔬</div>
                <div style="color:#94a3b8;font-size:.93rem;text-align:center;line-height:1.6;">
                    Upload a blood smear image and fill in patient<br>details, then click <b style="color:#00A896;">Analyze &amp; Save</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:rgba(0,168,150,0.2);margin:.6rem 0 .8rem;'>", unsafe_allow_html=True)

        if analyze_btn:
            # Validation
            errors = []
            if not uploaded:  errors.append("• Please upload a blood smear image.")
            if not pid.strip():    errors.append("• Patient ID is required.")
            if not pname.strip():  errors.append("• Patient Name is required.")
            if not facility.strip(): errors.append("• Facility is required.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                with st.spinner("🧠 Running AI analysis…"):
                    import time
                    time.sleep(0.4)  # UX pause
                    if model_ok:
                        label, conf = predict(model, img)
                    else:
                        label, conf = demo_predict()

                # Store
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                record = {
                    "Date":           now,
                    "Patient ID":     pid.strip(),
                    "Patient Name":   pname.strip(),
                    "Gender":         gender,
                    "Age":            int(age) if age else None,
                    "Facility":       facility.strip(),
                    "Severity":       label,
                    "Confidence (%)": conf,
                    "Technician":     st.session_state["username"],
                }
                st.session_state["records"].append(record)
                st.session_state["last_result"] = record
                save_records_to_disk(st.session_state["records"])

                # Display result
                sev   = label == "Severe"
                color = "#EF4444" if sev else "#10B981"
                emoji = "🚨" if sev else "✅"

                st.markdown(f"""
                <div class="{'result-severe' if sev else 'result-nonsevere'}">
                    <div class="result-emoji">{emoji}</div>
                    <div class="result-label" style="color:{color};">{label}</div>
                    <div class="result-conf">Confidence: <b>{conf:.1f}%</b></div>
                    <div class="conf-bar-outer">
                        <div class="conf-bar-inner" style="width:{conf:.0f}%;background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:1.2rem;">
                    <table style="width:100%;border-collapse:collapse;font-size:.88rem;">
                        <tr><td style="color:#94a3b8;padding:.3rem .5rem;">Patient ID</td><td style="color:#e2e8f0;font-weight:600;">{pid}</td></tr>
                        <tr><td style="color:#94a3b8;padding:.3rem .5rem;">Name</td><td style="color:#e2e8f0;">{pname}</td></tr>
                        <tr><td style="color:#94a3b8;padding:.3rem .5rem;">Gender / Age</td><td style="color:#e2e8f0;">{gender} / {age if age else '—'}</td></tr>
                        <tr><td style="color:#94a3b8;padding:.3rem .5rem;">Facility</td><td style="color:#e2e8f0;">{facility}</td></tr>
                        <tr><td style="color:#94a3b8;padding:.3rem .5rem;">Technician</td><td style="color:#e2e8f0;">{st.session_state['username']}</td></tr>
                        <tr><td style="color:#94a3b8;padding:.3rem .5rem;">Timestamp</td><td style="color:#e2e8f0;">{now}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

                st.success("✅ Record saved successfully!")
        elif img is not None:
            # Image uploaded but form not submitted yet — show a prompt
            st.markdown("""
            <div style="text-align:center;padding:.6rem;color:#94a3b8;font-size:.9rem;opacity:.85;">
                ✅ Image ready &nbsp;·&nbsp; Fill in patient details and click
                <b style="color:#00A896;">Analyze &amp; Save</b> to run AI classification.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#   PAGE: PATIENT RECORDS
# ════════════════════════════════════════════════════════════════
def page_records():
    render_hero(
        "Patient Records",
        "View, search and export all diagnosis records from this session.",
        f"📋 {len(st.session_state['records'])} Records",
    )

    records = st.session_state["records"]

    if not records:
        st.info("No records yet. Perform an image analysis to create the first record.")
        return

    df = pd.DataFrame(records)

    # ── Filter bar ──
    st.markdown("""<div class="section-card">
    <div class="section-title">🔍 Filter & Search</div>""", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sev_filter = st.selectbox("Severity", ["All", "Severe", "Non-severe"])
    with fc2:
        gender_filter = st.selectbox("Gender", ["All"] + sorted(df["Gender"].unique().tolist()))
    with fc3:
        search = st.text_input("Search Patient Name / ID", placeholder="Type to search…")

    st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters
    fdf = df.copy()
    if sev_filter != "All":
        fdf = fdf[fdf["Severity"] == sev_filter]
    if gender_filter != "All":
        fdf = fdf[fdf["Gender"] == gender_filter]
    if search.strip():
        q = search.strip().lower()
        fdf = fdf[
            fdf["Patient Name"].str.lower().str.contains(q) |
            fdf["Patient ID"].str.lower().str.contains(q)
        ]

    # ── Display table ──
    st.markdown("""<div class="section-card">
    <div class="section-title">📊 Records Table</div>""", unsafe_allow_html=True)

    st.markdown(f"<span style='color:#94a3b8;font-size:.82rem;'>Showing **{len(fdf)}** of **{len(df)}** records</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    def sev_style(v):
        if v == "Severe":
            return "color:#EF4444;font-weight:700"
        return "color:#10B981;font-weight:700"

    # Add S/N column starting from 1 + clean up number formatting
    display_df = fdf.copy().reset_index(drop=True)
    # Age: show integer or dash — never floats
    if "Age" in display_df.columns:
        display_df["Age"] = display_df["Age"].apply(
            lambda x: str(int(x)) if (x is not None and pd.notna(x) and str(x) not in ("", "None", "nan")) else "—"
        )
    # Confidence: 1 decimal place
    if "Confidence (%)" in display_df.columns:
        display_df["Confidence (%)"] = display_df["Confidence (%)"].apply(
            lambda x: f"{float(x):.1f}" if (x is not None and str(x) not in ("", "None", "nan")) else "—"
        )
    display_df.insert(0, "S/N", range(1, len(display_df) + 1))
    styled = display_df.style.applymap(sev_style, subset=["Severity"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Export ──
    st.markdown("""<div class="section-card">
    <div class="section-title">📥 Export Records</div>""", unsafe_allow_html=True)

    # Fix date for Excel: prefix with tab character so Excel treats as text, not date
    export_df = fdf.copy()
    if "Date" in export_df.columns:
        export_df["Date"] = export_df["Date"].astype(str)

    # Build CSV with UTF-8-BOM so Excel opens it correctly
    csv = export_df.to_csv(index=False).encode("utf-8-sig")
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    col_dl, col_print, col_del, col_info = st.columns([1, 1, 1, 1.5])
    with col_dl:
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"malaria_records_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_print:
        if st.button("🖨️ Print Table", use_container_width=True, key="print_btn"):
            st.session_state["show_print"] = True
    with col_del:
        if st.button("🗑️ Clear Records", use_container_width=True, key="clear_recs"):
            st.session_state["records"] = []
            save_records_to_disk(st.session_state["records"])
            st.toast("✅ All records have been deleted.")
            st.rerun()
    with col_info:
        st.markdown(f"""
        <div style="padding:.6rem 1rem;background:rgba(0,168,150,0.08);border-radius:10px;font-size:.83rem;color:#94a3b8;margin-top:.3rem;">
            Exporting <b style="color:#e2e8f0;">{len(fdf)}</b> records with all fields:
            Date · Patient ID · Name · Gender · Age · Facility · Severity · Confidence · Technician
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Word-style printable table ──
    if st.session_state.get("show_print"):
        st.markdown("""<div class="section-card">
        <div class="section-title">🖨️ Print Patient Records</div>""", unsafe_allow_html=True)

        # Build table rows as plain HTML strings (no f-string nesting issues)
        row_parts = []
        for sn, (_, row) in enumerate(fdf.iterrows(), start=1):
            sev_color = "#c00000" if str(row.get("Severity", "")) == "Severe" else "#1f7a1f"
            td = "border:1px solid #000;padding:6px 10px;"
            row_parts.append(
                "<tr>"
                f"<td style='{td}text-align:center;'>{sn}</td>"
                f"<td style='{td}'>{row.get('Date','')}</td>"
                f"<td style='{td}'>{row.get('Patient ID','')}</td>"
                f"<td style='{td}'>{row.get('Patient Name','')}</td>"
                f"<td style='{td}'>{row.get('Gender','')}</td>"
                f"<td style='{td}'>{row.get('Age','')}</td>"
                f"<td style='{td}'>{row.get('Facility','')}</td>"
                f"<td style='{td}color:{sev_color};font-weight:bold;'>{row.get('Severity','')}</td>"
                f"<td style='{td}'>{row.get('Confidence (%)','')}</td>"
                f"<td style='{td}'>{row.get('Technician','')}</td>"
                "</tr>"
            )
        rows_html = "".join(row_parts)

        now_str    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        tech_name  = st.session_state["username"]
        th         = "border:1px solid #000;padding:6px 10px;"

        full_html = (
            "<html><head><meta charset='utf-8'>"
            "<style>"
            "body{font-family:Arial,sans-serif;margin:20px;background:#fff;color:#000;}"
            "table{width:100%;border-collapse:collapse;font-size:9pt;color:#000;}"
            "h2{text-align:center;font-size:14pt;margin-bottom:4px;color:#000;}"
            ".sub{text-align:center;font-size:9pt;color:#555;margin-bottom:10px;}"
            ".foot{font-size:8pt;color:#666;text-align:right;margin-top:8px;}"
            ".print-btn{margin-top:12px;padding:8px 24px;background:#00A896;color:#fff;"
            "border:none;border-radius:8px;font-size:10pt;cursor:pointer;font-weight:bold;}"
            "@media print{.print-btn{display:none!important;}}"
            "</style></head><body>"
            "<div id='print-area'>"
            "<h2>AI Malaria Lab Assistant &mdash; Patient Records</h2>"
            f"<p class='sub'>Generated: {now_str} &nbsp;|&nbsp; Technician: {tech_name}</p>"
            f"<table>"
            f"<thead><tr style='background:#00A896;color:#fff;'>"
            f"<th style='{th}text-align:center;'>S/N</th>"
            f"<th style='{th}'>Date</th>"
            f"<th style='{th}'>Patient ID</th>"
            f"<th style='{th}'>Patient Name</th>"
            f"<th style='{th}'>Gender</th>"
            f"<th style='{th}'>Age</th>"
            f"<th style='{th}'>Facility</th>"
            f"<th style='{th}'>Severity</th>"
            f"<th style='{th}'>Confidence (%)</th>"
            f"<th style='{th}'>Technician</th>"
            f"</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
            "<p class='foot'>AI Malaria Lab Assistant v2.0 &copy; 2026 Ahmad Yahaya</p>"
            "</div>"
            "<button class='print-btn' onclick='window.print()'>&#128424; Print / Save as PDF</button>"
            "</body></html>"
        )

        import streamlit.components.v1 as components
        # Calculate dynamic height: header ~120px + 30px per row
        table_height = 160 + (len(fdf) * 32)
        components.html(full_html, height=table_height, scrolling=True)

        if st.button("✖ Close Print View", key="close_print"):
            st.session_state["show_print"] = False
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#   PAGE: ANALYTICS
# ════════════════════════════════════════════════════════════════
def page_analytics():
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        PLOTLY_OK = True
    except ImportError:
        PLOTLY_OK = False

    render_hero(
        "Analytics Dashboard",
        "Real-time statistics and visualisations of malaria severity trends.",
        "📈 Live Data",
    )

    records = st.session_state["records"]

    if not records:
        st.info("No data yet. Perform analyses to populate the analytics dashboard.")
        return

    df = pd.DataFrame(records)

    # ── Summary metrics ──
    n       = len(df)
    severe  = (df["Severity"] == "Severe").sum()
    nonsev  = n - severe
    avg_c   = df["Confidence (%)"].mean()
    sev_pct = (severe / n * 100) if n else 0

    c1, c2, c3, c4 = st.columns(4)
    mets = [
        (c1, "🧪", str(n),           "Total Tests",     "#00A896"),
        (c2, "🚨", str(severe),       "Severe Cases",    "#EF4444"),
        (c3, "✅", str(nonsev),       "Non-Severe",      "#10B981"),
        (c4, "📊", f"{avg_c:.1f}%",  "Avg Confidence",  "#06B6D4"),
    ]
    for col, icon, val, lbl, color in mets:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not PLOTLY_OK:
        st.warning("Plotly not installed. Install it with `pip install plotly` to see charts.")
        return

    # ── Charts row 1 ──
    ch1, ch2 = st.columns(2)

    chart_bg    = "rgba(0,0,0,0)"
    paper_color = "rgba(0,0,0,0)"
    font_color  = "#94a3b8"

    with ch1:
        st.markdown("""<div class="section-card">
        <div class="section-title">🥧 Severity Distribution</div>""", unsafe_allow_html=True)

        pie = go.Figure(go.Pie(
            labels=["Non-Severe", "Severe"],
            values=[int(nonsev), int(severe)],
            hole=0.55,
            marker=dict(colors=["#10B981", "#EF4444"], line=dict(color="#1e293b", width=3)),
            textinfo="percent+label",
            textfont=dict(color="#fff", size=13),
        ))
        pie.update_layout(
            paper_bgcolor=paper_color, plot_bgcolor=chart_bg,
            font=dict(color=font_color, family="Inter"),
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(font=dict(color="#e2e8f0")),
            showlegend=True,
            annotations=[dict(text=f"<b>{sev_pct:.0f}%</b><br>Severe",
                              x=0.5, y=0.5, font_size=14, font_color="#EF4444", showarrow=False)],
        )
        st.plotly_chart(pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ch2:
        st.markdown("""<div class="section-card">
        <div class="section-title">📊 Confidence Distribution</div>""", unsafe_allow_html=True)

        hist = go.Figure()
        for sev_val, color in [("Non-severe", "#10B981"), ("Severe", "#EF4444")]:
            sub = df[df["Severity"] == sev_val]["Confidence (%)"]
            if not sub.empty:
                hist.add_trace(go.Histogram(
                    x=sub, name=sev_val, marker_color=color, opacity=0.78,
                    xbins=dict(start=50, end=101, size=5),
                ))
        hist.update_layout(
            paper_bgcolor=paper_color, plot_bgcolor=chart_bg,
            font=dict(color=font_color, family="Inter"),
            barmode="overlay", margin=dict(t=20, b=30, l=30, r=20),
            xaxis=dict(title="Confidence (%)", color=font_color, gridcolor="#1e293b"),
            yaxis=dict(title="Count", color=font_color, gridcolor="#1e293b"),
            legend=dict(font=dict(color="#e2e8f0")),
        )
        st.plotly_chart(hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Charts row 2 ──
    if "Facility" in df.columns and df["Facility"].nunique() > 0:
        ch3, ch4 = st.columns(2)

        with ch3:
            st.markdown("""<div class="section-card">
            <div class="section-title">🏥 Cases by Facility</div>""", unsafe_allow_html=True)

            fac_df = df.groupby(["Facility", "Severity"]).size().reset_index(name="Count")
            bar = px.bar(fac_df, x="Facility", y="Count", color="Severity",
                         color_discrete_map={"Severe": "#EF4444", "Non-severe": "#10B981"},
                         barmode="group")
            bar.update_layout(
                paper_bgcolor=paper_color, plot_bgcolor=chart_bg,
                font=dict(color=font_color, family="Inter"),
                margin=dict(t=10, b=40, l=30, r=20),
                xaxis=dict(color=font_color, gridcolor="#1e293b"),
                yaxis=dict(color=font_color, gridcolor="#1e293b"),
                legend=dict(font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with ch4:
            st.markdown("""<div class="section-card">
            <div class="section-title">👤 Cases by Gender</div>""", unsafe_allow_html=True)

            gen_df = df.groupby(["Gender", "Severity"]).size().reset_index(name="Count")
            bar2 = px.bar(gen_df, x="Gender", y="Count", color="Severity",
                          color_discrete_map={"Severe": "#EF4444", "Non-severe": "#10B981"},
                          barmode="stack")
            bar2.update_layout(
                paper_bgcolor=paper_color, plot_bgcolor=chart_bg,
                font=dict(color=font_color, family="Inter"),
                margin=dict(t=10, b=40, l=30, r=20),
                xaxis=dict(color=font_color, gridcolor="#1e293b"),
                yaxis=dict(color=font_color, gridcolor="#1e293b"),
                legend=dict(font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(bar2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Charts row 3: Age Distribution + Cases Over Time ──
    ch5, ch6 = st.columns(2)

    with ch5:
        st.markdown("""<div class="section-card">
        <div class="section-title">🎂 Age Distribution</div>""", unsafe_allow_html=True)

        age_df = df.copy()
        age_df["Age"] = pd.to_numeric(age_df["Age"], errors="coerce")
        age_df = age_df.dropna(subset=["Age"])

        if not age_df.empty:
            age_fig = go.Figure()
            for sev_val, color in [("Severe", "#EF4444"), ("Non-severe", "#10B981")]:
                sub = age_df[age_df["Severity"] == sev_val]["Age"]
                if not sub.empty:
                    age_fig.add_trace(go.Histogram(
                        x=sub, name=sev_val, marker_color=color, opacity=0.80,
                        xbins=dict(size=5),
                    ))
            age_fig.update_layout(
                paper_bgcolor=paper_color, plot_bgcolor=chart_bg,
                font=dict(color=font_color, family="Inter"),
                barmode="overlay",
                margin=dict(t=10, b=30, l=30, r=20),
                xaxis=dict(title="Age (years)", color=font_color, gridcolor="#1e293b"),
                yaxis=dict(title="Patients", color=font_color, gridcolor="#1e293b"),
                legend=dict(font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(age_fig, use_container_width=True)
        else:
            st.info("No valid age data available yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    with ch6:
        st.markdown("""<div class="section-card">
        <div class="section-title">📅 Cases Over Time</div>""", unsafe_allow_html=True)

        try:
            time_df = df.copy()
            time_df["Date"] = pd.to_datetime(time_df["Date"], errors="coerce")
            time_df = time_df.dropna(subset=["Date"])
            time_df["Day"] = time_df["Date"].dt.date
            trend = time_df.groupby(["Day", "Severity"]).size().reset_index(name="Count")

            if not trend.empty:
                time_fig = px.line(
                    trend, x="Day", y="Count", color="Severity",
                    color_discrete_map={"Severe": "#EF4444", "Non-severe": "#10B981"},
                    markers=True,
                )
                time_fig.update_traces(line=dict(width=2.5))
                time_fig.update_layout(
                    paper_bgcolor=paper_color, plot_bgcolor=chart_bg,
                    font=dict(color=font_color, family="Inter"),
                    margin=dict(t=10, b=30, l=30, r=20),
                    xaxis=dict(title="Date", color=font_color, gridcolor="#1e293b"),
                    yaxis=dict(title="Cases", color=font_color, gridcolor="#1e293b"),
                    legend=dict(font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(time_fig, use_container_width=True)
            else:
                st.info("Not enough time data to plot trend.")
        except Exception:
            st.info("Timeline data unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Technician summary ──
    st.markdown("""<div class="section-card">
    <div class="section-title">👨‍⚕️ Technician Activity Summary</div>""", unsafe_allow_html=True)

    tech_df = df.groupby("Technician").agg(
        Tests=("Severity", "count"),
        Severe_Cases=("Severity", lambda x: (x == "Severe").sum()),
        Non_Severe_Cases=("Severity", lambda x: (x == "Non-severe").sum()),
        Avg_Confidence=("Confidence (%)", lambda x: round(float(x.mean()), 1)),
    ).reset_index()
    tech_df.columns = ["Technician", "Total Tests", "Severe Cases", "Non-severe Cases", "Avg Confidence (%)"]

    st.dataframe(tech_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#   MAIN ROUTER
# ════════════════════════════════════════════════════════════════
def main():
    if not st.session_state["authenticated"]:
        show_login()
        return

    page = render_sidebar()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Image Analysis":
        page_analysis()
    elif page == "Patient Records":
        page_records()
    elif page == "Analytics":
        page_analytics()

if __name__ == "__main__":
    main()
