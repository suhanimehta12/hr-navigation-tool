import streamlit as st
import sys
import os

# ── Ensure _pages is importable regardless of working directory ──────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="HR Navigator — Workforce Intelligence Platform",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0B0F1A;
    color: #E8EAF0;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    letter-spacing: -0.02em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1628 0%, #111827 100%);
    border-right: 1px solid #1E2A40;
}

section[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif;
    color: #94A3B8 !important;
    font-size: 0.9rem;
    padding: 6px 0;
    transition: color 0.2s;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    color: #38BDF8 !important;
}

/* Cards */
.card {
    background: #111827;
    border: 1px solid #1E2A40;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s, transform 0.2s;
}
.card:hover {
    border-color: #38BDF8;
    transform: translateY(-2px);
}

/* Metric tiles */
.metric-tile {
    background: linear-gradient(135deg, #111827 0%, #0F1628 100%);
    border: 1px solid #1E2A40;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #38BDF8;
    line-height: 1;
}
.metric-label {
    color: #64748B;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
}

/* Risk badges */
.badge-high   { background:#FF4757;color:#fff;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:600; }
.badge-medium { background:#FF9F43;color:#fff;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:600; }
.badge-low    { background:#26DE81;color:#111;padding:4px 12px;border-radius:20px;font-size:0.78rem;font-weight:600; }

/* Page header banner */
.page-header {
    background: linear-gradient(135deg, #0F2744 0%, #0B1D35 100%);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
}
.page-header h1 {
    font-size: 2rem;
    font-weight: 800;
    color: #F1F5F9;
    margin: 0 0 6px 0;
}
.page-header p {
    color: #64748B;
    margin: 0;
    font-size: 0.95rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0EA5E9, #2563EB);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 10px 24px;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}

/* Tables */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Divider */
hr { border-color: #1E2A40; }

/* Selectbox / slider label */
label { color: #94A3B8 !important; font-size: 0.88rem !important; }

/* Success / warning / error boxes */
.stSuccess, .stWarning, .stError, .stInfo {
    border-radius: 10px;
}

/* Progress bar */
.stProgress > div > div { background: linear-gradient(90deg,#0EA5E9,#6366F1); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 28px 0;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#F1F5F9;'>
            🧭 HR Navigator
        </div>
        <div style='color:#334155;font-size:0.75rem;margin-top:4px;letter-spacing:0.08em;text-transform:uppercase;'>
            Workforce Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📄 Recruitment & Resume AI",
            "🔄 Retention & Early Warning",
            "🏆 Promotion Intelligence",
            "📊 Platform Analytics"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#334155;font-size:0.72rem;padding:8px 0;line-height:1.7;'>
        <div style='color:#475569;font-weight:600;margin-bottom:6px;'>HOW IT WORKS</div>
        Candidate data flows from<br>
        Recruitment → Retention →<br>
        Promotion automatically.<br><br>
        One continuous employee<br>
        journey. Zero data resets.
    </div>
    """, unsafe_allow_html=True)

# ── Page Routing ─────────────────────────────────────────────────────────────
if page == "🏠 Home":
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("home", os.path.join(ROOT, "_pages", "home.py"))
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.show()

elif page == "📄 Recruitment & Resume AI":
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("recruitment", os.path.join(ROOT, "_pages", "recruitment.py"))
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.show()

elif page == "🔄 Retention & Early Warning":
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("retention", os.path.join(ROOT, "_pages", "retention.py"))
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.show()

elif page == "🏆 Promotion Intelligence":
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("promotion", os.path.join(ROOT, "_pages", "promotion.py"))
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.show()

elif page == "📊 Platform Analytics":
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("analytics", os.path.join(ROOT, "_pages", "analytics.py"))
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.show()
