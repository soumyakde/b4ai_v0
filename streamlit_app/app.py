"""
streamlit_app/app.py
Basics4AI — Minimal Authentication Portal
Login → Student / Teacher / Admin dashboards

Derived-state architecture.
No JSON.
No stored completion flags.
All module resolution via registry.
"""

import os
import sys
from pathlib import Path

# 1. SET ENVIRONMENT VARIABLES (Must be before rpy2 is imported)
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.5.2'
os.environ['RPY2_CFFI_MODE'] = 'ABI'

# 2. PATH SETUP
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# 3. IMPORT R (optional — not available on Streamlit Cloud)
try:
    from rpy2.robjects import r
    from core.analytics.r_utils import to_r, to_pd
except Exception:
    r = None
    to_r = to_pd = None

# 4. STREAMLIT & OTHER IMPORTS
import streamlit as st

from modules.registry.register_modules import register_all_modules
from modules.registry.module_registry import module_registry
from core.admin import user_service
from core.db_utils import init_db as init_app_db
from dashboards.student_dashboard import show_student_dashboard
from dashboards.teacher_dashboard import show_teacher_dashboard
from dashboards.admin_dashboard import show_admin_dashboard

from auth.user_manager import (
    init_db as init_auth_db,
    register_user,
    authenticate_user,
    get_user_role,
)

# ----------------------------------------------------------
# INITIALIZATION (ONCE)
# ----------------------------------------------------------
if "modules_registered" not in st.session_state:
    register_all_modules()
    st.session_state["modules_registered"] = True

# Ensure databases exist
init_app_db()
init_auth_db()

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Basics4AI Portal",
    page_icon="🤖",
    layout="centered",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Left-align all dataframe cells */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th { text-align: left !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# REGISTRATION PAGE
# ----------------------------------------------------------
def show_registration():
    st.markdown(
        "<h2 style='font-size:1.6rem;font-weight:700;'>Register</h2>",
        unsafe_allow_html=True
    )

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    role = st.selectbox("Role", ["student", "teacher", "admin"])

    cohort_id = None
    current_user_role = (
        st.session_state.get("username") and get_user_role(st.session_state.get("username"))
    )

    cohort_list = user_service.get_all_cohorts()
    if current_user_role in ("admin", "teacher") or current_user_role is None:
        selected_cohort = st.selectbox("Assign Cohort (optional)", ["None"] + cohort_list)
        cohort_id = None if selected_cohort == "None" else selected_cohort

    if st.button("Register"):
        if not username or not password:
            st.warning("Enter username and password.")
            return
        if password != confirm:
            st.error("Passwords do not match.")
            return

        try:
            register_user(username.strip(), password, role=role, cohort_id=cohort_id)
            st.success("Registration successful. Please login.")
            st.session_state.mode = "login"
            st.rerun()
        except Exception as e:
            st.error(str(e))


# ----------------------------------------------------------
# LOGIN PAGE
# ----------------------------------------------------------
def show_login():
    st.markdown(
        "<h2 style='font-size:1.6rem;font-weight:700;'>Login</h2>",
        unsafe_allow_html=True
    )

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate_user(username.strip(), password):
            st.session_state.logged_in = True
            st.session_state.username = username.strip()
            st.session_state.mode = "dashboard"
            st.rerun()
        else:
            st.error("Invalid credentials.")


# ----------------------------------------------------------
# ROLE GUARD
# ----------------------------------------------------------
def require_role(allowed_roles):
    username = st.session_state.get("username")
    if not username:
        st.error("Not authenticated.")
        st.stop()

    role = get_user_role(username)
    if role not in allowed_roles:
        st.error("Unauthorized access.")
        st.stop()


# ----------------------------------------------------------
# MODULE VIEW RENDERER
# ----------------------------------------------------------
def render_active_module(username: str):
    module_id = st.session_state.get("view")
    if not module_id:
        return

    if not module_registry.has(module_id):
        st.error(f"Module not found: {module_id}")
        st.session_state.view = None
        st.rerun()

    module = module_registry.get(module_id)
    meta = module.get("meta", {})
    title = meta.get("title", module_id)

    if st.button("⬅ Back to Dashboard"):
        st.session_state.view = None
        st.rerun()

    st.title(f"📘 {title}")
    st.divider()

    render_fn = module.get("render")
    if not callable(render_fn):
        st.error(f"Module '{module_id}' does not define a callable render(username) function.")
        st.stop()

    render_fn(username)
    st.stop()  # Prevent dashboard bleed-through


# ----------------------------------------------------------
# MAIN CONTROLLER
# ----------------------------------------------------------
def main():
    # Session defaults
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("mode", "register")
    st.session_state.setdefault("view", None)

    # ── Logo + Portal title (top of every page) ──────────────────────────────
    from pathlib import Path as _Path
    import os as _os
    _logo_candidates = [
        _Path(__file__).resolve().parent / "assets" / "images" / "B4AI_Logo_trimmed.png",
        _Path(__file__).resolve().parents[1] / "streamlit_app" / "assets" / "images" / "B4AI_Logo_trimmed.png",
    ]
    _logo_path = next((p for p in _logo_candidates if p.exists()), None)
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if _logo_path:
            st.image(str(_logo_path), width=90)
        else:
            st.markdown("🤖")
    with col_title:
        st.markdown(
            "<h1 style='margin-top:0.2rem; font-size:2.2rem; font-weight:800;"
            " color:#0077BB; letter-spacing:-0.5px;'>Basics4AI Portal</h1>"
            "<p style='margin:0; color:#555; font-size:0.95rem;'>"
            "AI Literacy for Young Learners</p>",
            unsafe_allow_html=True,
        )
    st.divider()

    # ---------------------- NOT LOGGED IN ----------------------
    if not st.session_state.logged_in:
        choice = st.sidebar.radio("Navigation", ["Register", "Login"])
        st.session_state.mode = choice.lower()
        if st.session_state.mode == "register":
            show_registration()
        else:
            show_login()
        return

    # ---------------------- LOGGED IN ----------------------
    username = st.session_state.username
    role = get_user_role(username)

    st.sidebar.success(f"Logged in as {username} ({role})")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    # ── References & Disclaimer ──────────────────────────────────────────────
    st.sidebar.divider()
    with st.sidebar.expander("📚 Instruments & References", expanded=False):
        st.markdown(
            "**Instruments**\n\n"
            "- CCCES — Conceptual Change Cognitive Engagement Scale\n"
            "- SCES — Situational Cognitive Engagement Scale\n"
            "- SIMS — Situational Motivation Scale\n"
            "- AI-CI — AI Conceptual Inventory\n"
            "- AIM-F — AI Misconceptions Framework\n\n"
            "**Qualitative Analysis**\n\n"
            "Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. "
            "*Qualitative Research in Psychology, 3*(2), 77–101. "
            "https://doi.org/10.1191/1478088706qp063oa\n\n"
            "De Paoli, S. (2024). Performing an inductive thematic analysis of "
            "semi-structured interviews with a large language model. "
            "*Social Science Computer Review, 42*(4), 997–1019. "
            "https://doi.org/10.1177/08944393231220483\n\n"
            "Bingham, A. J. (2023). From data management to actionable findings. "
            "*International Journal of Qualitative Methods, 22*, 16094069231183620. "
            "https://doi.org/10.1177/16094069231183620\n\n"
            "*Disclaimer: LLM-assisted thematic analysis is exploratory and does "
            "not establish formal procedures for I-/D-TA with LLMs.*"
        )

    # ---------------------- STUDENT MODULE VIEW ----------------------
    if role == "student" and st.session_state.get("view"):
        require_role(["student"])
        render_active_module(username)

    # ---------------------- DASHBOARD ROUTING ----------------------
    if role == "admin":
        require_role(["admin"])
        show_admin_dashboard(username)
    elif role == "teacher":
        require_role(["teacher"])
        show_teacher_dashboard(username)
    else:
        require_role(["student"])
        show_student_dashboard(username)


# ----------------------------------------------------------
# BOOT
# ----------------------------------------------------------
if __name__ == "__main__":
    main()