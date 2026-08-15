"""
streamlit_app/app.py
Basics4AI — Authentication Portal
Login → Student / Teacher / Admin dashboards

Changes v3:
  - Overview diagram shown on every page (top-center)
  - Super Admin gate: registrations are 'pending' until skde approves them
  - Pending users see a friendly holding message instead of the dashboard

Changes v4 (security pre-freeze):
  - Login lockout: 3 failures in 5 min → 5-min timed lockout (DB-backed)
  - Auto-backup scheduler started at module level (survives Streamlit reruns)
"""

import os
import sys
from pathlib import Path

# 0. FORCE UTF-8 STDOUT/STDERR
# Windows consoles default to a legacy codepage (e.g. cp1252) that cannot
# encode the emoji used in status print()s throughout this app (e.g.
# "[ModuleRegistry] ✅ Registered module: ..."). Without this, any local
# run outside Docker/Streamlit Cloud (which default to UTF-8) crashes with
# UnicodeEncodeError the moment such a line is printed. Must run before
# any other module in this app writes to stdout/stderr.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 1. SET ENVIRONMENT VARIABLES (must be before rpy2 is imported)
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.5.2'
os.environ['RPY2_CFFI_MODE'] = 'ABI'

# Load .env file into os.environ (python-dotenv — safe no-op if file absent)
#try:
#    from dotenv import load_dotenv
#    load_dotenv()
#except ImportError:
#    pass

# above code worked but can't check survey scores table updates, so changed to the following
try:
    from dotenv import load_dotenv
    if os.getenv("RAILWAY_ENVIRONMENT") is None:
        load_dotenv()
except ImportError:
    pass
# -------------END

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

# 3b. Suppress sentence-transformers / HuggingFace loading noise
#import logging as _logging
#_logging.getLogger("transformers").setLevel(_logging.ERROR)
#_logging.getLogger("sentence_transformers").setLevel(_logging.ERROR)
#try:
#    import transformers as _tf
#    _tf.logging.set_verbosity_error()
#except Exception:
#    pass
# 3b. Suppress transformers / Streamlit noise (STRONG suppression)
import logging as _logging
_logging.getLogger("transformers").setLevel(_logging.CRITICAL)
_logging.getLogger("sentence_transformers").setLevel(_logging.CRITICAL)
_logging.getLogger("streamlit").setLevel(_logging.ERROR)
try:
    import transformers as _tf
    _tf.logging.set_verbosity_error()
except Exception:
    pass

# 4. STREAMLIT & OTHER IMPORTS
import streamlit as st

# Message whether R is loaded or not?
#st.write("🚨 TOP OF SCRIPT EXECUTED")
#
#if r is None:
#    st.write("🚨 R = None")
#else:
#    st.write("🚨 R LOADED")
# End of message

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
    get_user_status,
    is_super_admin,
    seed_super_admin,
)

# ── NEW: Login security ───────────────────────────────────────────────────────
from auth.login_security import (
    init_login_attempts_table,
    record_attempt,
    is_locked_out,
    get_lockout_remaining_seconds,
)
# ─────────────────────────────────────────────────────────────────────────────

# ----------------------------------------------------------
# AUTO-BACKUP SCHEDULER — module-level boot (runs once per process)
# Module-level globals survive Streamlit reruns within the same process.
# NEVER move this inside a render/dashboard function — it would restart
# the scheduler on every user interaction.
# ----------------------------------------------------------
_AUTO_BACKUP_STARTED = False  # module-level flag — NOT session_state

if not _AUTO_BACKUP_STARTED:
    _AUTO_BACKUP_STARTED = True
    try:
        from core.admin.system_service import start_auto_backup_scheduler
        start_auto_backup_scheduler()
    except Exception:
        pass  # APScheduler absent — manual backups still work

# ----------------------------------------------------------
# INITIALIZATION (ONCE)
# ----------------------------------------------------------
if "modules_registered" not in st.session_state:
    register_all_modules()
    st.session_state["modules_registered"] = True

# Ensure databases exist
init_app_db()
init_auth_db()

# ── NEW: Ensure login_attempts table exists ───────────────────────────────────
init_login_attempts_table()
# ─────────────────────────────────────────────────────────────────────────────

# Seed super admin on first boot (idempotent — safe to call every run)
seed_super_admin()

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
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th { text-align: left !important; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# HELPER: resolve asset paths
# ----------------------------------------------------------
def _asset(filename: str) -> Path | None:
    candidates = [
        Path(__file__).resolve().parent / "assets" / "images" / filename,
        Path(__file__).resolve().parents[1] / "streamlit_app" / "assets" / "images" / filename,
    ]
    return next((p for p in candidates if p.exists()), None)


# ----------------------------------------------------------
# HEADER: logo + portal title (shown on every page)
# ----------------------------------------------------------
def _render_header():
    logo_path = _asset("B4AI_Logo_trimmed.png")
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if logo_path:
            st.image(str(logo_path), width=90)
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

    diagram_path = _asset("Integrated_Dashboard_for_Data_infographic_2K.jpeg")
    if diagram_path:
        _, col_mid, _ = st.columns([0.5, 9, 0.5])
        with col_mid:
            st.image(
                str(diagram_path),
                caption="Basics4ai — Qual+Quant Hybrid Analysis Platform",
                width="stretch",
            )
        st.divider()


# ----------------------------------------------------------
# REGISTRATION PAGE  (unchanged)
# ----------------------------------------------------------
def show_registration():
    st.markdown(
        "<h2 style='font-size:1.6rem;font-weight:700;'>Register</h2>",
        unsafe_allow_html=True,
    )
    st.info(
        "⏳ After registration your account will be **pending approval** by the "
        "system administrator. You will be able to log in once approved.",
        icon="🔐",
    )

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm  = st.text_input("Confirm Password", type="password")
    role     = st.selectbox("Role", ["student", "teacher", "admin"])

    cohort_id = None
    current_user_role = (
        st.session_state.get("username")
        and get_user_role(st.session_state.get("username"))
    )
    cohort_list = user_service.get_all_cohorts()
    if current_user_role in ("admin", "teacher", "super_admin") or current_user_role is None:
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
            register_user(
                username.strip(),
                password,
                role=role,
                cohort_id=cohort_id,
                status="pending",
            )
            # Bug fixed 2026-08-09: st.success() here never reliably reached
            # the screen -- the st.rerun() below fires before the browser
            # paints it, AND the sidebar Navigation radio (main(), no
            # explicit key) recomputed mode="register" from its own
            # persisted widget state on the very next run anyway, silently
            # overwriting mode="login" set here. Net effect: students landed
            # back on a blank-looking registration page with no visible
            # confirmation and assumed it failed -- the likely actual cause
            # of the impatient-duplicate-account problem. Fixed with a
            # one-shot flash flag (mirroring _locked_out_username) shown on
            # the login page after the redirect, plus bumping _nav_gen so
            # the radio widget is freshly created and actually honors the
            # "Login" default instead of clinging to its stale state.
            st.session_state["_just_registered_username"] = username.strip()
            st.session_state["_nav_gen"] = st.session_state.get("_nav_gen", 0) + 1
            st.session_state.mode = "login"
            st.rerun()
        except Exception as e:
            _emsg = str(e)
            if "already exists" in _emsg.lower() or "unique" in _emsg.lower():
                st.error(
                    f"❌ **Username already taken.** "
                    f"'{username.strip()}' is registered. "
                    "Please choose a different username."
                )
            else:
                st.error(_emsg)


# ----------------------------------------------------------
# LOCKOUT COUNTDOWN  (auto-refreshing fragment)
# ----------------------------------------------------------
# A plain st.error() computed once at click time never updates on its
# own, so participants (especially younger ones) see a frozen timer
# and think it's broken. @st.fragment(run_every=1) re-executes just
# this block every second, independent of the rest of the page, so the
# countdown actually ticks down in real time without any further clicks.
@st.fragment(run_every=1)
def _lockout_countdown_fragment(username: str):
    if not is_locked_out(username):
        st.success("✅ You can try logging in again now.")
        st.session_state.pop("_locked_out_username", None)
        return
    remaining = get_lockout_remaining_seconds(username)
    minutes   = remaining // 60
    seconds   = remaining % 60
    st.error(
        f"🔒 Account temporarily locked after too many failed attempts. "
        f"Please try again in **{minutes}m {seconds}s**."
    )


# ----------------------------------------------------------
# LOGIN PAGE  ← MODIFIED for lockout
# ----------------------------------------------------------
def show_login():
    st.markdown(
        "<h2 style='font-size:1.6rem;font-weight:700;'>Login</h2>",
        unsafe_allow_html=True,
    )

    # One-shot confirmation after a successful registration (see the fix
    # note in show_registration()) -- shown once here, then popped, so a
    # plain page reload doesn't bring it back.
    _just_registered = st.session_state.pop("_just_registered_username", None)
    if _just_registered:
        st.success(
            f"✅ Registration successful for **'{_just_registered}'**! "
            "Please wait for **Admin approval** before logging in — "
            "you don't need to register again.",
            icon="🎉",
        )

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # If a previous attempt on this page already found this account
    # locked out, keep showing the live countdown even without another
    # click — this is what makes the timer actually tick in real time.
    _locked_uname = st.session_state.get("_locked_out_username")
    if _locked_uname and is_locked_out(_locked_uname):
        _lockout_countdown_fragment(_locked_uname)

    if st.button("Login"):
        uname = username.strip()

        # ── Step 1: Check lockout BEFORE attempting authentication ────────────
        # Lockout is DB-backed — survives page refreshes and multiple tabs.
        if is_locked_out(uname):
            st.session_state["_locked_out_username"] = uname
            st.rerun()
            return
        # ─────────────────────────────────────────────────────────────────────

        if authenticate_user(uname, password):

            # ── Step 2a: Record successful attempt ───────────────────────────
            record_attempt(uname, success=True)
            # ─────────────────────────────────────────────────────────────────

            status = get_user_status(uname)
            if status == "pending":
                st.warning(
                    "⏳ Your account is **pending approval** by the administrator. "
                    "You will receive access once it is reviewed."
                )
                return
            if status == "rejected":
                st.error("❌ Your registration was not approved. Please contact the administrator.")
                return
            st.session_state.logged_in = True
            st.session_state.username  = uname
            st.session_state.mode      = "dashboard"
            # Migration 2 — resolve RID at login, cache in session
            try:
                from auth.user_manager import get_user_rid
                st.session_state.rid = get_user_rid(uname)
            except Exception:
                st.session_state.rid = None
            st.rerun()

        else:
            # ── Step 2b: Record failed attempt, then re-check lockout ────────
            record_attempt(uname, success=False)

            if is_locked_out(uname):
                # This attempt just triggered the lockout threshold
                st.session_state["_locked_out_username"] = uname
                st.rerun()
            else:
                st.error("Invalid credentials.")
            # ─────────────────────────────────────────────────────────────────


# ----------------------------------------------------------
# ROLE GUARD  (unchanged)
# ----------------------------------------------------------
def require_role(allowed_roles):
    username = st.session_state.get("username")
    if not username:
        st.error("Not authenticated.")
        st.stop()
    role = get_user_role(username)
    if role == "super_admin":
        return
    if role not in allowed_roles:
        st.error("Unauthorized access.")
        st.stop()


# ----------------------------------------------------------
# MODULE VIEW RENDERER  (unchanged)
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
    meta   = module.get("meta", {})
    title  = meta.get("title", module_id)

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
    st.stop()


# ----------------------------------------------------------
# MAIN CONTROLLER  (unchanged)
# ----------------------------------------------------------
def main():
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("mode", "register")
    st.session_state.setdefault("view", None)

    _render_header()

    if not st.session_state.logged_in:
        # Bug fixed 2026-08-09: this radio had no explicit key=, so once
        # Streamlit created its widget state, index= was only honored on
        # the very first render -- a later programmatic mode="login" (e.g.
        # right after registering) had no effect, since the widget just
        # replayed its own last-clicked value ("Register") and immediately
        # overwrote mode back on the next line. Folding a generation
        # counter into the key (bumped by show_registration() on success)
        # forces a genuinely fresh widget when we need to redirect, the
        # same fix pattern already used for the admin dashboard's cohort
        # selectors this session.
        st.session_state.setdefault("_nav_gen", 0)
        _nav_gen = st.session_state["_nav_gen"]
        _nav_options = ["Register", "Login"]
        _nav_index = _nav_options.index(st.session_state.mode.capitalize()) \
            if st.session_state.mode.capitalize() in _nav_options else 0
        choice = st.sidebar.radio(
            "Navigation", _nav_options, index=_nav_index, key=f"_nav_radio_{_nav_gen}"
        )
        st.session_state.mode = choice.lower()
        if st.session_state.mode == "register":
            show_registration()
        else:
            show_login()
        return

    username = st.session_state.username
    role     = get_user_role(username)

    st.sidebar.success(f"Logged in as **{username}** ({role})")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.sidebar.divider()
    with st.sidebar.expander("📚 Instruments & References", expanded=False):
        st.markdown(
            "**Instruments**\n\n"
            "- SCCCES (adapted) draws sub-constructs from two source scales:\n"
            "  - SCES — Rotgans, J. I., & Schmidt, H. G. (2011). *Advances in "
            "Health Sciences Education, 16*(4), 465–479.\n"
            "  - CCCES — Heddy, B. C., Taasoobshirazi, G., Chancey, J. B., & "
            "Danielson, R. W. (2018). *Frontiers in Education, 3*(43), 1–9.\n"
            "- SIMS — Guay, F., Vallerand, R. J., & Blanchard, C. (2000). "
            "*Motivation and Emotion, 24*(3), 175–213. (Deci & Ryan, 1985 is "
            "the underlying Self-Determination Theory, not the SIMS instrument itself.)\n"
            "- AI-CI — AI Conceptual Inventory\n"
            "- AIM-F — AI Misconceptions Framework\n\n"
            "*Full reference list, including every citation used in the "
            "Correlations, Inferential Statistics, and Basic Statistics "
            "tabs: Report Generation → viii. Instruments & References.*\n\n"
            "**Quantitative Analysis (CPI)**\n\n"
            "Crocker, L., & Algina, J. (1986). *Introduction to classical and "
            "modern test theory*. Holt, Rinehart and Winston. *(CPI_quant / CTT)*\n\n"
            "Baker, F. B. (1985). *The basics of item response theory*. "
            "Heinemann. *(IRT)*\n\n"
            "Hake, R. R. (1998). Interactive-engagement versus traditional methods. "
            "*American Journal of Physics, 66*(1), 64–74. *(Normalised gain)*\n\n"
            "Pellegrino, J. W., & Hilton, M. L. (Eds.). (2012). *Education for "
            "life and work*. National Academies Press. *(CPI+ framework)*\n\n"
            "Dawes, R. M. (1979). The robust beauty of improper linear models. "
            "*Psychological Bulletin, 86*(2), 571–582. *(Equal weighting)*\n\n"
            "Rosli, M. S., Saleh, N. S., Alshammari, S. H., Ibrahim, M. M., "
            "Atan, A. S., & Atan, N. A. (2021). Improving Questionnaire Reliability "
            "using Construct Reliability for Researches in Educational Technology. "
            "*iJIM, 15*(04), 109. "
            "https://doi.org/10.3991/ijim.v15i04.20199 *(Construct Reliability / CR)*\n\n"
            "**Qualitative Analysis**\n\n"
            "Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. "
            "*Qualitative Research in Psychology, 3*(2), 77–101.\n\n"
            "De Paoli, S. (2024). Performing an inductive thematic analysis with a LLM. "
            "*Social Science Computer Review, 42*(4), 997–1019.\n\n"
            "Bingham, A. J. (2023). From data management to actionable findings. "
            "*International Journal of Qualitative Methods, 22*.\n\n"
            "Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and "
            "Chatbot Arena. *arXiv*. https://doi.org/10.48550/ARXIV.2306.05685 "
            "*(CPI_qual / LLM-as-a-Judge)*\n\n"
            "*Disclaimer: LLM-assisted thematic analysis is exploratory and does "
            "not establish formal procedures for I-/D-TA with LLMs.*"
        )

    if role == "student" and st.session_state.get("view"):
        require_role(["student"])
        render_active_module(username)

    if role in ("admin", "super_admin"):
        require_role(["admin", "super_admin"])
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
