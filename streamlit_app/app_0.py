"""
streamlit_app/app.py
Basics4AI — Minimal Authentication Portal
Login → Student / Teacher / Admin dashboards

Derived-state architecture.
No JSON.
No stored completion flags.
All module resolution via registry.
"""

import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

# ----------------------------------------------------------
# PATH SETUP (MUST COME BEFORE ANY PROJECT IMPORTS)
# ----------------------------------------------------------
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# ----------------------------------------------------------
# SAFE PROJECT IMPORTS
# ----------------------------------------------------------
import streamlit as st

from modules.registry.register_modules import register_all_modules
from modules.registry.module_registry import module_registry

from core.db_utils import init_db as init_app_db

from auth.user_manager import (
    init_db as init_auth_db,
    register_user,
    authenticate_user,
    get_user_role,
)

from dashboards.student_dashboard import show_student_dashboard
from dashboards.teacher_dashboard import show_teacher_dashboard
from dashboards.admin_dashboard import show_admin_dashboard


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
st.set_page_config(page_title="Basics4AI Portal", layout="centered")


# ----------------------------------------------------------
# REGISTRATION PAGE
# ----------------------------------------------------------
def show_registration():
    st.title("Register")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    role = st.selectbox("Role", ["student", "teacher", "admin"])

    if st.button("Register"):

        if not username or not password:
            st.warning("Enter username and password.")
            return

        if password != confirm:
            st.error("Passwords do not match.")
            return

        try:
            register_user(username.strip(), password, role=role)
            st.success("Registration successful. Please login.")
            st.session_state.mode = "login"

        except Exception as e:
            st.error(str(e))


# ----------------------------------------------------------
# LOGIN PAGE
# ----------------------------------------------------------
def show_login():
    st.title("Login")

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
# ✅ MODULE VIEW RENDERER (ARCHITECTURALLY CORRECT)
# ----------------------------------------------------------
def render_active_module(username: str):
    """
    Dynamically render selected module.

    - Ensures module exists
    - Adds consistent navigation
    - Prevents dashboard bleed-through
    """

    module_id = st.session_state.get("view")

    if not module_id:
        return

    # Ensure module exists in registry
    if not module_registry.has(module_id):
        st.error(f"Module not found: {module_id}")
        st.session_state.view = None
        st.rerun()

    module = module_registry.get(module_id)
    meta = module.get("meta", {})
    title = meta.get("title", module_id)

    # ✅ Back button ALWAYS at top
    if st.button("⬅ Back to Dashboard"):
        st.session_state.view = None
        st.rerun()

    st.title(f"📘 {title}")
    st.divider()

    render_fn = module.get("render")

    if not callable(render_fn):
        st.error(
            f"Module '{module_id}' does not define a callable render(username) function."
        )
        st.stop()

    # Render module content
    render_fn(username)

    # ✅ Stop dashboard from rendering underneath
    st.stop()


# ----------------------------------------------------------
# MAIN CONTROLLER
# ----------------------------------------------------------
def main():

    # Session defaults
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("mode", "register")
    st.session_state.setdefault("view", None)

    # ----------------------------------------------------------
    # NOT LOGGED IN
    # ----------------------------------------------------------
    if not st.session_state.logged_in:

        choice = st.sidebar.radio("Navigation", ["Register", "Login"])
        st.session_state.mode = choice.lower()

        if st.session_state.mode == "register":
            show_registration()
        else:
            show_login()

        return

    # ----------------------------------------------------------
    # LOGGED IN
    # ----------------------------------------------------------
    username = st.session_state.username
    role = get_user_role(username)

    st.sidebar.success(f"Logged in as {username} ({role})")

    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    # ----------------------------------------------------------
    # ✅ STUDENT MODULE VIEW
    # ----------------------------------------------------------
    if role == "student" and st.session_state.get("view"):
        require_role(["student"])
        render_active_module(username)

    # ----------------------------------------------------------
    # DASHBOARD ROUTING
    # ----------------------------------------------------------
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