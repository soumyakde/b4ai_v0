"""
launch.py  —  B4AI entry-point
==============================
Initialises R on the main thread BEFORE handing control to Streamlit.
This prevents rpy2 from stealing SIGINT (Ctrl-C).

Usage:
    python launch.py
"""
import os
import sys
from pathlib import Path

# ── 1. R environment ──────────────────────────────────────────────────────────
R_HOME = r"C:\Program Files\R\R-4.5.2"
os.environ["R_HOME"]         = R_HOME
os.environ["RPY2_CFFI_MODE"] = "ABI"
os.environ["PATH"]           = os.path.join(R_HOME, "bin", "x64") + \
                                os.pathsep + os.environ.get("PATH", "")

# ── 2. Project root on sys.path ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 3. Initialise R on the main thread ───────────────────────────────────────
#    Must happen here, before streamlit.web.cli.main() launches worker threads.
import threading
assert threading.current_thread() is threading.main_thread(), \
    "launch.py must be run directly, not imported."

try:
    import rpy2.robjects  # noqa: F401 — side-effect: initialises R
    print("[launch] R initialised on main thread — Ctrl-C will work.")
except Exception as e:
    print(f"[launch] rpy2/R not available ({e}) — IRT features disabled.")

# ── 4. Restore SIGINT to Python BEFORE handing off to Streamlit ──────────────
import signal
signal.signal(signal.SIGINT, signal.default_int_handler)

# ── 5. Launch Streamlit ───────────────────────────────────────────────────────
from streamlit.web import cli as _st_cli
sys.argv = [
    "streamlit", "run",
    str(ROOT / "streamlit_app" / "app.py"),
    "--server.headless", "false",
]
_st_cli.main()
