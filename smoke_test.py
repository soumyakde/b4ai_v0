"""
smoke_test.py — b4ai_v0 environment verification
Run from project root: python smoke_test.py

Checks T1.P, T2.P, T3.P before manual testing.
Exit code 0 = all passed. Exit code 1 = failures found.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
failures = []

def check(label, fn):
    try:
        result = fn()
        print(f"{PASS}  {label}: {result}")
    except Exception as e:
        print(f"{FAIL}  {label}: {e}")
        failures.append(label)

print("\n=== b4ai_v0 smoke test ===\n")

# T1.P — Auth DB
def t1_auth():
    import sqlite3
    path = os.getenv("USERS_DB_PATH", "users.db")
    conn = sqlite3.connect(path)
    n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    assert n > 0, f"users table empty at {path}"
    return f"{n} users found at {path}"

check("T1.P  auth DB reachable", t1_auth)

# T2.P — responses & completions tables
def t2_responses():
    from core.db_utils import get_connection
    conn = get_connection()
    r = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    c = conn.execute("SELECT COUNT(*) FROM completions").fetchone()[0]
    conn.close()
    return f"responses={r} rows, completions={c} rows"

check("T2.P  responses DB reachable", t2_responses)

# T3.P — canonical_loader returns data (empty DB is valid on clean slate)
def t3_canonical():
    from core.analytics.datasets.canonical_loader import load_canonical_data
    canonical_df, demographics_df, cohort_map = load_canonical_data()
    assert canonical_df is not None, "canonical_df is None"
    # cohort_map may be empty on clean slate — check users.db is reachable instead
    from auth.user_manager import get_user_cohort_map
    cohort_map = get_user_cohort_map()
    n_users = len(cohort_map)
    return f"canonical_df shape={canonical_df.shape}, cohort_map={n_users} users (0 = clean slate ok)"

check("T3.P  canonical_loader works", t3_canonical)

# research.db reachable
def t_research():
    import sqlite3
    path = os.getenv("RESEARCH_DB_PATH", "research.db")
    conn = sqlite3.connect(path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    conn.close()
    assert "llm_result_cache" in tables, "llm_result_cache table missing"
    return f"research.db ok at {path}"

check("T_RES  research DB reachable", t_research)

def t_wal():
    from core.db_utils import get_connection
    conn = get_connection()
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    conn.close()
    assert mode == "wal", f"Expected WAL mode, got: {mode}"
    assert busy >= 5000, f"busy_timeout too low: {busy}ms"
    return f"journal_mode={mode}, busy_timeout={busy}ms"

check("T_WAL  WAL mode active", t_wal)

# Summary
print(f"\n{'='*30}")
if failures:
    print(f"\033[91m{len(failures)} check(s) failed:\033[0m {', '.join(failures)}")
    print("Fix these before proceeding to manual tests.\n")
    sys.exit(1)
else:
    print("\033[92mAll checks passed — proceed to manual tests.\033[0m\n")
    sys.exit(0)
