"""
Phase 1 — Step 1b Integration Test
test_descriptive_stats_integration.py

Runs descriptive_stats against real responses.db + users.db.

Usage:
    python tests/phase1/test_descriptive_stats_integration.py
    python tests/phase1/test_descriptive_stats_integration.py path/to/responses.db path/to/users.db
"""
import sys, os, importlib.util, sqlite3
import pandas as pd

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DB_RESPONSES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "responses.db")
DB_USERS     = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_ROOT, "users.db")

for path, name in [(DB_RESPONSES,"responses.db"),(DB_USERS,"users.db")]:
    if not os.path.exists(path):
        print(f"ERROR: {name} not found at {path}"); sys.exit(1)

print(f"\n  responses.db : {DB_RESPONSES}")
print(f"  users.db     : {DB_USERS}")

# -----------------------------------------------------------------------
# Import modules
# -----------------------------------------------------------------------
def _load(candidates, name):
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location(name, p)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    print(f"ERROR: {name} not found"); sys.exit(1)

demo_mod = _load([
    os.path.join(_ROOT,"core","analytics","filters","demographics_extractor.py"),
    os.path.join(os.path.dirname(__file__),"demographics_extractor.py"),
], "demographics_extractor")

stats_mod = _load([
    os.path.join(_ROOT,"core","analytics","descriptive","descriptive_stats.py"),
    os.path.join(os.path.dirname(__file__),"descriptive_stats.py"),
], "descriptive_stats")

extract_demographics = demo_mod.extract_demographics
count_participants   = stats_mod.count_participants
count_by_gender      = stats_mod.count_by_gender
count_by_grade       = stats_mod.count_by_grade
count_by_language    = stats_mod.count_by_language
count_by_cohort      = stats_mod.count_by_cohort
participant_summary  = stats_mod.participant_summary

# -----------------------------------------------------------------------
# Load real data
# -----------------------------------------------------------------------
print("\n[Step 1] Loading real demographics and cohort data...")
demographics_df = extract_demographics(DB_RESPONSES)
print(f"  → {len(demographics_df)} users with demographics")

conn = sqlite3.connect(DB_USERS)
rows = conn.execute("SELECT username, cohort_id FROM users").fetchall()
conn.close()
cohort_map = {username: cohort_id for username, cohort_id in rows}
print(f"  → {len(cohort_map)} users in cohort_map")

# -----------------------------------------------------------------------
# Test runner
# -----------------------------------------------------------------------
PASS = FAIL = 0
def check(label, got, expected):
    global PASS, FAIL
    ok = (got == expected)
    if ok: print(f"  ✅ PASS  {label}"); PASS += 1
    else:
        print(f"  ❌ FAIL  {label}")
        print(f"           got: {got!r}  expected: {expected!r}"); FAIL += 1

def check_true(label, cond, detail=""):
    global PASS, FAIL
    if cond: print(f"  ✅ PASS  {label}"); PASS += 1
    else: print(f"  ❌ FAIL  {label}  {detail}"); FAIL += 1

# -----------------------------------------------------------------------
# Structural checks
# -----------------------------------------------------------------------
print("\n[Structural Checks]")

total = count_participants(demographics_df)
check("total participants = 9", total, 9)

g = count_by_gender(demographics_df)
check_true("gender n sums to 9",      g["n"].sum() == 9)
check_true("gender pcts sum ~100",    abs(g["pct"].sum() - 100.0) < 0.1)
check_true("no unexpected gender values",
    set(g["gender"]).issubset({"Male","Female","Unknown"}))

gr = count_by_grade(demographics_df)
check_true("grade n sums to 9",       gr["n"].sum() == 9)
check_true("grade pcts sum ~100",     abs(gr["pct"].sum() - 100.0) < 0.1)
check_true("all grade_levels valid",
    gr["grade_level"].apply(lambda x: x in (4,5,6,7,8,"Adult","Unknown")).all())

lang = count_by_language(demographics_df)
check_true("language n sums to 9",    lang["n"].sum() == 9)
check_true("language pcts sum ~100",  abs(lang["pct"].sum() - 100.0) < 0.1)
check_true("English row present",     "English" in lang["label"].values)

coh = count_by_cohort(demographics_df, cohort_map)
check_true("cohort n sums to 9",      coh["n"].sum() == 9)
check_true("cohort pcts sum ~100",    abs(coh["pct"].sum() - 100.0) < 0.1)

# -----------------------------------------------------------------------
# Print full summary tables for manual verification
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("PARTICIPANT SUMMARY — verify against your records")
print("="*60)
print(f"\n  Total participants: {total}")

print("\n  By Gender:")
print(g.to_string(index=False))

print("\n  By Grade:")
print(gr.to_string(index=False))

print("\n  By Language:")
print(lang.to_string(index=False))

print("\n  By Cohort:")
print(coh.to_string(index=False))

# -----------------------------------------------------------------------
# Known-user spot checks (ant_ub confirmed from DB)
# -----------------------------------------------------------------------
print("\n[Known-User Checks]")
if "ant_ub" in demographics_df["user_id"].values:
    row = demographics_df[demographics_df["user_id"]=="ant_ub"].iloc[0]
    check("ant_ub in Female count",
        row["gender"] == "Female", True)
    check("ant_ub in grade 4 count",
        row["grade_level"] == 4, True)
    check("ant_ub in English count",
        row["first_language_english"] == True, True)

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL STRUCTURAL CHECKS PASSED.")
    print("  👁  Please verify the printed tables match your records.")
else:
    print("  ❌ SOME CHECKS FAILED — review output above.")
print('='*60)
