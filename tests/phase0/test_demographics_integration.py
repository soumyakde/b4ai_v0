"""
Phase 0 — Step 6 Integration Test
test_demographics_integration.py

Runs extract_demographics() against the REAL responses.db.

Unlike the unit tests (which use synthetic data), this test:
- Reads your actual database
- Prints every row returned so you can verify correctness
- Checks structural invariants only (no hardcoded expected values)
  EXCEPT for the one real user we know: ant_ub

Usage:
    # From project root (DB at responses.db):
    python tests/phase0/test_demographics_integration.py

    # Or specify DB path explicitly:
    python tests/phase0/test_demographics_integration.py path/to/responses.db

Expected: structural checks pass + printed table matches your records.
"""

import sys
import os
import importlib.util

import pandas as pd

# -----------------------------------------------------------------------
# Resolve DB path
# -----------------------------------------------------------------------
if len(sys.argv) > 1:
    DB_PATH = sys.argv[1]
else:
    # Default: responses.db at project root
    DB_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "responses.db"
    )

DB_PATH = os.path.normpath(DB_PATH)

if not os.path.exists(DB_PATH):
    print(f"ERROR: responses.db not found at: {DB_PATH}")
    print("Usage: python tests/phase0/test_demographics_integration.py [path/to/responses.db]")
    sys.exit(1)

print(f"\n  DB path : {DB_PATH}")

# -----------------------------------------------------------------------
# Import extractor
# -----------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_THIS_DIR, "..", "..", "core", "analytics", "filters", "demographics_extractor.py"),
    os.path.join(_THIS_DIR, "demographics_extractor.py"),
]
_path = None
for _p in _CANDIDATES:
    if os.path.exists(os.path.normpath(_p)):
        _path = os.path.normpath(_p)
        break

if _path is None:
    print("ERROR: demographics_extractor.py not found.")
    sys.exit(1)

_spec = importlib.util.spec_from_file_location("demographics_extractor", _path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
extract_demographics = _mod.extract_demographics

# -----------------------------------------------------------------------
# Test runner
# -----------------------------------------------------------------------
PASS = 0
FAIL = 0

def check(label, got, expected):
    global PASS, FAIL
    if got == expected:
        print(f"  ✅ PASS  {label}")
        PASS += 1
    else:
        print(f"  ❌ FAIL  {label}")
        print(f"           got:      {got!r}")
        print(f"           expected: {expected!r}")
        FAIL += 1

def check_true(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✅ PASS  {label}")
        PASS += 1
    else:
        print(f"  ❌ FAIL  {label}  {detail}")
        FAIL += 1

# -----------------------------------------------------------------------
# Run extractor
# -----------------------------------------------------------------------
print("\n[Step 1] Running extract_demographics() on real responses.db ...")
try:
    df = extract_demographics(DB_PATH)
    print(f"  → Returned {len(df)} rows\n")
except Exception as e:
    print(f"  ❌ FATAL: extractor raised {type(e).__name__}: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Print full results table for manual verification
# -----------------------------------------------------------------------
print("=" * 75)
print("FULL DEMOGRAPHICS TABLE (verify each row against your records)")
print("=" * 75)
if df.empty:
    print("  (no rows returned — check that precourse_demographics_survey rows exist)")
else:
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    pd.set_option("display.colheader_justify", "left")
    print(df.to_string(index=False))
print("=" * 75)

# -----------------------------------------------------------------------
# Structural invariants — no hardcoded expected values
# -----------------------------------------------------------------------
print("\n[Structural Checks]")

check("correct columns present",
      list(df.columns),
      ["user_id", "grade", "grade_level", "gender", "first_language_english"])

check_true("no duplicate user_ids",
           df["user_id"].is_unique,
           f"duplicates: {df[df.duplicated('user_id')]['user_id'].tolist()}")

check_true("user_ids are sorted alphabetically",
           list(df["user_id"]) == sorted(df["user_id"].tolist()),
           "not sorted")

check_true("grade_level values are int 4-8 or 'Adult' or None",
           df["grade_level"].dropna().apply(
               lambda x: x in (4, 5, 6, 7, 8, "Adult")
           ).all(),
           f"unexpected: {df['grade_level'].dropna().unique().tolist()}")

check_true("gender values are Male / Female / None only",
           df["gender"].dropna().isin(["Male", "Female"]).all(),
           f"unexpected: {df['gender'].dropna().unique().tolist()}")

check_true("first_language_english is bool or None only",
           df["first_language_english"].dropna().apply(
               lambda x: isinstance(x, bool)
           ).all(),
           f"unexpected: {df['first_language_english'].dropna().unique().tolist()}")

# -----------------------------------------------------------------------
# Known-user check: ant_ub
# Confirmed from your DB sample:
#   Q2_2 = "Fourth (4th) grade"
#   Q2_3 = "True"
#   Q2_4 = "Female"
# -----------------------------------------------------------------------
print("\n[Known-User Check: ant_ub]")

if "ant_ub" in df["user_id"].values:
    row = df[df["user_id"] == "ant_ub"].iloc[0]
    check("ant_ub grade",                   row["grade"],                  "Fourth (4th) grade")
    check("ant_ub grade_level",             row["grade_level"],             4)
    check("ant_ub gender",                  row["gender"],                  "Female")
    check("ant_ub first_language_english",  row["first_language_english"],  True)
else:
    print("  ⚠️  SKIP  ant_ub not found in DB (may have been deleted)")

# -----------------------------------------------------------------------
# Coverage summary
# -----------------------------------------------------------------------
print("\n[Coverage Summary — for your review]")
print(f"  Total users with demographics : {len(df)}")

grade_counts = df["grade_level"].value_counts(dropna=False).sort_index()
print(f"\n  Grade distribution:")
for grade, count in grade_counts.items():
    print(f"    grade_level={grade!r:>8}  →  {count} student(s)")

gender_counts = df["gender"].value_counts(dropna=False)
print(f"\n  Gender distribution:")
for gender, count in gender_counts.items():
    print(f"    gender={gender!r:>10}  →  {count} student(s)")

lang_counts = df["first_language_english"].value_counts(dropna=False)
print(f"\n  First language English:")
for lang, count in lang_counts.items():
    print(f"    first_language_english={lang!r}  →  {count} student(s)")

missing_grade   = df["grade"].isna().sum()
missing_gender  = df["gender"].isna().sum()
missing_lang    = df["first_language_english"].isna().sum()
if any([missing_grade, missing_gender, missing_lang]):
    print(f"\n  ⚠️  Missing values:")
    if missing_grade:  print(f"    grade missing for {missing_grade} user(s)")
    if missing_gender: print(f"    gender missing for {missing_gender} user(s)")
    if missing_lang:   print(f"    first_language_english missing for {missing_lang} user(s)")
else:
    print("\n  ✅ No missing values across grade, gender, first_language_english")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Structural checks: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL STRUCTURAL CHECKS PASSED.")
    print("  👁  Please verify the printed table above matches your records.")
else:
    print("  ❌ SOME CHECKS FAILED — review output above.")
print('='*60)
