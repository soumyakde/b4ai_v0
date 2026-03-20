"""
Phase 0 — Step 6 Verification
test_demographics_extractor.py

Tests: core/analytics/filters/demographics_extractor.py

Covers:
  - All grade options (4th through 8th + Adult)
  - first_language_english normalization (True/False, case variants)
  - gender normalization
  - Missing questions handled gracefully (None)
  - Duplicate responses: first-attempt-only rule
  - Users with no demographics rows excluded
  - Empty DB result returns correct empty schema
  - In-memory variant (extract_demographics_from_df)
  - Output schema and dtypes
  - Deterministic sort

Run from project root:
    python tests/phase0/test_demographics_extractor.py

Expected result: ALL TESTS PASSED
"""

import sys
import os
import sqlite3
import tempfile
import importlib.util

import pandas as pd

# -----------------------------------------------------------------------
# Import extractor by file path
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
    print("Expected at: core/analytics/filters/demographics_extractor.py")
    sys.exit(1)

_spec = importlib.util.spec_from_file_location("demographics_extractor", _path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

extract_demographics         = _mod.extract_demographics
extract_demographics_from_df = _mod.extract_demographics_from_df

# -----------------------------------------------------------------------
# Helpers
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

def make_db(rows: list) -> str:
    """
    Create a temporary SQLite DB with a responses table,
    insert provided rows, return path.
    rows: list of (user_id, instrument_name, question_id, response_value)
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)
    conn.execute("""
        CREATE TABLE responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            instrument_name TEXT NOT NULL,
            question_id TEXT NOT NULL,
            response_value TEXT,
            submitted_at TEXT NOT NULL
        )
    """)
    conn.executemany(
        "INSERT INTO responses (user_id, instrument_name, question_id, response_value, submitted_at) VALUES (?,?,?,?,?)",
        [(r[0], r[1], r[2], r[3], "2026-01-01T00:00:00") for r in rows]
    )
    conn.commit()
    conn.close()
    return tmp.name

INSTR = "precourse_demographics_survey"

# -----------------------------------------------------------------------
# Test Group 1 — Grade normalization: all six YAML options
# -----------------------------------------------------------------------
print("\n[Test Group 1] Grade normalization — all YAML options")

grade_cases = [
    ("Fourth (4th) grade",  "student_g4", 4),
    ("Fifth (5th) grade",   "student_g5", 5),
    ("Sixth (6th) grade",   "student_g6", 6),
    ("Seventh (7th) grade", "student_g7", 7),
    ("Eighth (8th) grade",  "student_g8", 8),
    ("Adult",               "student_ga", "Adult"),
]

rows = []
for grade_str, uid, _ in grade_cases:
    rows.append((uid, INSTR, "Q2_2", grade_str))
    rows.append((uid, INSTR, "Q2_3", "True"))
    rows.append((uid, INSTR, "Q2_4", "Female"))

db = make_db(rows)
df = extract_demographics(db)

for grade_str, uid, expected_level in grade_cases:
    row = df[df["user_id"] == uid]
    check(f"{uid} grade raw",   row["grade"].iloc[0],       grade_str)
    check(f"{uid} grade_level", row["grade_level"].iloc[0], expected_level)

# -----------------------------------------------------------------------
# Test Group 2 — first_language_english normalization (case variants)
# -----------------------------------------------------------------------
print("\n[Test Group 2] first_language_english — all case variants")

lang_cases = [
    ("True",  True),
    ("true",  True),
    ("TRUE",  True),
    ("False", False),
    ("false", False),
    ("FALSE", False),
]

for stored_val, expected_bool in lang_cases:
    db = make_db([
        ("u_lang", INSTR, "Q2_2", "Fourth (4th) grade"),
        ("u_lang", INSTR, "Q2_3", stored_val),
        ("u_lang", INSTR, "Q2_4", "Male"),
    ])
    df = extract_demographics(db)
    check(
        f"Q2_3={stored_val!r} → {expected_bool}",
        df["first_language_english"].iloc[0],
        expected_bool,
    )

# -----------------------------------------------------------------------
# Test Group 3 — Gender normalization
# -----------------------------------------------------------------------
print("\n[Test Group 3] Gender normalization")

for gender_val in ("Male", "Female"):
    db = make_db([
        ("u_gen", INSTR, "Q2_2", "Fifth (5th) grade"),
        ("u_gen", INSTR, "Q2_3", "True"),
        ("u_gen", INSTR, "Q2_4", gender_val),
    ])
    df = extract_demographics(db)
    check(f"gender={gender_val!r}", df["gender"].iloc[0], gender_val)

# -----------------------------------------------------------------------
# Test Group 4 — Realistic full row (matches actual DB sample)
# ant_ub: grade=Fourth (4th), lang=True, gender=Female
# -----------------------------------------------------------------------
print("\n[Test Group 4] Realistic full row matching actual DB sample")

db = make_db([
    ("ant_ub", INSTR, "Q2_2", "Fourth (4th) grade"),
    ("ant_ub", INSTR, "Q2_3", "True"),
    ("ant_ub", INSTR, "Q2_4", "Female"),
])
df = extract_demographics(db)
row = df[df["user_id"] == "ant_ub"].iloc[0]

check("ant_ub grade",                   row["grade"],                  "Fourth (4th) grade")
check("ant_ub grade_level",             row["grade_level"],             4)
check("ant_ub gender",                  row["gender"],                  "Female")
check("ant_ub first_language_english",  row["first_language_english"],  True)

# -----------------------------------------------------------------------
# Test Group 5 — Missing questions produce None (not crash)
# -----------------------------------------------------------------------
print("\n[Test Group 5] Missing individual questions → None gracefully")

# User answered Q2_2 only
db = make_db([("u_partial", INSTR, "Q2_2", "Sixth (6th) grade")])
df = extract_demographics(db)
row = df[df["user_id"] == "u_partial"].iloc[0]
check("partial: grade present",             row["grade"],                  "Sixth (6th) grade")
check("partial: grade_level present",       row["grade_level"],             6)
check("partial: gender is None",            row["gender"],                  None)
check("partial: first_language None",       row["first_language_english"],  None)

# -----------------------------------------------------------------------
# Test Group 6 — Duplicate responses: first-attempt-only rule
# -----------------------------------------------------------------------
print("\n[Test Group 6] Duplicate Q2_2 responses — first kept")

db = make_db([
    ("u_dup", INSTR, "Q2_2", "Fourth (4th) grade"),   # first → kept
    ("u_dup", INSTR, "Q2_2", "Eighth (8th) grade"),   # duplicate → dropped
    ("u_dup", INSTR, "Q2_3", "True"),
    ("u_dup", INSTR, "Q2_4", "Male"),
])
df = extract_demographics(db)
check("duplicate: first grade kept",  df[df["user_id"]=="u_dup"]["grade"].iloc[0], "Fourth (4th) grade")
check("duplicate: one row only",      len(df[df["user_id"]=="u_dup"]), 1)

# -----------------------------------------------------------------------
# Test Group 7 — Non-demographics rows are ignored
# -----------------------------------------------------------------------
print("\n[Test Group 7] Non-demographics instrument rows ignored")

db = make_db([
    ("u_noise", INSTR,                          "Q2_2", "Fifth (5th) grade"),
    ("u_noise", INSTR,                          "Q2_3", "False"),
    ("u_noise", INSTR,                          "Q2_4", "Female"),
    ("u_noise", "module1_b4ai_sccces_survey",   "Q2_1", "Agree"),   # noise
    ("u_noise", "pre_ai_misconceptions_assessment", "Q3_1", "True"), # noise
])
df = extract_demographics(db)
check("noise: only 1 row for u_noise",    len(df[df["user_id"]=="u_noise"]), 1)
check("noise: grade correct",             df[df["user_id"]=="u_noise"]["grade"].iloc[0], "Fifth (5th) grade")
check("noise: first_language_english",    df[df["user_id"]=="u_noise"]["first_language_english"].iloc[0], False)

# -----------------------------------------------------------------------
# Test Group 8 — Multiple users, deterministic sort by user_id
# -----------------------------------------------------------------------
print("\n[Test Group 8] Multiple users — sorted by user_id")

db = make_db([
    ("zzz_user", INSTR, "Q2_2", "Adult"),         ("zzz_user", INSTR, "Q2_3", "True"),  ("zzz_user", INSTR, "Q2_4", "Male"),
    ("aaa_user", INSTR, "Q2_2", "Seventh (7th) grade"), ("aaa_user", INSTR, "Q2_3", "False"), ("aaa_user", INSTR, "Q2_4", "Female"),
    ("mmm_user", INSTR, "Q2_2", "Eighth (8th) grade"),  ("mmm_user", INSTR, "Q2_3", "True"),  ("mmm_user", INSTR, "Q2_4", "Male"),
])
df = extract_demographics(db)
check("3 users total",         len(df), 3)
check("first row is aaa_user", df["user_id"].iloc[0], "aaa_user")
check("last row is zzz_user",  df["user_id"].iloc[2], "zzz_user")
check("aaa_user grade_level",  df[df["user_id"]=="aaa_user"]["grade_level"].iloc[0], 7)
check("zzz_user grade_level",  df[df["user_id"]=="zzz_user"]["grade_level"].iloc[0], "Adult")

# -----------------------------------------------------------------------
# Test Group 9 — Output schema is correct
# -----------------------------------------------------------------------
print("\n[Test Group 9] Output schema and columns")

expected_cols = ["user_id", "grade", "grade_level", "gender", "first_language_english"]
check("all expected columns present", list(df.columns), expected_cols)
check("user_id dtype is object/str",  df["user_id"].dtype.kind, "O")

# -----------------------------------------------------------------------
# Test Group 10 — Empty DB returns correct empty schema
# -----------------------------------------------------------------------
print("\n[Test Group 10] Empty demographics → correct empty DataFrame")

db = make_db([("u_other", "module1_b4ai_sccces_survey", "Q2_1", "Agree")])
df_empty = extract_demographics(db)
check("empty: 0 rows",          len(df_empty), 0)
check("empty: columns correct", list(df_empty.columns),
      ["user_id", "grade", "grade_level", "gender", "first_language_english"])

# -----------------------------------------------------------------------
# Test Group 11 — In-memory variant: extract_demographics_from_df
# -----------------------------------------------------------------------
print("\n[Test Group 11] extract_demographics_from_df (in-memory variant)")

responses_df = pd.DataFrame([
    {"user_id": "u_mem", "instrument_name": INSTR, "question_id": "Q2_2", "response_value": "Eighth (8th) grade", "submitted_at": "2026-01-01"},
    {"user_id": "u_mem", "instrument_name": INSTR, "question_id": "Q2_3", "response_value": "false",              "submitted_at": "2026-01-01"},
    {"user_id": "u_mem", "instrument_name": INSTR, "question_id": "Q2_4", "response_value": "Male",               "submitted_at": "2026-01-01"},
    {"user_id": "u_mem", "instrument_name": "module1_b4ai_sccces_survey", "question_id": "Q2_1", "response_value": "Agree", "submitted_at": "2026-01-01"},
])
df_mem = extract_demographics_from_df(responses_df)
row = df_mem[df_mem["user_id"] == "u_mem"].iloc[0]
check("in-memory: grade",                   row["grade"],                  "Eighth (8th) grade")
check("in-memory: grade_level",             row["grade_level"],             8)
check("in-memory: gender",                  row["gender"],                  "Male")
check("in-memory: first_language_english",  row["first_language_english"],  False)
check("in-memory: noise row excluded",      len(df_mem), 1)

# -----------------------------------------------------------------------
# Test Group 12 — FileNotFoundError for bad path
# -----------------------------------------------------------------------
print("\n[Test Group 12] FileNotFoundError for non-existent DB path")

try:
    extract_demographics("/nonexistent/path/responses.db")
    print("  ❌ FAIL  No error raised for missing DB")
    FAIL += 1
except FileNotFoundError as e:
    print(f"  ✅ PASS  FileNotFoundError raised: {e}")
    PASS += 1

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — demographics_extractor verified.")
else:
    print("  ❌ SOME TESTS FAILED — review output above.")
print('='*60)

# cleanup temp files
import os as _os
