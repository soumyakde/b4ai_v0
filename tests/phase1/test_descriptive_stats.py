"""
Phase 1 — Step 1b Unit Tests
test_descriptive_stats.py

Tests: core/analytics/descriptive/descriptive_stats.py

Run from project root:
    python tests/phase1/test_descriptive_stats.py

Expected: ALL TESTS PASSED
"""

import sys, os, importlib.util
import pandas as pd

# -----------------------------------------------------------------------
# Import by file path
# -----------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_THIS_DIR, "..", "..", "core", "analytics", "descriptive", "descriptive_stats.py"),
    os.path.join(_THIS_DIR, "descriptive_stats.py"),
]
_path = next((os.path.normpath(p) for p in _CANDIDATES if os.path.exists(os.path.normpath(p))), None)
if not _path:
    print("ERROR: descriptive_stats.py not found."); sys.exit(1)

_spec = importlib.util.spec_from_file_location("descriptive_stats", _path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# -----------------------------------------------------------------------
# Shared helper — must match descriptive_stats._pct exactly
# -----------------------------------------------------------------------
def _pct(n, total):
    return round(n / total * 100, 2)

count_participants = _mod.count_participants
count_by_gender    = _mod.count_by_gender
count_by_grade     = _mod.count_by_grade
count_by_language  = _mod.count_by_language
count_by_cohort    = _mod.count_by_cohort
participant_summary= _mod.participant_summary

# -----------------------------------------------------------------------
# Test runner
# -----------------------------------------------------------------------
PASS = FAIL = 0

def check(label, got, expected):
    global PASS, FAIL
    if isinstance(expected, float):
        ok = isinstance(got, (int, float)) and abs(float(got) - expected) < 1e-4
    else:
        ok = (got == expected)
    if ok:
        print(f"  ✅ PASS  {label}")
        PASS += 1
    else:
        print(f"  ❌ FAIL  {label}")
        print(f"           got:      {got!r}")
        print(f"           expected: {expected!r}")
        FAIL += 1

def check_true(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  ✅ PASS  {label}")
        PASS += 1
    else:
        print(f"  ❌ FAIL  {label}  {detail}")
        FAIL += 1

# -----------------------------------------------------------------------
# Fixture — 9 participants matching real DB demographics structure
# 5 Female, 4 Male
# grades: 4×4th, 2×5th, 1×6th, 1×7th, 1×Adult
# language: 6 English, 2 Non-English, 1 Unknown
# cohorts: cohort_A×5, cohort_B×3, None×1
# -----------------------------------------------------------------------
demographics_df = pd.DataFrame([
    {"user_id": "aa_ub",  "grade": "Fourth (4th) grade",  "grade_level": 4, "gender": "Female", "first_language_english": True},
    {"user_id": "am_ub",  "grade": "Fifth (5th) grade",   "grade_level": 5, "gender": "Male",   "first_language_english": False},
    {"user_id": "amt_ub", "grade": "Fourth (4th) grade",  "grade_level": 4, "gender": "Female", "first_language_english": True},
    {"user_id": "ant_ub", "grade": "Fourth (4th) grade",  "grade_level": 4, "gender": "Female", "first_language_english": True},
    {"user_id": "dl_ub",  "grade": "Sixth (6th) grade",   "grade_level": 6, "gender": "Male",   "first_language_english": True},
    {"user_id": "gl_ub",  "grade": "Seventh (7th) grade", "grade_level": 7, "gender": "Female", "first_language_english": False},
    {"user_id": "mz_ub",  "grade": "Fifth (5th) grade",   "grade_level": 5, "gender": "Male",   "first_language_english": True},
    {"user_id": "oz_ub",  "grade": "Fourth (4th) grade",  "grade_level": 4, "gender": "Male",   "first_language_english": True},
    {"user_id": "svs_ub", "grade": "Adult",               "grade_level": "Adult", "gender": "Female", "first_language_english": None},
])

cohort_map = {
    "aa_ub":  "cohort_A",
    "am_ub":  "cohort_A",
    "amt_ub": "cohort_A",
    "ant_ub": "cohort_A",
    "dl_ub":  "cohort_A",
    "gl_ub":  "cohort_B",
    "mz_ub":  "cohort_B",
    "oz_ub":  "cohort_B",
    "svs_ub": None,        # unassigned
}

# -----------------------------------------------------------------------
# Group 1 — count_participants
# -----------------------------------------------------------------------
print("\n[Group 1] count_participants")

check("9 participants total",   count_participants(demographics_df), 9)
check("empty df returns 0",     count_participants(pd.DataFrame(columns=demographics_df.columns)), 0)
check("None df returns 0",      count_participants(None), 0)

# -----------------------------------------------------------------------
# Group 2 — count_by_gender
# -----------------------------------------------------------------------
print("\n[Group 2] count_by_gender")

g = count_by_gender(demographics_df)

check_true("correct columns", set(g.columns) == {"gender", "n", "pct"})
check("row count = 2 genders",  len(g), 2)  # no Unknown in this fixture

female = g[g.gender == "Female"].iloc[0]
male   = g[g.gender == "Male"].iloc[0]

check("Female n=5",   int(female.n),    5)
check("Female pct",   float(female.pct), _pct(5, 9))
check("Male n=4",     int(male.n),      4)
check("Male pct",     float(male.pct),   _pct(4, 9))
check_true("pcts sum to 100", abs(g["pct"].sum() - 100.0) < 0.01)

# Sort order: Male first? No — Female first per spec (Male, Female, Unknown)
check("sort: Male first",   g.iloc[0]["gender"], "Male")
check("sort: Female second", g.iloc[1]["gender"], "Female")

# -----------------------------------------------------------------------
# Group 3 — count_by_gender with Unknown
# -----------------------------------------------------------------------
print("\n[Group 3] count_by_gender — Unknown handling")

df_unknown = demographics_df.copy()
df_unknown.loc[df_unknown.user_id == "svs_ub", "gender"] = None

g2 = count_by_gender(df_unknown)
check("3 rows with Unknown", len(g2), 3)
unknown_row = g2[g2.gender == "Unknown"].iloc[0]
check("Unknown n=1",    int(unknown_row.n),    1)
check("Unknown pct",    float(unknown_row.pct), _pct(1, 9))
check("Unknown is last", g2.iloc[-1]["gender"], "Unknown")

# -----------------------------------------------------------------------
# Group 4 — count_by_grade
# -----------------------------------------------------------------------
print("\n[Group 4] count_by_grade")

gr = count_by_grade(demographics_df)

check_true("correct columns", set(gr.columns) == {"grade_level", "grade", "n", "pct"})
# grades in fixture: 4th×4, 5th×2, 6th×1, 7th×1, Adult×1
check("5 grade rows", len(gr), 5)

row_4 = gr[gr.grade_level == 4].iloc[0]
check("grade 4 n=4",   int(row_4.n), 4)
check("grade 4 pct",   float(row_4.pct), _pct(4, 9))
check("grade 4 label", row_4.grade, "Fourth (4th) grade")

row_adult = gr[gr.grade_level == "Adult"].iloc[0]
check("Adult n=1",  int(row_adult.n), 1)
check("Adult label", row_adult.grade, "Adult")

# Sort: 4, 5, 6, 7, Adult
check("sort order: grade 4 first",  gr.iloc[0]["grade_level"], 4)
check("sort order: Adult last",     gr.iloc[-1]["grade_level"], "Adult")
check_true("pcts sum ~100 (rounding tolerance 0.1)", abs(gr["pct"].sum() - 100.0) < 0.1)

# -----------------------------------------------------------------------
# Group 5 — count_by_grade with Unknown
# -----------------------------------------------------------------------
print("\n[Group 5] count_by_grade — Unknown handling")

df_unk_grade = demographics_df.copy()
df_unk_grade.loc[df_unk_grade.user_id == "oz_ub", "grade_level"] = None
df_unk_grade.loc[df_unk_grade.user_id == "oz_ub", "grade"]       = None

gr2 = count_by_grade(df_unk_grade)
unknown_grade = gr2[gr2.grade_level == "Unknown"]
check("Unknown grade row present", len(unknown_grade), 1)
check("Unknown grade n=1", int(unknown_grade.iloc[0].n), 1)
check("Unknown grade is last", gr2.iloc[-1]["grade_level"], "Unknown")
# Adult should precede Unknown
adult_idx   = gr2[gr2.grade_level == "Adult"].index[0]
unknown_idx = gr2[gr2.grade_level == "Unknown"].index[0]
check_true("Adult before Unknown", adult_idx < unknown_idx)

# -----------------------------------------------------------------------
# Group 6 — count_by_language
# -----------------------------------------------------------------------
print("\n[Group 6] count_by_language")

lang = count_by_language(demographics_df)

check_true("correct columns",
    set(lang.columns) == {"first_language_english", "label", "n", "pct"})

# fixture: 6 English, 2 Non-English, 1 Unknown (svs_ub)
check("3 language rows", len(lang), 3)

eng  = lang[lang.label == "English"].iloc[0]
neng = lang[lang.label == "Non-English"].iloc[0]
unk  = lang[lang.label == "Unknown"].iloc[0]

check("English n=6",            int(eng.n), 6)
check("English pct",            float(eng.pct), _pct(6, 9))
check("English bool=True",      eng.first_language_english, True)
check("Non-English n=2",        int(neng.n), 2)
check("Non-English bool=False", neng.first_language_english, False)
check("Unknown n=1",            int(unk.n), 1)
check("Unknown bool=None",      unk.first_language_english, None)

check("sort: English first",     lang.iloc[0]["label"], "English")
check("sort: Non-English second",lang.iloc[1]["label"], "Non-English")
check("sort: Unknown last",      lang.iloc[2]["label"], "Unknown")
check_true("pcts sum to 100", abs(lang["pct"].sum() - 100.0) < 0.01)

# -----------------------------------------------------------------------
# Group 7 — count_by_cohort
# -----------------------------------------------------------------------
print("\n[Group 7] count_by_cohort")

coh = count_by_cohort(demographics_df, cohort_map)

check_true("correct columns", set(coh.columns) == {"cohort_id", "n", "pct"})
check("3 cohort rows (A, B, Unassigned)", len(coh), 3)

ca = coh[coh.cohort_id == "cohort_A"].iloc[0]
cb = coh[coh.cohort_id == "cohort_B"].iloc[0]
cu = coh[coh.cohort_id == "Unassigned"].iloc[0]

check("cohort_A n=5",          int(ca.n), 5)
check("cohort_A pct",          float(ca.pct), _pct(5, 9))
check("cohort_B n=3",          int(cb.n), 3)
check("Unassigned n=1",        int(cu.n), 1)
check("Unassigned last",       coh.iloc[-1]["cohort_id"], "Unassigned")
check_true("pcts sum to 100",  abs(coh["pct"].sum() - 100.0) < 0.01)

# -----------------------------------------------------------------------
# Group 8 — count_by_cohort: user not in cohort_map
# -----------------------------------------------------------------------
print("\n[Group 8] count_by_cohort — user absent from cohort_map")

partial_map = {uid: v for uid, v in cohort_map.items() if uid != "svs_ub"}
# svs_ub not in map at all → should be "Unassigned"
coh2 = count_by_cohort(demographics_df, partial_map)
unassigned2 = coh2[coh2.cohort_id == "Unassigned"].iloc[0]
check("missing-from-map → Unassigned", int(unassigned2.n), 1)

# -----------------------------------------------------------------------
# Group 9 — participant_summary
# -----------------------------------------------------------------------
print("\n[Group 9] participant_summary")

summary = participant_summary(demographics_df, cohort_map)

check_true("summary has required keys",
    {"total","by_gender","by_grade","by_language","by_cohort"}.issubset(summary.keys()))
check("summary total=9",          summary["total"], 9)
check_true("by_gender is DataFrame",  isinstance(summary["by_gender"],   pd.DataFrame))
check_true("by_grade is DataFrame",   isinstance(summary["by_grade"],    pd.DataFrame))
check_true("by_language is DataFrame",isinstance(summary["by_language"], pd.DataFrame))
check_true("by_cohort is DataFrame",  isinstance(summary["by_cohort"],   pd.DataFrame))

# Without cohort_map — by_cohort should be None
summary_no_cohort = participant_summary(demographics_df)
check("by_cohort=None without cohort_map", summary_no_cohort["by_cohort"], None)

# -----------------------------------------------------------------------
# Group 10 — percentage correctness: hand-verified spot checks
# -----------------------------------------------------------------------
print("\n[Group 10] Percentage arithmetic — hand-verified")

check("5/9 pct = 55.56", _pct(5, 9), 55.56)
check("4/9 pct = 44.44", _pct(4, 9), 44.44)
check("6/9 pct = 66.67", _pct(6, 9), 66.67)
check("2/9 pct = 22.22", _pct(2, 9), 22.22)
check("1/9 pct = 11.11", _pct(1, 9), 11.11)

# Verify pct column uses same formula
female_pct = count_by_gender(demographics_df)[
    count_by_gender(demographics_df)["gender"] == "Female"
]["pct"].iloc[0]
check("Female pct matches _pct(5,9)", float(female_pct), _pct(5, 9))

# -----------------------------------------------------------------------
# Group 11 — Edge cases
# -----------------------------------------------------------------------
print("\n[Group 11] Edge cases")

# Single participant
single = pd.DataFrame([{
    "user_id": "only_one", "grade": "Fifth (5th) grade",
    "grade_level": 5, "gender": "Male", "first_language_english": True,
}])
check("single: count=1",     count_participants(single), 1)
check("single: gender n=1",  int(count_by_gender(single).iloc[0]["n"]), 1)
check("single: pct=100.0",   float(count_by_gender(single).iloc[0]["pct"]), 100.0)

# All Unknown
all_unknown = pd.DataFrame([{
    "user_id": f"u{i}", "grade": None,
    "grade_level": None, "gender": None, "first_language_english": None,
} for i in range(3)])
g_unk = count_by_gender(all_unknown)
check("all Unknown gender: 1 row",      len(g_unk), 1)
check("all Unknown gender label",       g_unk.iloc[0]["gender"], "Unknown")
check("all Unknown gender pct=100",     float(g_unk.iloc[0]["pct"]), 100.0)

lang_unk = count_by_language(all_unknown)
check("all Unknown language: 1 row",    len(lang_unk), 1)
check("all Unknown language pct=100",   float(lang_unk.iloc[0]["pct"]), 100.0)

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — descriptive_stats verified.")
else:
    print("  ❌ SOME TESTS FAILED — review output above.")
print('='*60)
