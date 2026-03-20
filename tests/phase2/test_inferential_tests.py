"""
Phase 2 — Inferential Statistics Unit Tests
test_inferential_tests.py

Run from project root:
    python tests/phase2/test_inferential_tests.py
"""
import sys, os, importlib.util, math
import pandas as pd
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_THIS_DIR,"..","..", "core","analytics","inferential","inferential_tests.py"),
    os.path.join(_THIS_DIR, "inferential_tests.py"),
]
_path = next((os.path.normpath(p) for p in _CANDIDATES if os.path.exists(os.path.normpath(p))), None)
if not _path:
    print("ERROR: inferential_tests.py not found"); sys.exit(1)

_spec = importlib.util.spec_from_file_location("inferential_tests", _path)
_mod  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)

run_paired_comparison  = _mod.run_paired_comparison
run_between_groups     = _mod.run_between_groups
run_repeated_measures  = _mod.run_repeated_measures

PASS = FAIL = 0
def check(label, got, expected, tol=1e-3):
    global PASS, FAIL
    if isinstance(expected, float):
        ok = isinstance(got,(int,float)) and abs(float(got)-expected) < tol
    else:
        ok = (got == expected)
    if ok: print(f"  ✅ PASS  {label}"); PASS += 1
    else:
        print(f"  ❌ FAIL  {label}  got={got!r}  expected={expected!r}"); FAIL += 1

def check_true(label, cond, detail=""):
    global PASS, FAIL
    if cond: print(f"  ✅ PASS  {label}"); PASS += 1
    else: print(f"  ❌ FAIL  {label}  {detail}"); FAIL += 1

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------
def _make_rows(user_id, instrument_key, module_id, scores, construct=None):
    rows = []
    for i, s in enumerate(scores):
        rows.append(dict(
            user_id=user_id, instrument_key=instrument_key,
            question_id=f"Q{i}", item_score=float(s),
            construct=construct, module_id=module_id,
            grade="Fourth (4th) grade", cohort_id="cohort_A",
            response_value="X", submitted_at="2026-01-01", completed_at=None,
        ))
    return rows

# 9 users, pre and post misconceptions (8 items each)
# Designed so post > pre for most users → significant improvement
PRE_SCORES  = [5, 2, 5, 5, 5, 5, 4, 3, 2]  # ant_ub, etc.
POST_SCORES = [7, 5, 6, 7, 8, 7, 6, 5, 4]
USERS = ["aa_ub","am_ub","amt_ub","ant_ub","dl_ub","gl_ub","mz_ub","oz_ub","svs_ub"]

rows = []
for u, pre, post in zip(USERS, PRE_SCORES, POST_SCORES):
    rows += _make_rows(u, "precourse_pre_ai_misconceptions_assessment",  "global",
                       [1.0]*pre + [0.0]*(8-pre))
    rows += _make_rows(u, "postcourse_post_ai_misconceptions_assessment", "global",
                       [1.0]*post + [0.0]*(8-post))

# Grade groups: 4th (u1-u4), 5th (u5-u6), Adult (u7-u9)
GRADES = ["Fourth (4th) grade"]*4 + ["Fifth (5th) grade"]*2 + ["Adult"]*3
demographics_df = pd.DataFrame([
    {"user_id": u, "grade": g, "grade_level": None,
     "gender": "Female", "first_language_english": True}
    for u, g in zip(USERS, GRADES)
])

# MCQ repeated measures across 3 modules (simplified)
# Each user gets scores improving slightly across modules
for n_mod, pct in enumerate([0.5, 0.6, 0.7], 1):
    for u in USERS:
        correct = int(pct * 10)
        rows += _make_rows(u, f"module{n_mod}_content_mcq_assessment",
                           f"module_{n_mod}",
                           [1.0]*correct + [0.0]*(10-correct))

# SIMS intrinsic_motivation across 3 modules
for n_mod in range(1, 4):
    for i, u in enumerate(USERS):
        # Slight increase per module
        base = 2.5 + i*0.1 + n_mod*0.1
        scores = [min(4.0, round(base + j*0.05, 1)) for j in range(3)]
        rows += _make_rows(u, f"module{n_mod}_b4ai_sims_survey",
                           f"module_{n_mod}", scores,
                           construct="intrinsic_motivation")

canonical_df = pd.DataFrame(rows)

# -----------------------------------------------------------------------
# Group 1 — run_paired_comparison: basic structure
# -----------------------------------------------------------------------
print("\n[Group 1] run_paired_comparison — structure and keys")

r = run_paired_comparison(
    canonical_df,
    pre_instrument="precourse_pre_ai_misconceptions_assessment",
    post_instrument="postcourse_post_ai_misconceptions_assessment",
    alpha=0.05, use_pct=True,
)

check("no error",       r["error"], None)
check("n_pairs=9",      r["n_pairs"], 9)
check_true("pre_mean in range",  0 < r["pre_mean"] < 100)
check_true("post_mean > pre_mean (improvement)", r["post_mean"] > r["pre_mean"])
check_true("mean_diff > 0",      r["mean_diff"] > 0)
check_true("t_stat is float",    isinstance(r["t_stat"], float))
check_true("t_p_value in [0,1]", 0 <= r["t_p_value"] <= 1)
check_true("significant is bool",isinstance(r["significant"], bool))
check_true("cohens_d > 0",       r["cohens_d"] > 0)
check_true("effect_size_label valid",
    r["effect_size_label"] in ("negligible","small","medium","large"))
check("wilcoxon None by default", r["wilcoxon_stat"], None)
check_true("power_achieved in [0,1]", 0 <= r["power_achieved"] <= 1)
check_true("n_needed_80 >= 2",   r["n_needed_80"] >= 2)
check_true("n_needed_95 >= n_needed_80", r["n_needed_95"] >= r["n_needed_80"])
check("low_n_warning True (n=9)", r["low_n_warning"], True)

# -----------------------------------------------------------------------
# Group 2 — Wilcoxon toggle
# -----------------------------------------------------------------------
print("\n[Group 2] Wilcoxon signed-rank toggle")

r_w = run_paired_comparison(
    canonical_df,
    "precourse_pre_ai_misconceptions_assessment",
    "postcourse_post_ai_misconceptions_assessment",
    include_wilcoxon=True,
)
check_true("wilcoxon_stat is float", isinstance(r_w["wilcoxon_stat"], float))
check_true("wilcoxon_p in [0,1]",    0 <= r_w["wilcoxon_p"] <= 1)

# -----------------------------------------------------------------------
# Group 3 — Known math: perfect improvement (d should be large)
# -----------------------------------------------------------------------
print("\n[Group 3] Effect size math — known data")

# All users improve by exactly 2 items (consistent effect)
rows_known = []
for u in USERS:
    rows_known += _make_rows(u, "pre_test", "global", [1.0]*4 + [0.0]*4)
    rows_known += _make_rows(u, "post_test", "global", [1.0]*6 + [0.0]*2)
df_known = pd.DataFrame(rows_known)

r_k = run_paired_comparison(df_known, "pre_test", "post_test",
                            use_pct=True)
check("no error",          r_k["error"], None)
# pre pct = 50.0, post pct = 75.0, diff = 25.0 for all users
check("pre_mean=50.0",     r_k["pre_mean"],  50.0)
check("post_mean=75.0",    r_k["post_mean"], 75.0)
check("mean_diff=25.0",    r_k["mean_diff"], 25.0)
# d = mean_diff / std_diff; all diffs identical → std=0 → d=0 edge case handled
check_true("cohens_d finite", math.isfinite(r_k["cohens_d"]))
check("significant=True",  r_k["significant"], True)

# -----------------------------------------------------------------------
# Group 4 — run_between_groups: structure and keys
# -----------------------------------------------------------------------
print("\n[Group 4] run_between_groups — structure")

r_bg = run_between_groups(
    canonical_df,
    instrument_key="precourse_pre_ai_misconceptions_assessment",
    group_col="grade",
    demographics_df=demographics_df,
    alpha=0.05, use_pct=True,
)

check("no error",               r_bg["error"], None)
check_true("3 grade groups",    len(r_bg["groups"]) == 3)
check_true("n_per_group sums to 9",
    sum(r_bg["n_per_group"].values()) == 9)
check_true("group_means all in 0-100",
    all(0 <= v <= 100 for v in r_bg["group_means"].values()))
check_true("f_stat >= 0",       r_bg["f_stat"] >= 0)
check_true("anova_p in [0,1]",  0 <= r_bg["anova_p"] <= 1)
check_true("eta_squared in [0,1]", 0 <= r_bg["eta_squared"] <= 1)
check_true("kruskal_stat >= 0", r_bg["kruskal_stat"] >= 0)
check_true("kruskal_p in [0,1]",0 <= r_bg["kruskal_p"] <= 1)
check_true("power_achieved in [0,1]", 0 <= r_bg["power_achieved"] <= 1)
check_true("low_n_warning True", r_bg["low_n_warning"])

# -----------------------------------------------------------------------
# Group 5 — run_between_groups: error on missing demographics
# -----------------------------------------------------------------------
print("\n[Group 5] run_between_groups — error handling")

r_err = run_between_groups(
    canonical_df, "precourse_pre_ai_misconceptions_assessment",
    group_col="gender",   # requires demographics_df
    demographics_df=None,
)
check_true("error set when demo missing", r_err["error"] is not None)
check_true("error mentions demographics_df",
    "demographics_df" in str(r_err["error"]))

# -----------------------------------------------------------------------
# Group 6 — run_repeated_measures: MCQ across modules
# -----------------------------------------------------------------------
print("\n[Group 6] run_repeated_measures — MCQ across modules")

r_rm = run_repeated_measures(
    canonical_df,
    instrument_key="content_mcq_assessment",
    construct=None,
    alpha=0.05,
)
check("no error",               r_rm["error"], None)
check("n_subjects=9",           r_rm["n_subjects"], 9)
check_true("3 time points",     len(r_rm["time_points"]) == 3)
check_true("means_by_time populated",
    len(r_rm["means_by_time"]) == 3)
check_true("friedman_stat >= 0", r_rm["friedman_stat"] >= 0)
check_true("p_value in [0,1]",  0 <= r_rm["p_value"] <= 1)
check_true("kendalls_w in [0,1]", 0 <= r_rm["kendalls_w"] <= 1)
check_true("effect_size_label valid",
    r_rm["effect_size_label"] in ("negligible","small","medium","large"))
check("low_n_warning True",     r_rm["low_n_warning"], True)

# -----------------------------------------------------------------------
# Group 7 — run_repeated_measures: survey construct across modules
# -----------------------------------------------------------------------
print("\n[Group 7] run_repeated_measures — SIMS construct across modules")

r_s = run_repeated_measures(
    canonical_df,
    instrument_key="b4ai_sims_survey",
    construct="intrinsic_motivation",
)
check("no error",       r_s["error"], None)
check("n_subjects=9",   r_s["n_subjects"], 9)
check_true("3 time points", len(r_s["time_points"]) == 3)
check_true("all means in Likert range",
    all(1.0 <= v <= 4.0 for v in r_s["means_by_time"].values()))
check_true("kendalls_w in [0,1]", 0 <= r_s["kendalls_w"] <= 1)

# -----------------------------------------------------------------------
# Group 8 — Power analysis math spot checks
# -----------------------------------------------------------------------
print("\n[Group 8] Power analysis — spot checks")

# Known: n=9, d=0.8 (large), alpha=0.05 → power should be modest
_power_paired_ttest    = _mod._power_paired_ttest
_n_needed_paired_ttest = _mod._n_needed_paired_ttest
p9 = _power_paired_ttest(9, 0.8, 0.05)
check_true("power(n=9,d=0.8) > 0.3",  p9 > 0.3)
check_true("power(n=9,d=0.8) < 1.0",  p9 < 1.0)

# n=50 should give much higher power
p50 = _power_paired_ttest(50, 0.5, 0.05)
check_true("power(n=50,d=0.5) > 0.7", p50 > 0.7)

# n_needed should decrease as d increases
n80_small  = _n_needed_paired_ttest(0.2, 0.80)
n80_large  = _n_needed_paired_ttest(0.8, 0.80)
check_true("n_needed: small d > large d", n80_small > n80_large)

# n_needed_95 > n_needed_80 for same d
n80 = _n_needed_paired_ttest(0.5, 0.80)
n95 = _n_needed_paired_ttest(0.5, 0.95)
check_true("n_needed_95 > n_needed_80", n95 > n80)

# -----------------------------------------------------------------------
# Group 9 — Edge cases
# -----------------------------------------------------------------------
print("\n[Group 9] Edge cases")

# Instrument not in canonical_df
r_miss = run_paired_comparison(
    canonical_df, "nonexistent_pre", "nonexistent_post",
)
check_true("error on missing instrument", r_miss["error"] is not None)

# Only 1 user → error
single_row = pd.DataFrame([rows[0]])
r_single = run_paired_comparison(
    single_row, "precourse_pre_ai_misconceptions_assessment",
    "postcourse_post_ai_misconceptions_assessment",
)
check_true("error on n<2", r_single["error"] is not None)

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — inferential_tests verified.")
else:
    print("  ❌ SOME TESTS FAILED — review output above.")
print('='*60)
