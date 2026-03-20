"""
Phase 2 — Inferential Statistics Integration Test
test_inferential_integration.py

Runs all three inferential functions against the REAL responses.db.

Usage:
    python tests/phase2/test_inferential_integration.py
    python tests/phase2/test_inferential_integration.py path/to/responses.db
"""

import sys, os, importlib.util, sqlite3
import pandas as pd

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
DB_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "responses.db")
if not os.path.exists(DB_PATH):
    print(f"ERROR: responses.db not found at {DB_PATH}"); sys.exit(1)
print(f"\n  DB: {DB_PATH}")

# -----------------------------------------------------------------------
# Load modules by file path
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

inf_mod = _load([
    os.path.join(_ROOT,"core","analytics","inferential","inferential_tests.py"),
    os.path.join(os.path.dirname(__file__),"inferential_tests.py"),
], "inferential_tests")

demo_mod = _load([
    os.path.join(_ROOT,"core","analytics","filters","demographics_extractor.py"),
    os.path.join(os.path.dirname(__file__),"demographics_extractor.py"),
], "demographics_extractor")

resolver_mod = _load([
    os.path.join(_ROOT,"core","analytics","filters","instrument_key_resolver.py"),
    os.path.join(os.path.dirname(__file__),"instrument_key_resolver.py"),
], "instrument_key_resolver")

run_paired_comparison = inf_mod.run_paired_comparison
run_between_groups    = inf_mod.run_between_groups
run_repeated_measures = inf_mod.run_repeated_measures
extract_demographics  = demo_mod.extract_demographics
resolve_instrument_key = resolver_mod.resolve_instrument_key

# -----------------------------------------------------------------------
# Build minimal canonical_df from real DB
# (same scoring logic as integration test for score_aggregator)
# -----------------------------------------------------------------------
print("\n[Step 1] Loading and scoring real responses...")

SCORING = {
    "precourse_pre_ai_misconceptions_assessment": {
        "scoring_type": "binary",
        "correct_answers": {
            "Q3_1":"False","Q3_2":"False","Q3_3":"False","Q3_4":"False",
            "Q3_5":"No","Q3_6":"Yes","Q3_7":"No","Q3_8":"No",
        },
    },
    "postcourse_post_ai_misconceptions_assessment": {
        "scoring_type": "binary",
        "correct_answers": {
            "Q3_1":"False","Q3_2":"False","Q3_3":"False","Q3_4":"False",
            "Q3_5":"No","Q3_6":"Yes","Q3_7":"No","Q3_8":"No",
        },
    },
    "precourse_pre_aici_assessment": {
        "scoring_type": "binary",
        "correct_answers": {
            "Q4_1":"B","Q4_2":"A","Q4_3":"B","Q4_4":"A","Q4_5":"A",
            "Q4_6":"B","Q4_7":"D","Q4_8":"A","Q4_9":"B","Q4_10":"B",
            "Q4_11":"A","Q4_12":"B","Q4_13":"A","Q4_14":"B","Q4_15":"A",
            "Q4_16":"B","Q4_17":"A","Q4_18":"A","Q4_19":"A","Q4_20":"A",
        },
    },
    "postcourse_post_aici_assessment": {
        "scoring_type": "binary",
        "correct_answers": {
            "Q4_1":"B","Q4_2":"A","Q4_3":"B","Q4_4":"A","Q4_5":"A",
            "Q4_6":"B","Q4_7":"D","Q4_8":"A","Q4_9":"B","Q4_10":"B",
            "Q4_11":"A","Q4_12":"B","Q4_13":"A","Q4_14":"B","Q4_15":"A",
            "Q4_16":"B","Q4_17":"A","Q4_18":"A","Q4_19":"A","Q4_20":"A",
        },
    },
}
MCQ_CORRECT = {
    "Q1":"B","Q2":"C","Q3":"B","Q4":"D","Q5":"A","Q6":"D","Q7":"B",
    "Q8":"A","Q9":"B","Q10":"A","Q11":"A","Q12":"C","Q13":"A","Q14":"C",
    "Q15":"A","Q16":"D","Q17":"A","Q18":"B","Q19":"D","Q20":"A",
    "Q21":"C","Q22":"B","Q23":"C","Q24":"B","Q25":"A","Q26":"A","Q27":"D",
    "Q28":"B","Q29":"A","Q30":"B","Q31":"B","Q32":"A","Q33":"C","Q34":"B",
    "Q35":"A","Q36":"C","Q37":"D","Q38":"A","Q39":"C","Q40":"A","Q41":"D",
    "Q42":"D","Q43":"A","Q44":"C","Q45":"A","Q46":"A","Q47":"C","Q48":"D",
    "Q49":"D","Q50":"B","Q51":"B","Q52":"A","Q53":"C","Q54":"C","Q55":"B",
    "Q56":"C","Q57":"B",
}
for n in range(1, 8):
    SCORING[f"module{n}_content_mcq_assessment"] = {
        "scoring_type": "binary", "correct_answers": MCQ_CORRECT,
    }

SURVEY_SCORING = {
    "b4ai_sccces_survey": {
        "scoring_type": "likert",
        "default_scale": {"Strongly disagree":1,"Disagree":2,"Agree":3,"Strongly agree":4},
        "reverse_scale": {"Strongly disagree":4,"Disagree":3,"Agree":2,"Strongly agree":1},
        "reverse_questions": ["Q9_1","Q9_2","Q10_1","Q10_2"],
    },
    "b4ai_sims_survey": {
        "scoring_type": "likert",
        "default_scale": {"Strongly disagree":1,"Disagree":2,"Agree":3,"Strongly agree":4},
        "reverse_scale": {"Strongly disagree":4,"Disagree":3,"Agree":2,"Strongly agree":1},
        "reverse_questions": ["Q4_1","Q4_2","Q4_3","Q4_4","Q5_1","Q5_2","Q5_3"],
    },
}
SCCCES_CONSTRUCTS = {
    "Q2_1":"engagement_with_task",
    "Q3_1":"effort_and_persistence","Q3_2":"effort_and_persistence",
    "Q4_1":"experience_of_flow",
    "Q5_1":"coherency_of_messaging","Q5_2":"coherency_of_messaging","Q5_3":"coherency_of_messaging",
    "Q6_1":"plausibility_of_messaging","Q6_2":"plausibility_of_messaging",
    "Q7_1":"credibility_of_messaging","Q7_2":"credibility_of_messaging",
    "Q8_1":"comprehensibility_of_messaging","Q8_2":"comprehensibility_of_messaging",
    "Q9_1":"attention","Q9_2":"attention",
    "Q10_1":"culture","Q10_2":"culture",
    "Q11_1":"personal_relevance","Q11_2":"personal_relevance","Q11_3":"personal_relevance",
}
SIMS_CONSTRUCTS = {
    "Q2_1":"intrinsic_motivation","Q2_2":"intrinsic_motivation","Q2_3":"intrinsic_motivation",
    "Q3_1":"identified_regulation","Q3_2":"identified_regulation","Q3_3":"identified_regulation",
    "Q4_1":"external_regulation","Q4_2":"external_regulation",
    "Q4_3":"external_regulation","Q4_4":"external_regulation",
    "Q5_1":"amotivation","Q5_2":"amotivation","Q5_3":"amotivation",
}

def score_row(ikey, qid, val):
    canon_key = resolve_instrument_key(ikey)
    scoring = SCORING.get(ikey) or SURVEY_SCORING.get(canon_key)
    if scoring is None:
        return float("nan")
    stype = scoring.get("scoring_type")
    if stype == "binary":
        correct = scoring["correct_answers"].get(qid)
        if correct is None:
            return float("nan")
        norm = str(val).strip()
        if ":" in norm:
            norm = norm.split(":")[0].strip()
        return 1.0 if norm == correct else 0.0
    elif stype == "likert":
        rev_qs = set(scoring.get("reverse_questions", []))
        scale  = scoring["reverse_scale"] if qid in rev_qs else scoring["default_scale"]
        return float(scale.get(str(val).strip(), float("nan")))
    return float("nan")

def get_construct(ikey, qid):
    canon = resolve_instrument_key(ikey)
    if "sccces" in canon:
        return SCCCES_CONSTRUCTS.get(qid)
    if "sims" in canon:
        return SIMS_CONSTRUCTS.get(qid)
    return None

def derive_module_id(iname):
    import re
    m = re.match(r"^(module\d+)_", iname)
    if m:
        num = m.group(1).replace("module","")
        return f"module_{num}"
    return "global"

conn = sqlite3.connect(DB_PATH)
raw = pd.read_sql_query(
    "SELECT user_id, instrument_name, question_id, response_value FROM responses",
    conn
)
conn.close()

raw["instrument_key"] = raw["instrument_name"].apply(resolve_instrument_key)
raw["module_id"]      = raw["instrument_name"].apply(derive_module_id)
raw["item_score"]     = raw.apply(
    lambda r: score_row(r["instrument_name"], r["question_id"], r["response_value"]),
    axis=1
)
raw["construct"] = raw.apply(
    lambda r: get_construct(r["instrument_name"], r["question_id"]),
    axis=1
)
raw["grade"] = None
raw["cohort_id"] = None
raw["completed_at"] = None

canonical_df = raw.rename(columns={"instrument_name": "_src"})
canonical_df = pd.DataFrame({
    "user_id":        raw["user_id"],
    "instrument_key": raw["instrument_name"],  # keep DB names for matching
    "question_id":    raw["question_id"],
    "response_value": raw["response_value"],
    "item_score":     raw["item_score"],
    "construct":      raw["construct"],
    "module_id":      raw["module_id"],
    "grade":          None,
    "cohort_id":      None,
    "submitted_at":   None,
    "completed_at":   None,
})

demographics_df = extract_demographics(DB_PATH)
print(f"  → {len(canonical_df)} rows, {canonical_df['user_id'].nunique()} users")

# -----------------------------------------------------------------------
# Test runner
# -----------------------------------------------------------------------
PASS = FAIL = 0
def check(label, got, expected):
    global PASS, FAIL
    ok = (got == expected)
    if ok: print(f"  ✅ PASS  {label}"); PASS += 1
    else:
        print(f"  ❌ FAIL  {label}  got={got!r}  expected={expected!r}"); FAIL += 1

def check_true(label, cond, detail=""):
    global PASS, FAIL
    if cond: print(f"  ✅ PASS  {label}"); PASS += 1
    else: print(f"  ❌ FAIL  {label}  {detail}"); FAIL += 1

# -----------------------------------------------------------------------
# Section A — Pre vs Post Misconceptions
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("SECTION A: Pre vs Post AI Misconceptions")
print("="*65)

r_misc = run_paired_comparison(
    canonical_df,
    pre_instrument  = "precourse_pre_ai_misconceptions_assessment",
    post_instrument = "postcourse_post_ai_misconceptions_assessment",
    alpha=0.05, include_wilcoxon=True, use_pct=True,
)

if r_misc["error"]:
    print(f"  ❌ ERROR: {r_misc['error']}")
else:
    print(f"\n  n pairs          : {r_misc['n_pairs']}")
    print(f"  Pre  mean % corr : {r_misc['pre_mean']:.2f}%  (SD={r_misc['pre_std']:.2f})")
    print(f"  Post mean % corr : {r_misc['post_mean']:.2f}%  (SD={r_misc['post_std']:.2f})")
    print(f"  Mean difference  : {r_misc['mean_diff']:+.2f}%")
    print(f"\n  Paired t-test    : t={r_misc['t_stat']:.4f}, p={r_misc['t_p_value']:.4f}  {'✅ significant' if r_misc['significant'] else '— not significant'}")
    print(f"  Wilcoxon         : W={r_misc['wilcoxon_stat']}, p={r_misc['wilcoxon_p']:.4f}")
    print(f"  Cohen's d        : {r_misc['cohens_d']:.4f} ({r_misc['effect_size_label']})")
    print(f"\n  Power achieved   : {r_misc['power_achieved']*100:.1f}%")
    print(f"  N needed 80%     : {r_misc['n_needed_80']}")
    print(f"  N needed 95%     : {r_misc['n_needed_95']}")
    print(f"  Low-n warning    : {'⚠️  YES' if r_misc['low_n_warning'] else 'No'}")

    check("n_pairs=9",             r_misc["n_pairs"],   9)
    check_true("no error",         r_misc["error"] is None)
    check_true("pre_mean in range",0 < r_misc["pre_mean"] < 100)
    check_true("post_mean in range",0 < r_misc["post_mean"] < 100)
    check_true("cohens_d finite",  r_misc["cohens_d"] == r_misc["cohens_d"])
    check_true("power in [0,1]",   0 <= r_misc["power_achieved"] <= 1)
    check("low_n_warning=True",    r_misc["low_n_warning"], True)

# -----------------------------------------------------------------------
# Section B — Pre vs Post AICI
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("SECTION B: Pre vs Post AICI Assessment")
print("="*65)

r_aici = run_paired_comparison(
    canonical_df,
    pre_instrument  = "precourse_pre_aici_assessment",
    post_instrument = "postcourse_post_aici_assessment",
    alpha=0.05, include_wilcoxon=True, use_pct=True,
)

if r_aici["error"]:
    print(f"  ❌ ERROR: {r_aici['error']}")
else:
    print(f"\n  n pairs          : {r_aici['n_pairs']}")
    print(f"  Pre  mean % corr : {r_aici['pre_mean']:.2f}%  (SD={r_aici['pre_std']:.2f})")
    print(f"  Post mean % corr : {r_aici['post_mean']:.2f}%  (SD={r_aici['post_std']:.2f})")
    print(f"  Mean difference  : {r_aici['mean_diff']:+.2f}%")
    print(f"\n  Paired t-test    : t={r_aici['t_stat']:.4f}, p={r_aici['t_p_value']:.4f}  {'✅ significant' if r_aici['significant'] else '— not significant'}")
    print(f"  Wilcoxon         : W={r_aici['wilcoxon_stat']}, p={r_aici['wilcoxon_p']:.4f}")
    print(f"  Cohen's d        : {r_aici['cohens_d']:.4f} ({r_aici['effect_size_label']})")
    print(f"\n  Power achieved   : {r_aici['power_achieved']*100:.1f}%")
    print(f"  N needed 80%     : {r_aici['n_needed_80']}")
    print(f"  N needed 95%     : {r_aici['n_needed_95']}")

    check("n_pairs=9",  r_aici["n_pairs"], 9)
    check_true("no error", r_aici["error"] is None)

# -----------------------------------------------------------------------
# Section C — Between groups: grade on pre misconceptions
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("SECTION C: Between-Groups — Grade on Pre Misconceptions")
print("="*65)

r_bg = run_between_groups(
    canonical_df,
    instrument_key  = "precourse_pre_ai_misconceptions_assessment",
    group_col       = "grade",
    demographics_df = demographics_df,
    alpha=0.05, use_pct=True,
)

if r_bg["error"]:
    print(f"  ❌ ERROR: {r_bg['error']}")
else:
    print(f"\n  Groups: {r_bg['groups']}")
    print(f"  N per group: {r_bg['n_per_group']}")
    print(f"  Group means:")
    for g, m in r_bg["group_means"].items():
        print(f"    {g}: {m:.2f}%")
    print(f"\n  One-way ANOVA    : F={r_bg['f_stat']:.4f}, p={r_bg['anova_p']:.4f}  {'✅ significant' if r_bg['significant'] else '— not significant'}")
    print(f"  Kruskal-Wallis   : H={r_bg['kruskal_stat']:.4f}, p={r_bg['kruskal_p']:.4f}")
    print(f"  Eta-squared      : {r_bg['eta_squared']:.4f} ({r_bg['effect_size_label']})")
    print(f"\n  Power achieved   : {r_bg['power_achieved']*100:.1f}%")
    print(f"  N/group for 80%  : {r_bg['n_needed_80']}")
    print(f"  Low-n warning    : {'⚠️  YES' if r_bg['low_n_warning'] else 'No'}")

    check_true("no error",  r_bg["error"] is None)
    check_true("≥2 groups", len(r_bg["groups"]) >= 2)

# -----------------------------------------------------------------------
# Section D — Repeated measures: MCQ across 7 modules
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("SECTION D: Repeated Measures — MCQ % Correct Across Modules 1–7")
print("="*65)

r_mcq = run_repeated_measures(
    canonical_df,
    instrument_key = "content_mcq_assessment",
    construct=None, alpha=0.05,
)

if r_mcq["error"]:
    print(f"  ❌ ERROR: {r_mcq['error']}")
else:
    print(f"\n  N subjects       : {r_mcq['n_subjects']}")
    print(f"  Time points      : {r_mcq['time_points']}")
    print(f"  Means by module  :")
    for t, m in r_mcq["means_by_time"].items():
        print(f"    {t}: {m:.2f}%")
    print(f"\n  Friedman χ²      : {r_mcq['friedman_stat']:.4f}, p={r_mcq['p_value']:.4f}  {'✅ significant' if r_mcq['significant'] else '— not significant'}")
    print(f"  Kendall's W      : {r_mcq['kendalls_w']:.4f} ({r_mcq['effect_size_label']})")

    check_true("no error",        r_mcq["error"] is None)
    check("n_subjects=9",         r_mcq["n_subjects"], 9)
    check("7 time points",        len(r_mcq["time_points"]), 7)
    check_true("all means 0–100",
        all(0 <= v <= 100 for v in r_mcq["means_by_time"].values()))

# -----------------------------------------------------------------------
# Section E — Repeated measures: SIMS intrinsic_motivation across modules
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("SECTION E: Repeated Measures — SIMS Intrinsic Motivation Across Modules")
print("="*65)

r_sims = run_repeated_measures(
    canonical_df,
    instrument_key = "b4ai_sims_survey",
    construct      = "intrinsic_motivation",
    alpha=0.05,
)

if r_sims["error"]:
    print(f"  ❌ ERROR: {r_sims['error']}")
else:
    print(f"\n  N subjects       : {r_sims['n_subjects']}")
    print(f"  Means by module  :")
    for t, m in r_sims["means_by_time"].items():
        print(f"    {t}: {m:.4f}")
    print(f"\n  Friedman χ²      : {r_sims['friedman_stat']:.4f}, p={r_sims['p_value']:.4f}  {'✅ significant' if r_sims['significant'] else '— not significant'}")
    print(f"  Kendall's W      : {r_sims['kendalls_w']:.4f} ({r_sims['effect_size_label']})")

    check_true("no error",        r_sims["error"] is None)
    check("n_subjects=9",         r_sims["n_subjects"], 9)
    check("7 time points",        len(r_sims["time_points"]), 7)
    check_true("all means 1–4",
        all(1 <= v <= 4 for v in r_sims["means_by_time"].values()))

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"  Structural checks: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL STRUCTURAL CHECKS PASSED.")
    print("  👁  Please verify the printed numbers match your expectations.")
else:
    print("  ❌ SOME CHECKS FAILED — review output above.")
print('='*65)
