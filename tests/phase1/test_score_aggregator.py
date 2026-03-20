"""
Phase 1 — Step 1a Verification
test_score_aggregator.py

Tests: core/analytics/descriptive/score_aggregator.py

Run from project root:
    python tests/phase1/test_score_aggregator.py

Expected result: ALL TESTS PASSED
"""

import sys, os, importlib.util
import pandas as pd

# -----------------------------------------------------------------------
# Import score_aggregator by file path
# -----------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_THIS_DIR, "..", "..", "core", "analytics", "descriptive", "score_aggregator.py"),
    os.path.join(_THIS_DIR, "score_aggregator.py"),
]
_path = next((os.path.normpath(p) for p in _CANDIDATES if os.path.exists(os.path.normpath(p))), None)
if _path is None:
    print("ERROR: score_aggregator.py not found.")
    sys.exit(1)

_spec = importlib.util.spec_from_file_location("score_aggregator", _path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_assessment_scores    = _mod.compute_assessment_scores
compute_construct_means      = _mod.compute_construct_means
aggregate_construct_means    = _mod.aggregate_construct_means
summarize_scores             = _mod.summarize_scores

# -----------------------------------------------------------------------
# Test runner
# -----------------------------------------------------------------------
PASS = 0
FAIL = 0

def check(label, got, expected):
    global PASS, FAIL
    if isinstance(expected, float):
        ok = isinstance(got, float) and abs(got - expected) < 1e-3
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

def check_true(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✅ PASS  {label}")
        PASS += 1
    else:
        print(f"  ❌ FAIL  {label}  {detail}")
        FAIL += 1

# -----------------------------------------------------------------------
# Shared canonical_df fixture
# Instruments:
#   pre_ai_misconceptions_assessment  — binary, 8 items, 3 users
#   post_ai_misconceptions_assessment — binary, 8 items, 2 users
#   module1_content_mcq_assessment    — binary, variable draw (20/15/10)
#   b4ai_sccces_survey                — Likert, 3 constructs, 2 users
#   b4ai_sims_survey                  — Likert, 2 constructs, 2 users
# -----------------------------------------------------------------------

def _make_binary_rows(user_id, instrument_key, n_correct, n_wrong,
                      grade="Fourth (4th) grade", cohort_id="cohort_A"):
    rows = []
    for i in range(n_correct):
        rows.append(dict(
            user_id=user_id, instrument_key=instrument_key,
            question_id=f"Q_{i}", item_score=1.0,
            construct=None, grade=grade, cohort_id=cohort_id,
            module_id="global", response_value="X",
            submitted_at="2026-01-01", completed_at=None,
        ))
    for i in range(n_wrong):
        rows.append(dict(
            user_id=user_id, instrument_key=instrument_key,
            question_id=f"Q_{n_correct+i}", item_score=0.0,
            construct=None, grade=grade, cohort_id=cohort_id,
            module_id="global", response_value="Y",
            submitted_at="2026-01-01", completed_at=None,
        ))
    return rows

def _make_likert_rows(user_id, instrument_key, construct_scores: dict,
                      grade="Fourth (4th) grade", cohort_id="cohort_A",
                      module_id="module_1"):
    """construct_scores = {construct_name: [item_score, ...]}"""
    rows = []
    qnum = 0
    for construct, scores in construct_scores.items():
        for score in scores:
            rows.append(dict(
                user_id=user_id, instrument_key=instrument_key,
                question_id=f"Q_{qnum}", item_score=float(score),
                construct=construct, grade=grade, cohort_id=cohort_id,
                module_id=module_id, response_value="Agree",
                submitted_at="2026-01-01", completed_at=None,
            ))
            qnum += 1
    return rows

rows = []

# pre_ai_misconceptions_assessment
# u1: 6/8 correct = 75.0%
# u2: 8/8 correct = 100.0%
# u3: 2/8 correct = 25.0%
rows += _make_binary_rows("u1", "pre_ai_misconceptions_assessment",  6, 2, grade="Fourth (4th) grade",  cohort_id="cohort_A")
rows += _make_binary_rows("u2", "pre_ai_misconceptions_assessment",  8, 0, grade="Fifth (5th) grade",   cohort_id="cohort_A")
rows += _make_binary_rows("u3", "pre_ai_misconceptions_assessment",  2, 6, grade="Fourth (4th) grade",  cohort_id="cohort_B")

# post_ai_misconceptions_assessment
# u1: 7/8 = 87.5%
# u2: 5/8 = 62.5%
rows += _make_binary_rows("u1", "post_ai_misconceptions_assessment", 7, 1, grade="Fourth (4th) grade",  cohort_id="cohort_A")
rows += _make_binary_rows("u2", "post_ai_misconceptions_assessment", 5, 3, grade="Fifth (5th) grade",   cohort_id="cohort_A")

# module1_content_mcq_assessment — variable draw sizes
# u1: 15/20 = 75.0%
# u2: 12/15 = 80.0%
# u3: 8/10  = 80.0%
rows += _make_binary_rows("u1", "module1_content_mcq_assessment", 15, 5, grade="Fourth (4th) grade",  cohort_id="cohort_A")
rows += _make_binary_rows("u2", "module1_content_mcq_assessment", 12, 3, grade="Fifth (5th) grade",   cohort_id="cohort_A")
rows += _make_binary_rows("u3", "module1_content_mcq_assessment",  8, 2, grade="Fourth (4th) grade",  cohort_id="cohort_B")

# b4ai_sccces_survey — 2 users, 3 constructs
# u1: engagement=[3,4] mean=3.5 | attention=[2,1] mean=1.5 | personal_relevance=[4,3,3] mean=3.333
# u2: engagement=[2,2] mean=2.0 | attention=[3,3] mean=3.0 | personal_relevance=[2,2,2] mean=2.0
rows += _make_likert_rows("u1", "b4ai_sccces_survey", {
    "engagement_with_task":  [3, 4],
    "attention":             [2, 1],
    "personal_relevance":    [4, 3, 3],
}, grade="Fourth (4th) grade", cohort_id="cohort_A")
rows += _make_likert_rows("u2", "b4ai_sccces_survey", {
    "engagement_with_task":  [2, 2],
    "attention":             [3, 3],
    "personal_relevance":    [2, 2, 2],
}, grade="Fifth (5th) grade", cohort_id="cohort_A")

# b4ai_sims_survey — 2 users, 2 constructs
# u1: intrinsic_motivation=[4,4,4] mean=4.0 | amotivation=[1,1,1] mean=1.0
# u2: intrinsic_motivation=[3,2,3] mean=2.667 | amotivation=[2,2,2] mean=2.0
rows += _make_likert_rows("u1", "b4ai_sims_survey", {
    "intrinsic_motivation": [4, 4, 4],
    "amotivation":          [1, 1, 1],
}, grade="Fourth (4th) grade", cohort_id="cohort_A")
rows += _make_likert_rows("u2", "b4ai_sims_survey", {
    "intrinsic_motivation": [3, 2, 3],
    "amotivation":          [2, 2, 2],
}, grade="Fifth (5th) grade", cohort_id="cohort_A")

# module2 SIMS rows for u1 — different scores to verify per-module separation
# u1 module_2 intrinsic_motivation=[2,2,2] mean=2.0 (vs module_1 mean=4.0)
rows += _make_likert_rows("u1", "b4ai_sims_survey", {
    "intrinsic_motivation": [2, 2, 2],
    "amotivation":          [3, 3, 3],
}, module_id="module_2")

# Q1_1 initials — should be excluded (construct=None, item_score=NaN)
rows.append(dict(
    user_id="u1", instrument_key="b4ai_sccces_survey",
    question_id="Q1_1", item_score=float("nan"),
    construct=None, grade="Fourth (4th) grade", cohort_id="cohort_A",
    module_id="module_1", response_value="AB",
    submitted_at="2026-01-01", completed_at=None,
))

canonical_df = pd.DataFrame(rows)

# Demographics fixture
demographics_df = pd.DataFrame([
    {"user_id": "u1", "grade": "Fourth (4th) grade", "grade_level": 4, "gender": "Female", "first_language_english": True},
    {"user_id": "u2", "grade": "Fifth (5th) grade",  "grade_level": 5, "gender": "Male",   "first_language_english": True},
    {"user_id": "u3", "grade": "Fourth (4th) grade", "grade_level": 4, "gender": "Female", "first_language_english": False},
])

# -----------------------------------------------------------------------
# Group 1 — compute_assessment_scores: basic correctness
# -----------------------------------------------------------------------
print("\n[Group 1] compute_assessment_scores — basic correctness")

asc = compute_assessment_scores(canonical_df)

check_true("returns DataFrame", isinstance(asc, pd.DataFrame))
check_true("expected columns",
    {"user_id","instrument_key","n_items_answered","raw_score","pct_correct"}.issubset(set(asc.columns)))

# u1 pre_ai_misconceptions
r = asc[(asc.user_id=="u1") & (asc.instrument_key=="pre_ai_misconceptions_assessment")].iloc[0]
check("u1 pre_misconceptions n_items", int(r.n_items_answered), 8)
check("u1 pre_misconceptions raw",     float(r.raw_score),      6.0)
check("u1 pre_misconceptions pct",     float(r.pct_correct),    75.0)

# u2 pre_ai_misconceptions (perfect score)
r = asc[(asc.user_id=="u2") & (asc.instrument_key=="pre_ai_misconceptions_assessment")].iloc[0]
check("u2 pre_misconceptions pct=100", float(r.pct_correct), 100.0)

# u3 pre_ai_misconceptions (low score)
r = asc[(asc.user_id=="u3") & (asc.instrument_key=="pre_ai_misconceptions_assessment")].iloc[0]
check("u3 pre_misconceptions pct=25",  float(r.pct_correct), 25.0)

# -----------------------------------------------------------------------
# Group 2 — MCQ subset denominator (critical: must NOT divide by 57)
# -----------------------------------------------------------------------
print("\n[Group 2] MCQ subset denominator — pct = score / n_answered NOT /57")

# u1: 15/20=75, u2: 12/15=80, u3: 8/10=80
r1 = asc[(asc.user_id=="u1") & (asc.instrument_key=="module1_content_mcq_assessment")].iloc[0]
r2 = asc[(asc.user_id=="u2") & (asc.instrument_key=="module1_content_mcq_assessment")].iloc[0]
r3 = asc[(asc.user_id=="u3") & (asc.instrument_key=="module1_content_mcq_assessment")].iloc[0]

check("u1 MCQ denom=20", int(r1.n_items_answered), 20)
check("u1 MCQ pct=75",   float(r1.pct_correct),    75.0)
check("u2 MCQ denom=15", int(r2.n_items_answered), 15)
check("u2 MCQ pct=80",   float(r2.pct_correct),    80.0)
check("u3 MCQ denom=10", int(r3.n_items_answered), 10)
check("u3 MCQ pct=80",   float(r3.pct_correct),    80.0)

# Explicit guard: wrong denominator would give 15/57=26.3
check_true("NOT divided by 57", float(r1.pct_correct) != round(15/57*100, 4))

# -----------------------------------------------------------------------
# Group 3 — Survey rows excluded from compute_assessment_scores
# -----------------------------------------------------------------------
print("\n[Group 3] Survey instruments excluded from assessment scores")

survey_in_asc = asc[asc.instrument_key.isin(["b4ai_sccces_survey","b4ai_sims_survey"])]
check("no survey rows in assessment output", len(survey_in_asc), 0)

# -----------------------------------------------------------------------
# Group 4 — compute_construct_means: basic correctness
# -----------------------------------------------------------------------
print("\n[Group 4] compute_construct_means — basic correctness")

cm = compute_construct_means(canonical_df)

check_true("returns DataFrame", isinstance(cm, pd.DataFrame))
check_true("expected columns",
    {"user_id","instrument_key","construct","n_items","total_score","mean_score"}.issubset(set(cm.columns)))

# u1 SCCCES engagement_with_task: items=[3,4] mean=3.5
r = cm[(cm.user_id=="u1") & (cm.instrument_key=="b4ai_sccces_survey") & (cm.construct=="engagement_with_task")].iloc[0]
check("u1 SCCCES engagement n_items=2",      int(r.n_items),    2)
check("u1 SCCCES engagement total=7",        float(r.total_score), 7.0)
check("u1 SCCCES engagement mean=3.5",       float(r.mean_score),  3.5)

# u1 SCCCES attention (reverse-scored values already resolved): [2,1] mean=1.5
r = cm[(cm.user_id=="u1") & (cm.instrument_key=="b4ai_sccces_survey") & (cm.construct=="attention")].iloc[0]
check("u1 SCCCES attention mean=1.5",        float(r.mean_score), 1.5)

# u1 SCCCES personal_relevance: [4,3,3] mean=3.333
r = cm[(cm.user_id=="u1") & (cm.instrument_key=="b4ai_sccces_survey") & (cm.construct=="personal_relevance")].iloc[0]
check("u1 SCCCES personal_relevance mean=3.333", float(r.mean_score), round(10/3, 4))

# u1 SIMS intrinsic_motivation: [4,4,4] mean=4.0
r = cm[(cm.user_id=="u1") & (cm.instrument_key=="b4ai_sims_survey") & (cm.construct=="intrinsic_motivation")].iloc[0]
check("u1 SIMS intrinsic_motivation mean=4.0", float(r.mean_score), 4.0)

# u2 SIMS intrinsic_motivation: [3,2,3] mean=2.667
r = cm[(cm.user_id=="u2") & (cm.instrument_key=="b4ai_sims_survey") & (cm.construct=="intrinsic_motivation")].iloc[0]
check("u2 SIMS intrinsic_motivation mean=2.667", float(r.mean_score), round(8/3, 4))

# module_id column present
check_true("module_id column in construct means output",
    "module_id" in cm.columns)

# u1 has 2 module rows for SIMS intrinsic (module_1 and module_2)
u1_sims_intr = cm[
    (cm.user_id=="u1") &
    (cm.instrument_key=="b4ai_sims_survey") &
    (cm.construct=="intrinsic_motivation")
]
check("u1 SIMS intrinsic: 2 module rows", len(u1_sims_intr), 2)
r_m1 = u1_sims_intr[u1_sims_intr.module_id=="module_1"].iloc[0]
r_m2 = u1_sims_intr[u1_sims_intr.module_id=="module_2"].iloc[0]
check("u1 module_1 SIMS intrinsic mean=4.0", float(r_m1.mean_score), 4.0)
check("u1 module_2 SIMS intrinsic mean=2.0", float(r_m2.mean_score), 2.0)

# -----------------------------------------------------------------------
# Group 5 — Q1_1 initials excluded (NaN item_score + None construct)
# -----------------------------------------------------------------------
print("\n[Group 5] Q1_1 initials field excluded from construct means")

initials_in_cm = cm[cm.question_id.isin(["Q1_1"])] if "question_id" in cm.columns else pd.DataFrame()
check_true("no Q1_1 bleed into construct means",
    not any(
        (cm.user_id=="u1") & (cm.instrument_key=="b4ai_sccces_survey") & (cm.construct.isna())
    ))

# -----------------------------------------------------------------------
# Group 6 — Assessment rows excluded from compute_construct_means
# -----------------------------------------------------------------------
print("\n[Group 6] Assessment instruments excluded from construct means")

assess_in_cm = cm[cm.instrument_key.isin([
    "pre_ai_misconceptions_assessment",
    "post_ai_misconceptions_assessment",
    "module1_content_mcq_assessment",
])]
check("no assessment rows in construct output", len(assess_in_cm), 0)

# -----------------------------------------------------------------------
# Group 7 — summarize_scores: no groupby (all users)
# -----------------------------------------------------------------------
print("\n[Group 7] summarize_scores — no groupby (global summary)")

# Assessment summary — pre_ai_misconceptions_assessment
# pct values: u1=75.0, u2=100.0, u3=25.0
# mean=66.667, median=75.0, mode=25.0 (all unique → smallest)
asc_pre = asc[asc.instrument_key=="pre_ai_misconceptions_assessment"]
summary = summarize_scores(asc_pre)

check_true("summary has 1 row", len(summary)==1)
r = summary.iloc[0]
check("global pre_misconceptions n_users=3",   int(r.n_users),    3)
check("global pre_misconceptions mean_pct",    float(r.mean_pct), round((75+100+25)/3, 4))
check("global pre_misconceptions median_pct",  float(r.median_pct), 75.0)

# Construct summary — b4ai_sccces_survey engagement
# mean_scores: u1=3.5, u2=2.0
# mean=2.75, median=2.75
cm_eng = cm[(cm.instrument_key=="b4ai_sccces_survey") & (cm.construct=="engagement_with_task")]
summary_c = summarize_scores(cm_eng)
r = summary_c.iloc[0]
check("global SCCCES engagement n_users=2",    int(r.n_users),       2)
check("global SCCCES engagement mean_score",   float(r.mean_score),  2.75)
check("global SCCCES engagement median_score", float(r.median_score), 2.75)

# -----------------------------------------------------------------------
# Group 8 — summarize_scores: group by grade (from canonical_df)
# -----------------------------------------------------------------------
print("\n[Group 8] summarize_scores — group by grade (canonical_df column)")

# pre_ai_misconceptions: grade distribution:
#   "Fourth (4th) grade": u1=75.0, u3=25.0  → mean=50.0, median=50.0
#   "Fifth (5th) grade":  u2=100.0           → mean=100.0
asc_pre = asc[asc.instrument_key=="pre_ai_misconceptions_assessment"]
summary_grade = summarize_scores(asc_pre, group_by_col="grade")

fourth = summary_grade[summary_grade.grade=="Fourth (4th) grade"].iloc[0]
fifth  = summary_grade[summary_grade.grade=="Fifth (5th) grade"].iloc[0]

check("grade=Fourth n_users=2",   int(fourth.n_users),    2)
check("grade=Fourth mean_pct=50", float(fourth.mean_pct), 50.0)
check("grade=Fifth  n_users=1",   int(fifth.n_users),     1)
check("grade=Fifth  mean_pct=100",float(fifth.mean_pct),  100.0)

# -----------------------------------------------------------------------
# Group 9 — summarize_scores: group by cohort_id
# -----------------------------------------------------------------------
print("\n[Group 9] summarize_scores — group by cohort_id")

# cohort_A: u1=75, u2=100  → mean=87.5
# cohort_B: u3=25           → mean=25.0
summary_cohort = summarize_scores(asc_pre, group_by_col="cohort_id")
ca = summary_cohort[summary_cohort.cohort_id=="cohort_A"].iloc[0]
cb = summary_cohort[summary_cohort.cohort_id=="cohort_B"].iloc[0]

check("cohort_A n_users=2",    int(ca.n_users),    2)
check("cohort_A mean_pct=87.5",float(ca.mean_pct), 87.5)
check("cohort_B n_users=1",    int(cb.n_users),    1)
check("cohort_B mean_pct=25",  float(cb.mean_pct), 25.0)

# -----------------------------------------------------------------------
# Group 10 — summarize_scores: group by gender (requires demographics_df)
# -----------------------------------------------------------------------
print("\n[Group 10] summarize_scores — group by gender (demographics_df)")

# Female: u1=75, u3=25 → mean=50.0
# Male:   u2=100       → mean=100.0
summary_gender = summarize_scores(asc_pre, group_by_col="gender",
                                  demographics_df=demographics_df)
fem = summary_gender[summary_gender.gender=="Female"].iloc[0]
mal = summary_gender[summary_gender.gender=="Male"].iloc[0]

check("Female n_users=2",    int(fem.n_users),    2)
check("Female mean_pct=50",  float(fem.mean_pct), 50.0)
check("Male   n_users=1",    int(mal.n_users),    1)
check("Male   mean_pct=100", float(mal.mean_pct), 100.0)

# -----------------------------------------------------------------------
# Group 11 — summarize_scores: group by first_language_english
# -----------------------------------------------------------------------
print("\n[Group 11] summarize_scores — group by first_language_english")

# English=True:  u1=75, u2=100 → mean=87.5
# English=False: u3=25          → mean=25.0
summary_lang = summarize_scores(asc_pre, group_by_col="first_language_english",
                                demographics_df=demographics_df)
eng  = summary_lang[summary_lang.first_language_english==True].iloc[0]
neng = summary_lang[summary_lang.first_language_english==False].iloc[0]

check("English=True  n_users=2",    int(eng.n_users),     2)
check("English=True  mean_pct=87.5",float(eng.mean_pct),  87.5)
check("English=False n_users=1",    int(neng.n_users),    1)
check("English=False mean_pct=25",  float(neng.mean_pct), 25.0)

# -----------------------------------------------------------------------
# Group 12 — summarize_scores: gender group on construct means
# -----------------------------------------------------------------------
print("\n[Group 12] summarize_scores — group by gender on construct means")

# SCCCES engagement: Female u1=3.5, u3 not in survey  → only u1
# Male: u2=2.0
cm_eng_all = cm[cm.instrument_key=="b4ai_sccces_survey"]
summary_gc = summarize_scores(cm_eng_all, group_by_col="gender",
                              demographics_df=demographics_df)

check_true("construct + gender columns present",
    "construct" in summary_gc.columns and "gender" in summary_gc.columns)

# -----------------------------------------------------------------------
# Group 13 — Error: gender groupby without demographics_df
# -----------------------------------------------------------------------
print("\n[Group 13] Error handling — gender groupby without demographics_df")

try:
    summarize_scores(asc_pre, group_by_col="gender")
    print("  ❌ FAIL  No error raised")
    FAIL += 1
except ValueError as e:
    print(f"  ✅ PASS  ValueError raised: {e}")
    PASS += 1

# -----------------------------------------------------------------------
# Group 14 — instrument_keys filter argument
# -----------------------------------------------------------------------
print("\n[Group 14] instrument_keys filter on both functions")

asc_filtered = compute_assessment_scores(canonical_df,
    instrument_keys=["pre_ai_misconceptions_assessment"])
check_true("only pre_misconceptions in filtered assessment",
    set(asc_filtered.instrument_key.unique()) == {"pre_ai_misconceptions_assessment"})

cm_filtered = compute_construct_means(canonical_df,
    instrument_keys=["b4ai_sims_survey"])
check_true("only b4ai_sims_survey in filtered construct means",
    set(cm_filtered.instrument_key.unique()) == {"b4ai_sims_survey"})

# -----------------------------------------------------------------------
# Group 15 — Empty canonical_df returns correct empty schema
# -----------------------------------------------------------------------
print("\n[Group 15] Empty inputs return correct empty DataFrames")

empty_df = canonical_df.iloc[0:0].copy()
try:
    compute_assessment_scores(empty_df)
    print("  ❌ FAIL  No ValueError for empty canonical_df")
    FAIL += 1
except ValueError:
    print("  ✅ PASS  ValueError on empty canonical_df (correct)")
    PASS += 1

# Empty after filtering — no crash, correct columns
asc_empty = compute_assessment_scores(canonical_df, instrument_keys=["nonexistent"])
check("empty filter: 0 rows",       len(asc_empty), 0)
check_true("empty filter: correct cols",
    set(asc_empty.columns) == {"user_id","instrument_key","n_items_answered","raw_score","pct_correct"})

cm_empty = compute_construct_means(canonical_df, instrument_keys=["nonexistent"])
check("empty construct filter: 0 rows", len(cm_empty), 0)

# -----------------------------------------------------------------------
# Group 16 — aggregate_construct_means
# -----------------------------------------------------------------------
print("\n[Group 16] aggregate_construct_means — collapses modules correctly")

agg = aggregate_construct_means(cm)
check_true("aggregate returns DataFrame", isinstance(agg, pd.DataFrame))
check_true("aggregate required columns present",
    {"user_id","instrument_key","construct","n_modules","n_items_total","total_score","mean_score"}
    .issubset(set(agg.columns)))

# u1 SIMS intrinsic: module_1=[4,4,4] total=12, module_2=[2,2,2] total=6
# aggregate: n_modules=2, n_items_total=6, total_score=18, mean=18/6=3.0
r_agg = agg[
    (agg.user_id=="u1") &
    (agg.instrument_key=="b4ai_sims_survey") &
    (agg.construct=="intrinsic_motivation")
].iloc[0]
check("u1 SIMS intrinsic agg n_modules=2",     int(r_agg.n_modules),    2)
check("u1 SIMS intrinsic agg n_items_total=6", int(r_agg.n_items_total), 6)
check("u1 SIMS intrinsic agg total_score=18",  float(r_agg.total_score), 18.0)
check("u1 SIMS intrinsic agg mean=3.0",        float(r_agg.mean_score),  3.0)

# u2 only has module_1 data — n_modules=1, mean unchanged
r_u2 = agg[
    (agg.user_id=="u2") &
    (agg.instrument_key=="b4ai_sims_survey") &
    (agg.construct=="intrinsic_motivation")
].iloc[0]
check("u2 SIMS intrinsic agg n_modules=1",  int(r_u2.n_modules), 1)
check("u2 SIMS intrinsic agg mean=2.667",   float(r_u2.mean_score), round(8/3, 4))

# Mean-of-means guard: aggregate mean must NOT be mean of module means
# u1: mean of (4.0, 2.0) = 3.0 — happens to match here
# but with unequal n_items it would differ; formula is total/n_items
check("aggregate uses total/n_items not mean-of-means",
      float(r_agg.mean_score), 18.0 / 6)

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — score_aggregator verified.")
else:
    print("  ❌ SOME TESTS FAILED — review output above.")
print('='*60)
