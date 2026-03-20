"""
Phase 1 — Step 1a Integration Test
test_score_aggregator_integration.py

Runs score_aggregator against the REAL responses.db.

Known DB state (confirmed 2026-03-15):
    9 users, all 7 modules complete
    module1/2 MCQ: 20 items per student (180 rows / 9 users)
    module3-7 MCQ: 10 items per student (90 rows / 9 users)
    All surveys: 21 scoreable items per student per module
    Pre/post assessments: 8 items (misconceptions), 20 items (AICI)

Usage:
    python tests/phase1/test_score_aggregator_integration.py
    python tests/phase1/test_score_aggregator_integration.py path/to/responses.db

Expected: all structural checks pass + printed tables look correct.
"""

import sys, os, importlib.util, sqlite3
import pandas as pd

# -----------------------------------------------------------------------
# DB path
# -----------------------------------------------------------------------
DB_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "responses.db")
)
if not os.path.exists(DB_PATH):
    print(f"ERROR: responses.db not found at {DB_PATH}")
    sys.exit(1)
print(f"\n  DB: {DB_PATH}")

# -----------------------------------------------------------------------
# Import modules by file path
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

_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

resolver_mod = _load([
    os.path.join(_root, "core", "analytics", "filters", "instrument_key_resolver.py"),
    os.path.join(os.path.dirname(__file__), "instrument_key_resolver.py"),
], "instrument_key_resolver")

agg_mod = _load([
    os.path.join(_root, "core", "analytics", "descriptive", "score_aggregator.py"),
    os.path.join(os.path.dirname(__file__), "score_aggregator.py"),
], "score_aggregator")

demo_mod = _load([
    os.path.join(_root, "core", "analytics", "filters", "demographics_extractor.py"),
    os.path.join(os.path.dirname(__file__), "demographics_extractor.py"),
], "demographics_extractor")

resolve_instrument_key       = resolver_mod.resolve_instrument_key
compute_assessment_scores    = agg_mod.compute_assessment_scores
compute_construct_means      = agg_mod.compute_construct_means
aggregate_construct_means    = agg_mod.aggregate_construct_means
summarize_scores             = agg_mod.summarize_scores
extract_demographics         = demo_mod.extract_demographics

# -----------------------------------------------------------------------
# Scoring dicts (embedded from YAML files — binary instruments only)
# -----------------------------------------------------------------------
SCORING_DICTS = {
    "pre_ai_misconceptions_assessment": {
        "scoring_type": "binary",
        "correct_answers": {
            "Q3_1":"False","Q3_2":"False","Q3_3":"False","Q3_4":"False",
            "Q3_5":"No","Q3_6":"Yes","Q3_7":"No","Q3_8":"No",
        },
    },
    "post_ai_misconceptions_assessment": {
        "scoring_type": "binary",
        "correct_answers": {
            "Q3_1":"False","Q3_2":"False","Q3_3":"False","Q3_4":"False",
            "Q3_5":"No","Q3_6":"Yes","Q3_7":"No","Q3_8":"No",
        },
    },
    "pre_aici_assessment": {
        "scoring_type": "binary",
        "correct_answers": {
            "Q4_1":"B","Q4_2":"A","Q4_3":"B","Q4_4":"A","Q4_5":"A",
            "Q4_6":"B","Q4_7":"D","Q4_8":"A","Q4_9":"B","Q4_10":"B",
            "Q4_11":"A","Q4_12":"B","Q4_13":"A","Q4_14":"B","Q4_15":"A",
            "Q4_16":"B","Q4_17":"A","Q4_18":"A","Q4_19":"A","Q4_20":"A",
        },
    },
    "post_aici_assessment": {
        "scoring_type": "binary",
        "correct_answers": {
            "Q4_1":"B","Q4_2":"A","Q4_3":"B","Q4_4":"A","Q4_5":"A",
            "Q4_6":"B","Q4_7":"D","Q4_8":"A","Q4_9":"B","Q4_10":"B",
            "Q4_11":"A","Q4_12":"B","Q4_13":"A","Q4_14":"B","Q4_15":"A",
            "Q4_16":"B","Q4_17":"A","Q4_18":"A","Q4_19":"A","Q4_20":"A",
        },
    },
}
# Module MCQ scoring dicts — add correct_answers per module as YAMLs arrive
# For now module1 is confirmed; others assumed same structure
MCQ_MODULE1_CORRECT = {
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
    SCORING_DICTS[f"module{n}_content_mcq_assessment"] = {
        "scoring_type": "binary",
        "correct_answers": MCQ_MODULE1_CORRECT,
    }

SURVEY_SCORING = {
    "b4ai_sccces_survey": {
        "scoring_type": "likert",
        "default_scale": {
            "Strongly disagree":1,"Disagree":2,"Agree":3,"Strongly agree":4
        },
        "reverse_scale": {
            "Strongly disagree":4,"Disagree":3,"Agree":2,"Strongly agree":1
        },
        "reverse_questions": ["Q9_1","Q9_2","Q10_1","Q10_2"],
    },
    "b4ai_sims_survey": {
        "scoring_type": "likert",
        "default_scale": {
            "Strongly disagree":1,"Disagree":2,"Agree":3,"Strongly agree":4
        },
        "reverse_scale": {
            "Strongly disagree":4,"Disagree":3,"Agree":2,"Strongly agree":1
        },
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

# -----------------------------------------------------------------------
# Build canonical_df from real DB
# -----------------------------------------------------------------------
print("\n[Step 1] Loading responses from real DB...")

conn = sqlite3.connect(DB_PATH)
raw = pd.read_sql_query(
    "SELECT user_id, instrument_name, question_id, response_value, submitted_at FROM responses",
    conn
)
conn.close()
print(f"  → {len(raw)} total rows, {raw['user_id'].nunique()} users")

# Resolve instrument keys
raw["instrument_key"] = raw["instrument_name"].apply(resolve_instrument_key)

# Apply item-level scoring
def score_row(row):
    ikey = row["instrument_key"]
    qid  = row["question_id"]
    val  = row["response_value"]
    scoring = SCORING_DICTS.get(ikey) or SURVEY_SCORING.get(ikey)
    if scoring is None:
        return float("nan")
    stype = scoring.get("scoring_type")
    if stype == "binary":
        correct = scoring["correct_answers"].get(qid)
        if correct is None:
            return float("nan")
        # Normalize: "A: Artificial Intelligence" -> "A"
        # This matches dataset_builder._apply_scoring() normalize_response()
        normalized = str(val).strip()
        if ":" in normalized:
            normalized = normalized.split(":")[0].strip()
        return 1.0 if normalized == correct else 0.0
    elif stype == "likert":
        rev_qs = set(scoring.get("reverse_questions", []))
        scale  = scoring["reverse_scale"] if qid in rev_qs else scoring["default_scale"]
        return float(scale.get(str(val).strip(), float("nan")))
    return float("nan")

# Attach construct labels
def get_construct(row):
    ikey = row["instrument_key"]
    qid  = row["question_id"]
    if ikey == "b4ai_sccces_survey":
        return SCCCES_CONSTRUCTS.get(qid)
    if ikey == "b4ai_sims_survey":
        return SIMS_CONSTRUCTS.get(qid)
    return None

print("  Scoring rows (may take a moment)...")
raw["item_score"] = raw.apply(score_row, axis=1)
raw["construct"]  = raw.apply(get_construct, axis=1)
raw["completed_at"] = None

# Derive module_id from instrument_name:
# module{N}_* -> module_N  |  precourse/postcourse/global -> global
def derive_module_id(instrument_name: str) -> str:
    import re
    m = re.match(r"^(module\d+)_", instrument_name)
    if m:
        raw_mod = m.group(1)            # e.g. "module1"
        # normalize_module_id: module1 -> module_1
        num = raw_mod.replace("module", "")
        return f"module_{num}"
    return "global"

raw["module_id"] = raw["instrument_name"].apply(derive_module_id)

canonical_df = pd.DataFrame({
    "user_id":        raw["user_id"],
    "instrument_key": raw["instrument_key"],
    "question_id":    raw["question_id"],
    "response_value": raw["response_value"],
    "item_score":     raw["item_score"],
    "construct":      raw["construct"],
    "grade":          None,
    "cohort_id":      None,
    "module_id":      raw["module_id"],
    "submitted_at":   raw["submitted_at"],
    "completed_at":   None,
})

print(f"  → canonical_df: {len(canonical_df)} rows")

demographics_df = extract_demographics(DB_PATH)
print(f"  → demographics_df: {len(demographics_df)} users")

# -----------------------------------------------------------------------
# Test runner
# -----------------------------------------------------------------------
PASS = FAIL = 0
def check(label, got, expected):
    global PASS, FAIL
    if isinstance(expected, float):
        ok = isinstance(got, (int, float)) and abs(float(got) - expected) < 0.01
    else:
        ok = (got == expected)
    if ok:
        print(f"  ✅ PASS  {label}")
        PASS += 1
    else:
        print(f"  ❌ FAIL  {label}  got={got!r}  expected={expected!r}")
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
# Group A — compute_assessment_scores structural invariants
# -----------------------------------------------------------------------
print("\n[Group A] Assessment scores — structural invariants")

asc = compute_assessment_scores(canonical_df)
print(f"  → {len(asc)} rows returned")

check("correct columns present",
      {"user_id","instrument_key","n_items_answered","raw_score","pct_correct"}.issubset(set(asc.columns)),
      True)

EXPECTED_ASSESSMENT_KEYS = {
    "pre_ai_misconceptions_assessment",
    "post_ai_misconceptions_assessment",
    "pre_aici_assessment",
    "post_aici_assessment",
    "module1_content_mcq_assessment",
    "module2_content_mcq_assessment",
    "module3_content_mcq_assessment",
    "module4_content_mcq_assessment",
    "module5_content_mcq_assessment",
    "module6_content_mcq_assessment",
    "module7_content_mcq_assessment",
}
found_keys = set(asc["instrument_key"].unique())
check_true("all 11 assessment instruments present",
    EXPECTED_ASSESSMENT_KEYS.issubset(found_keys),
    f"missing: {EXPECTED_ASSESSMENT_KEYS - found_keys}")

check_true("no survey instruments in assessment output",
    not {"b4ai_sccces_survey","b4ai_sims_survey"}.intersection(found_keys),
    f"found: {found_keys}")

check_true("9 users per assessment instrument",
    all(
        asc[asc.instrument_key==k]["user_id"].nunique() == 9
        for k in EXPECTED_ASSESSMENT_KEYS
    ), "some instrument has != 9 users")

check_true("pct_correct always 0–100",
    asc["pct_correct"].between(0, 100).all(),
    f"out of range: {asc[~asc['pct_correct'].between(0,100)][['user_id','instrument_key','pct_correct']].head()}")

check_true("raw_score <= n_items_answered",
    (asc["raw_score"] <= asc["n_items_answered"]).all(),
    "raw_score exceeds n_items_answered for some rows")

# MCQ denominator: modules 1-2 must be 20, modules 3-7 must be 10
for mod_n, expected_n in [(1,20),(2,20),(3,10),(4,10),(5,10),(6,10),(7,10)]:
    key = f"module{mod_n}_content_mcq_assessment"
    actual = asc[asc.instrument_key==key]["n_items_answered"].unique()
    check_true(f"module{mod_n} MCQ denom={expected_n} for all students",
        len(actual)==1 and actual[0]==expected_n,
        f"got: {actual}")

# -----------------------------------------------------------------------
# Group B — compute_construct_means structural invariants
# -----------------------------------------------------------------------
print("\n[Group B] Construct means — structural invariants")

cm = compute_construct_means(canonical_df)
print(f"  → {len(cm)} rows returned")

check("correct columns present",
      {"user_id","instrument_key","construct","n_items","total_score","mean_score"}.issubset(set(cm.columns)),
      True)

EXPECTED_SURVEY_KEYS = {"b4ai_sccces_survey","b4ai_sims_survey"}
found_survey_keys = set(cm["instrument_key"].unique())
check_true("both survey instruments present",
    EXPECTED_SURVEY_KEYS == found_survey_keys,
    f"got: {found_survey_keys}")

SCCCES_CONSTRUCTS_EXPECTED = {
    "engagement_with_task","effort_and_persistence","experience_of_flow",
    "coherency_of_messaging","plausibility_of_messaging","credibility_of_messaging",
    "comprehensibility_of_messaging","attention","culture","personal_relevance",
}
SIMS_CONSTRUCTS_EXPECTED = {
    "intrinsic_motivation","identified_regulation","external_regulation","amotivation",
}
sccces_found = set(cm[cm.instrument_key=="b4ai_sccces_survey"]["construct"].unique())
sims_found   = set(cm[cm.instrument_key=="b4ai_sims_survey"]["construct"].unique())
check_true("all 10 SCCCES constructs present",
    SCCCES_CONSTRUCTS_EXPECTED == sccces_found,
    f"missing: {SCCCES_CONSTRUCTS_EXPECTED - sccces_found}")
check_true("all 4 SIMS constructs present",
    SIMS_CONSTRUCTS_EXPECTED == sims_found,
    f"missing: {SIMS_CONSTRUCTS_EXPECTED - sims_found}")

check_true("module_id column present in construct means",
    "module_id" in cm.columns)
check_true("7 module rows per user per survey construct (SIMS intrinsic)",
    cm[
        (cm.instrument_key=="b4ai_sims_survey") &
        (cm.construct=="intrinsic_motivation") &
        (cm.user_id==cm.user_id.iloc[0])
    ]["module_id"].nunique() == 7,
    "expected 7 module_id values per user per construct")
check_true("mean_score in Likert range 1.0–4.0",
    cm["mean_score"].between(1.0, 4.0).all(),
    f"out of range: {cm[~cm['mean_score'].between(1.0,4.0)][['instrument_key','construct','mean_score']].head()}")

check_true("no NaN mean_scores",
    cm["mean_score"].notna().all())

check_true("9 users per survey × construct",
    all(
        cm[(cm.instrument_key==k) & (cm.construct==c)]["user_id"].nunique() == 9
        for k in EXPECTED_SURVEY_KEYS
        for c in (SCCCES_CONSTRUCTS_EXPECTED if k=="b4ai_sccces_survey" else SIMS_CONSTRUCTS_EXPECTED)
    ), "some survey×construct has != 9 users")

# -----------------------------------------------------------------------
# Group C — Print tables for manual visual verification
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("ASSESSMENT SCORES — pre_ai_misconceptions_assessment (verify manually)")
print("="*70)
pre_misc = asc[asc.instrument_key=="pre_ai_misconceptions_assessment"][
    ["user_id","n_items_answered","raw_score","pct_correct"]
].sort_values("user_id")
print(pre_misc.to_string(index=False))

print("\n" + "="*70)
print("ASSESSMENT SCORES — pre_aici_assessment (verify manually)")
print("="*70)
pre_aici = asc[asc.instrument_key=="pre_aici_assessment"][
    ["user_id","n_items_answered","raw_score","pct_correct"]
].sort_values("user_id")
print(pre_aici.to_string(index=False))

print("\n" + "="*70)
print("MCQ SCORES — module1_content_mcq_assessment (verify manually)")
print("="*70)
mcq1 = asc[asc.instrument_key=="module1_content_mcq_assessment"][
    ["user_id","n_items_answered","raw_score","pct_correct"]
].sort_values("user_id")
print(mcq1.to_string(index=False))

print("\n" + "="*70)
print("CONSTRUCT MEANS — b4ai_sims_survey, intrinsic_motivation, PER MODULE (verify manually)")
print("="*70)
sims_intr_pm = cm[
    (cm.instrument_key=="b4ai_sims_survey") & (cm.construct=="intrinsic_motivation")
]
cols = ["user_id","module_id","n_items","total_score","mean_score"]
cols = [c for c in cols if c in sims_intr_pm.columns]
print(sims_intr_pm[cols].sort_values(["user_id","module_id"]).to_string(index=False))

# Aggregate across all modules
agg_cm = aggregate_construct_means(cm)
print("\n" + "="*70)
print("CONSTRUCT MEANS — b4ai_sims_survey, intrinsic_motivation, AGGREGATE (all modules)")
print("="*70)
sims_intr_agg = agg_cm[
    (agg_cm.instrument_key=="b4ai_sims_survey") & (agg_cm.construct=="intrinsic_motivation")
][["user_id","n_modules","n_items_total","total_score","mean_score"]].sort_values("user_id")
print(sims_intr_agg.to_string(index=False))

# -----------------------------------------------------------------------
# Group D — summarize_scores spot checks
# -----------------------------------------------------------------------
print("\n[Group D] summarize_scores on real data")

pre_misc_asc = asc[asc.instrument_key=="pre_ai_misconceptions_assessment"]
global_summary = summarize_scores(pre_misc_asc)
check("global summary: 1 row",    len(global_summary), 1)
check_true("global summary: n_users=9", int(global_summary.iloc[0]["n_users"])==9)
check_true("global mean_pct is numeric",
    isinstance(global_summary.iloc[0]["mean_pct"], float))
print(f"  ℹ️  pre_misconceptions global mean_pct = {global_summary.iloc[0]['mean_pct']:.2f}%")

gender_summary = summarize_scores(pre_misc_asc, group_by_col="gender",
                                  demographics_df=demographics_df)
check_true("gender groups present",
    "gender" in gender_summary.columns and len(gender_summary) >= 1)
print(f"\n  Gender breakdown — pre_ai_misconceptions_assessment:")
print(gender_summary[["gender","n_users","mean_pct","median_pct"]].to_string(index=False))

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
