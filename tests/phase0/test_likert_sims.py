"""
Phase 0 — Scoring Verification Test 5
test_likert_sims.py

Tests: b4ai_sims_survey (module{N}_b4ai_sims_survey in DB)
Scoring type: Likert, 4-point scale
Reverse questions: Q4_1, Q4_2, Q4_3, Q4_4, Q5_1, Q5_2, Q5_3

Constructs (from b4ai_sims_survey.yaml sections, lowercase):
    intrinsic_motivation    → Q2_1, Q2_2, Q2_3       (forward)
    identified_regulation   → Q3_1, Q3_2, Q3_3       (forward)
    external_regulation     → Q4_1, Q4_2, Q4_3, Q4_4 (REVERSED)
    amotivation             → Q5_1, Q5_2, Q5_3       (REVERSED)
    (Q1_1 = text initials, skipped)

Run from project root:
    python tests/phase0/test_likert_sims.py

Expected result: ALL TESTS PASSED
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.scoring_engine import compute_score

# -----------------------------------------------------------------------
# Scoring YAML (embedded from b4ai_sims_scoring.yaml)
# -----------------------------------------------------------------------

SCORING = {
    "scoring_type": "likert",
    "default_scale": {
        "Strongly disagree": 1,
        "Disagree":          2,
        "Agree":             3,
        "Strongly agree":    4,
    },
    "reverse_scale": {
        "Strongly disagree": 4,
        "Disagree":          3,
        "Agree":             2,
        "Strongly agree":    1,
    },
    "reverse_questions": ["Q4_1", "Q4_2", "Q4_3", "Q4_4", "Q5_1", "Q5_2", "Q5_3"],
}

FORWARD_QUESTIONS  = ["Q2_1", "Q2_2", "Q2_3", "Q3_1", "Q3_2", "Q3_3"]       # 6
REVERSE_QUESTIONS  = ["Q4_1", "Q4_2", "Q4_3", "Q4_4", "Q5_1", "Q5_2", "Q5_3"]  # 7
ALL_SCORED         = FORWARD_QUESTIONS + REVERSE_QUESTIONS                    # 13

# -----------------------------------------------------------------------
# Test runner
# -----------------------------------------------------------------------

PASS = 0
FAIL = 0

def check(label, got, expected):
    global PASS, FAIL
    if got == expected:
        print(f"  ✅ PASS  {label}  →  score={got}")
        PASS += 1
    else:
        print(f"  ❌ FAIL  {label}  →  got={got}, expected={expected}")
        FAIL += 1

# -----------------------------------------------------------------------
# Test 5a — All "Strongly agree"
#           6 forward × 4 + 7 reverse × 1 = 31
# -----------------------------------------------------------------------
print("\n[Test 5a] All 'Strongly agree' on all 13 items — expected score: 31")
all_sa = {qid: "Strongly agree" for qid in ALL_SCORED}
check("all Strongly agree", compute_score(all_sa, SCORING), 31)

# -----------------------------------------------------------------------
# Test 5b — All "Strongly disagree"
#           6 forward × 1 + 7 reverse × 4 = 34
# -----------------------------------------------------------------------
print("\n[Test 5b] All 'Strongly disagree' on all 13 items — expected score: 34")
all_sd = {qid: "Strongly disagree" for qid in ALL_SCORED}
check("all Strongly disagree", compute_score(all_sd, SCORING), 34)

# -----------------------------------------------------------------------
# Test 5c — external_regulation construct (all reversed)
#           Q4_1–Q4_4
#           "Strongly agree"    → reverse → 1   (external pressure, scored down)
#           "Strongly disagree" → reverse → 4
# -----------------------------------------------------------------------
print("\n[Test 5c] External regulation (reversed) spot-checks")
check("Q4_1 Strongly agree → 1",    compute_score({"Q4_1": "Strongly agree"}, SCORING),    1)
check("Q4_1 Strongly disagree → 4", compute_score({"Q4_1": "Strongly disagree"}, SCORING), 4)
check("Q4_4 Agree → 2",             compute_score({"Q4_4": "Agree"}, SCORING),             2)
check("Q4_4 Disagree → 3",          compute_score({"Q4_4": "Disagree"}, SCORING),          3)

# -----------------------------------------------------------------------
# Test 5d — amotivation construct (all reversed)
#           Q5_1, Q5_2, Q5_3
# -----------------------------------------------------------------------
print("\n[Test 5d] Amotivation (reversed) spot-checks")
check("Q5_1 Strongly agree → 1",    compute_score({"Q5_1": "Strongly agree"}, SCORING),    1)
check("Q5_3 Strongly disagree → 4", compute_score({"Q5_3": "Strongly disagree"}, SCORING), 4)
check("Q5_2 Agree → 2",             compute_score({"Q5_2": "Agree"}, SCORING),             2)

# -----------------------------------------------------------------------
# Test 5e — intrinsic_motivation construct (forward)
#           Q2_1, Q2_2, Q2_3 — all "Strongly agree" → total = 12, mean = 4.0
# -----------------------------------------------------------------------
print("\n[Test 5e] Intrinsic motivation all Strongly agree — expected total: 12, mean: 4.0")
intrinsic = {"Q2_1": "Strongly agree", "Q2_2": "Strongly agree", "Q2_3": "Strongly agree"}
raw = compute_score(intrinsic, SCORING)
mean = raw / len(intrinsic)
check("intrinsic raw total", raw, 12)
if abs(mean - 4.0) < 1e-9:
    print(f"  ✅ PASS  intrinsic mean  →  {raw}/{len(intrinsic)} = {mean}")
    PASS += 1
else:
    print(f"  ❌ FAIL  intrinsic mean  →  got {mean}, expected 4.0")
    FAIL += 1

# -----------------------------------------------------------------------
# Test 5f — identified_regulation construct (forward)
#           Q3_1=Agree, Q3_2=Disagree, Q3_3=Strongly agree
#           3 + 2 + 4 = 9, mean = 3.0
# -----------------------------------------------------------------------
print("\n[Test 5f] Identified regulation mixed — expected total: 9, mean: 3.0")
identified = {"Q3_1": "Agree", "Q3_2": "Disagree", "Q3_3": "Strongly agree"}
raw = compute_score(identified, SCORING)
mean = raw / len(identified)
check("identified_reg raw total", raw, 9)
if abs(mean - 3.0) < 1e-9:
    print(f"  ✅ PASS  identified mean  →  {raw}/{len(identified)} = {mean}")
    PASS += 1
else:
    print(f"  ❌ FAIL  identified mean  →  got {mean}, expected 3.0")
    FAIL += 1

# -----------------------------------------------------------------------
# Test 5g — Q1_1 text initials field is silently skipped
# -----------------------------------------------------------------------
print("\n[Test 5g] Q1_1 text initials field silently skipped — expected score: 0")
check("Q1_1 initials skipped", compute_score({"Q1_1": "AB"}, SCORING), 0)

# -----------------------------------------------------------------------
# Test 5h — Realistic full response: all forward=Agree (3), all reverse=Agree (→2)
#           6 × 3 + 7 × 2 = 18 + 14 = 32
# -----------------------------------------------------------------------
print("\n[Test 5h] Realistic: all Agree forward + all Agree reverse — expected score: 32")
all_agree = {qid: "Agree" for qid in ALL_SCORED}
check("all Agree total", compute_score(all_agree, SCORING), 32)

# -----------------------------------------------------------------------
# Test 5i — Score boundaries: min and max
#           Max achievable (high self-determined motivation):
#               forward all SA=4, reverse all SD→4  → 6×4 + 7×4 = 52
#           Min achievable (no motivation):
#               forward all SD=1, reverse all SA→1  → 6×1 + 7×1 = 13
# -----------------------------------------------------------------------
print("\n[Test 5i] Score boundary check")
max_responses = {}
for qid in FORWARD_QUESTIONS:
    max_responses[qid] = "Strongly agree"     # forward max = 4
for qid in REVERSE_QUESTIONS:
    max_responses[qid] = "Strongly disagree"  # reverse max = 4 (reversed)
check("max possible score = 52", compute_score(max_responses, SCORING), 52)

min_responses = {}
for qid in FORWARD_QUESTIONS:
    min_responses[qid] = "Strongly disagree"  # forward min = 1
for qid in REVERSE_QUESTIONS:
    min_responses[qid] = "Strongly agree"     # reverse min = 1 (reversed)
check("min possible score = 13", compute_score(min_responses, SCORING), 13)

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — SIMS Likert + reverse scoring verified.")
else:
    print("  ❌ SOME TESTS FAILED — Review output above.")
print('='*60)
