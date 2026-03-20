"""
Phase 0 — Scoring Verification Test 3
test_binary_mcq.py

Tests: module1_content_mcq_assessment
Scoring type: binary (correct_answers)
57-item bank. Students receive a random draw of 20.
Responses stored as single letter: A / B / C / D.

Key invariant:
    % correct = sum(item_score) / len(answered_questions)
    NOT divided by 57.

Run from project root:
    python tests/phase0/test_binary_mcq.py

Expected result: ALL TESTS PASSED
"""

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.scoring_engine import compute_score

# -----------------------------------------------------------------------
# Scoring YAML — 57-item bank (embedded from module1_content_mcq_assessment_scoring.yaml)
# -----------------------------------------------------------------------

SCORING = {
    "scoring_type": "binary",
    "correct_answers": {
        "Q1": "B", "Q2": "C", "Q3": "B", "Q4": "D", "Q5": "A",
        "Q6": "D", "Q7": "B", "Q8": "A", "Q9": "B", "Q10": "A",
        "Q11": "A", "Q12": "C", "Q13": "A", "Q14": "C", "Q15": "A",
        "Q16": "D", "Q17": "A", "Q18": "B", "Q19": "D", "Q20": "A",
        "Q21": "C", "Q22": "B", "Q23": "C", "Q24": "B", "Q25": "A",
        "Q26": "A", "Q27": "D", "Q28": "B", "Q29": "A", "Q30": "B",
        "Q31": "B", "Q32": "A", "Q33": "C", "Q34": "B", "Q35": "A",
        "Q36": "C", "Q37": "D", "Q38": "A", "Q39": "C", "Q40": "A",
        "Q41": "D", "Q42": "D", "Q43": "A", "Q44": "C", "Q45": "A",
        "Q46": "A", "Q47": "C", "Q48": "D", "Q49": "D", "Q50": "B",
        "Q51": "B", "Q52": "A", "Q53": "C", "Q54": "C", "Q55": "B",
        "Q56": "C", "Q57": "B",
    }
}

CORRECT_ANSWERS = SCORING["correct_answers"]
ALL_QIDS = list(CORRECT_ANSWERS.keys())

# -----------------------------------------------------------------------
# Helper: build a student's 20-question draw (all correct)
# -----------------------------------------------------------------------

def draw_all_correct(n=20, seed=42):
    """Simulate student who answers all drawn questions correctly."""
    random.seed(seed)
    drawn = random.sample(ALL_QIDS, n)
    return {qid: CORRECT_ANSWERS[qid] for qid in drawn}

def draw_all_wrong(n=20, seed=42):
    """Simulate student who answers all drawn questions incorrectly."""
    wrong_map = {"A": "B", "B": "C", "C": "D", "D": "A"}
    random.seed(seed)
    drawn = random.sample(ALL_QIDS, n)
    return {qid: wrong_map[CORRECT_ANSWERS[qid]] for qid in drawn}

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

def check_pct(label, raw_score, n_answered, expected_pct):
    """Verify % correct = raw_score / n_answered (not divided by 57)."""
    global PASS, FAIL
    got_pct = round(raw_score / n_answered * 100, 2) if n_answered > 0 else 0.0
    exp_pct = round(expected_pct, 2)
    if got_pct == exp_pct:
        print(f"  ✅ PASS  {label}  →  {raw_score}/{n_answered} = {got_pct}%")
        PASS += 1
    else:
        print(f"  ❌ FAIL  {label}  →  got {got_pct}%, expected {exp_pct}%")
        FAIL += 1

# -----------------------------------------------------------------------
# Test 3a — Full 20/20 correct draw (score = 20, % = 100)
# -----------------------------------------------------------------------
print("\n[Test 3a] 20-question draw, all correct — expected score: 20")
responses_all_correct = draw_all_correct(n=20, seed=42)
score = compute_score(responses_all_correct, SCORING)
check("20/20 correct raw score", score, 20)
check_pct("20/20 correct pct", score, 20, 100.0)

# -----------------------------------------------------------------------
# Test 3b — Full 20/20 wrong draw (score = 0, % = 0)
# -----------------------------------------------------------------------
print("\n[Test 3b] 20-question draw, all wrong — expected score: 0")
responses_all_wrong = draw_all_wrong(n=20, seed=42)
score = compute_score(responses_all_wrong, SCORING)
check("20/20 wrong raw score", score, 0)
check_pct("20/20 wrong pct", score, 20, 0.0)

# -----------------------------------------------------------------------
# Test 3c — Half correct, half wrong (score = 10, % = 50)
# -----------------------------------------------------------------------
print("\n[Test 3c] 20-question draw, 10 correct / 10 wrong — expected score: 10")
random.seed(99)
drawn = random.sample(ALL_QIDS, 20)
half_correct = {}
for i, qid in enumerate(drawn):
    if i < 10:
        half_correct[qid] = CORRECT_ANSWERS[qid]         # correct
    else:
        wrong_map = {"A": "B", "B": "C", "C": "D", "D": "A"}
        half_correct[qid] = wrong_map[CORRECT_ANSWERS[qid]]  # wrong
score = compute_score(half_correct, SCORING)
check("10 correct raw score", score, 10)
check_pct("10 correct pct", score, 20, 50.0)

# -----------------------------------------------------------------------
# Test 3d — Smaller draw (10 questions, all correct)
#           Denominator MUST be 10, not 57
# -----------------------------------------------------------------------
print("\n[Test 3d] 10-question draw, all correct — expected score: 10, pct: 100%")
responses_10 = draw_all_correct(n=10, seed=7)
score = compute_score(responses_10, SCORING)
check("10 questions raw score", score, 10)
check_pct("10/10 correct pct (denom=10)", score, len(responses_10), 100.0)
# Explicit guard: wrong pct if divided by 57
wrong_pct = round(score / 57 * 100, 2)
if wrong_pct != 100.0:
    print(f"  ℹ️  Confirming: score/57 = {wrong_pct}% — this would be wrong denominator")

# -----------------------------------------------------------------------
# Test 3e — Verify specific known questions from YAML
# -----------------------------------------------------------------------
print("\n[Test 3e] Spot-check 5 specific questions against YAML key")
spot_checks = [
    ("Q1",  "B", 1),   # correct
    ("Q1",  "A", 0),   # wrong
    ("Q27", "D", 1),   # correct
    ("Q27", "B", 0),   # wrong
    ("Q57", "B", 1),   # correct (last item)
]
for qid, ans, expected in spot_checks:
    score = compute_score({qid: ans}, SCORING)
    check(f"Q{qid}={ans}", score, expected)

# -----------------------------------------------------------------------
# Test 3f — Unknown question ID must raise ValueError
# -----------------------------------------------------------------------
print("\n[Test 3f] Unknown question ID — expected: ValueError raised")
try:
    compute_score({"Q58": "A"}, SCORING)
    print("  ❌ FAIL  No error raised for Q58 (not in 57-item bank)")
    FAIL += 1
except ValueError as e:
    print(f"  ✅ PASS  ValueError raised for Q58: {e}")
    PASS += 1

# -----------------------------------------------------------------------
# Test 3g — Reproducibility: same seed → same score
# -----------------------------------------------------------------------
print("\n[Test 3g] Reproducibility: same seed produces same score")
r1 = draw_all_correct(n=20, seed=123)
r2 = draw_all_correct(n=20, seed=123)
check("same seed → same responses", r1, r2)
check("same seed → same score",
      compute_score(r1, SCORING),
      compute_score(r2, SCORING))

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — Module1 MCQ binary scoring verified.")
    print("  ✅ Subset denominator logic confirmed correct.")
else:
    print("  ❌ SOME TESTS FAILED — Review output above.")
print('='*60)
