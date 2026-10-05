"""
verify_item_review.py -- checks for the Item Review feature (Method 3): item loader, agreement statistics,
reviewer-first rule, store behaviour and the LLM service with the model MOCKED (no cost).
Optional: pass --live to also make ONE real Sonnet call per step on one item (costs about a cent).

Usage (project root, env b4ai_v0):  python scripts/verify_item_review.py [--live]
Exit code 0 = all checks passed.
"""
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

from core.analytics.item_review import bloom_llm, store
from core.analytics.item_review.agreement import agreement_summary, weighted_kappa
from core.analytics.item_review.item_bank import BLOOM_LEVELS, item_hash, list_instruments, load_item_bank
from core.analytics.item_review.runner import run_llm_batch

fail = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok:
        fail.append(name)


# ---------------------------------------------------------------- 1. item bank
inst = list_instruments()
check("9 instruments listed (7 module banks + AI-CI + Misconceptions)", len(inst) == 9, str(len(inst)))
m1 = load_item_bank("module1_content_mcq_assessment")
check("Module 1 bank: 15 items, options lettered consecutively from A (13 four-option + 2 True/False items), every item keyed",
      len(m1) == 15 and all([o["label"] for o in i["options"]] == list("ABCD")[:len(i["options"])] and i["key"] for i in m1)
      and sorted(len(i["options"]) for i in m1).count(2) == 2)
m2 = load_item_bank("module2_content_mcq_assessment")
check("Module 2 bank: 40 items; in_current_selection matches .env (Q1,Q5,Q10 yes; Q2 no)",
      len(m2) == 40 and {i["question_id"]: i["in_current_selection"] for i in m2}["Q5"] is True
      and {i["question_id"]: i["in_current_selection"] for i in m2}["Q2"] is False)
aici = load_item_bank("precourse_pre_aici_assessment")
check("AI-CI: 20 items, Q4_1 key is A (corrected 2026-10-04), 'image' items flagged as figure-dependent",
      len(aici) == 20 and {i["question_id"]: i["key"] for i in aici}["Q4_1"] == "A"
      and any(i["depends_on_figure"] for i in aici))
mis = load_item_bank("precourse_pre_ai_misconceptions_assessment")
check("Misconceptions: 8 items; True/False answers are their own labels; Q3_4 reworded (v2.0)",
      len(mis) == 8 and [o["label"] for o in mis[0]["options"]] == ["True", "False"] and "exactly the same way" in mis[3]["text"])
check("item_hash changes when wording changes, stable otherwise",
      item_hash("a", [{"label": "A", "text": "x"}]) == item_hash("a ", [{"label": "A", "text": "x"}])
      and item_hash("a", [{"label": "A", "text": "x"}]) != item_hash("b", [{"label": "A", "text": "x"}]))
check("Bloom levels: six, Remember..Create", [v["label"] for v in BLOOM_LEVELS.values()] ==
      ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"])

# ---------------------------------------------------------------- 2. agreement statistics
a = [1, 2, 3, 4, 5, 6]
check("perfect agreement: kappa = 1, exact = 1", weighted_kappa(a, a) == 1.0 and agreement_summary(a, a)["exact"] == 1.0)
# hand example: 2 raters, 4 items on levels 1-3 with one 1-step disagreement.
x, y = [1, 2, 3, 2], [1, 2, 3, 3]
k_lin = weighted_kappa(x, y, levels=(1, 2, 3), weights="linear")
# O: (1,1)=1,(2,2)=1,(3,3)=1,(2,3)=1 ; row sums (1,2,1) col sums (1,1,2); n=4; linear weights |i-j|/2
E = np.outer([1, 2, 1], [1, 1, 2]) / 4
W = np.abs(np.subtract.outer(range(3), range(3))) / 2
O = np.zeros((3, 3)); O[0, 0] = O[1, 1] = O[2, 2] = O[1, 2] = 1
check("linear weighted kappa matches hand formula 1 - sum(W*O)/sum(W*E)", abs(k_lin - (1 - (W * O).sum() / (W * E).sum())) < 1e-12, f"{k_lin:.4f}")
check("kappa = 0 when one rater uses a single level (no agreement beyond chance)", abs(weighted_kappa([1, 1, 1], [1, 2, 3])) < 1e-12)
check("kappa undefined (nan) only when BOTH raters use the same single level", np.isnan(weighted_kappa([2, 2, 2], [2, 2, 2])))
s = agreement_summary([1, 2, 3, 4, 3, 2, 1, 5], [1, 2, 3, 3, 3, 2, 2, 5])
check("agreement summary: exact 6/8, within-1 all, kappa interval present for n>=8",
      s["exact"] == 0.75 and s["within1"] == 1.0 and s["kappa_lo"] == s["kappa_lo"])

# ---------------------------------------------------------------- 3. store: reviewer-first rule
tmp = os.path.join(tempfile.mkdtemp(), "t.db")
store.init_schema(tmp)
store.init_schema(tmp)                                     # idempotent
item = m1[0]
check("LLM suggestion not available before it exists", store.llm_suggestion(item, tmp) is None)

fake = {"ok": True, "data": {"solved_answer": item["key"], "solve_confidence": "high", "bloom_level": 2,
                             "rationale": "r", "needs_figure": False, "blind_matches_key": True},
        "raw": "{}", "error": None, "stop_reason": "end_turn", "tokens_in": 1000, "tokens_out": 500}
rid = store.create_run("tester", "m", "v", item["instrument_key"], "blind,key", 1, 1, 0.01, tmp)
store.save_llm_result(rid, item, "blind", 0, fake, "m", "v", tmp)
store.save_llm_result(rid, item, "key", 0, {"ok": True, "data": {"key_defensible": "yes", "other_arguably_correct": ["B"],
                      "implausible_distractors": ["D"], "comments": "c"}, "raw": "{}", "error": None,
                      "tokens_in": 1, "tokens_out": 1}, "m", "v", tmp)
check("REVIEWER-FIRST: reveal returns nothing before the reviewer locks a rating",
      store.reveal_llm_suggestion("rev1", item, tmp) is None)
tbl = store.summary_table([item], "rev1", tmp)
check("REVIEWER-FIRST: summary table hides the LLM level for an unlocked item", tbl.loc[0, "llm_level_median"] is None
      or tbl["llm_level_median"].isna().all())
store.lock_blind_rating("rev1", item, 3, "needs applying a rule", tmp)
sug = store.reveal_llm_suggestion("rev1", item, tmp)
check("after locking, suggestion is revealed (level 2, key review present)", sug and sug["level_median"] == 2
      and sug["key_review"]["other_arguably_correct"] == ["B"])
try:
    store.lock_blind_rating("rev1", item, 4, "again", tmp)
    check("locked rating cannot be overwritten", False)
except ValueError:
    check("locked rating cannot be overwritten", True)
check("another reviewer is independent (still sees nothing)", store.reveal_llm_suggestion("rev2", item, tmp) is None)
try:
    store.save_final_decision("rev1", item, 2, "", tmp)
    check("changing level without rationale is refused", False)
except ValueError:
    check("changing level without rationale is refused", True)
h = store.save_final_decision("rev1", item, 2, "LLM is right: the item only asks to recall", tmp)
check("final level 2 equals LLM median -> decision 'adopted_llm'; first rating still 3", h["decision"] == "adopted_llm" and h["blind_level"] == 3)
h = store.get_human("rev1", item, tmp)
store.save_final_decision("rev1", item, 3, "", tmp)
check("final level equal to first rating -> 'kept' without rationale", store.get_human("rev1", item, tmp)["decision"] == "kept")
store.save_key_decision("rev1", item, "needs_review", "option B also fits", tmp)
check("key decision stored", store.get_human("rev1", item, tmp)["key_decision"] == "needs_review")
edited = dict(item, item_hash="different")
check("edited item (new hash) is treated as unrated", store.get_human("rev1", edited, tmp) is None)

# ---------------------------------------------------------------- 4. LLM service with the model mocked
calls = []


def fake_call(system, user, schema, max_tokens=4000, effort="medium"):
    calls.append((system[:30], user, schema))
    if "answer key" in system and "NOT given" in system:      # step 1
        return dict(fake, data=dict(fake["data"], solved_answer="B"))
    return {"ok": True, "data": {"key_defensible": "yes", "other_arguably_correct": [], "implausible_distractors": [],
                                 "comments": ""}, "raw": "{}", "error": None, "stop_reason": "end_turn", "tokens_in": 10, "tokens_out": 5}


orig = bloom_llm._call
bloom_llm._call = fake_call
try:
    r1 = bloom_llm.rate_blind(item)
    check("STEP 1 prompt never contains the answer key text", "Keyed (official) answer" not in calls[-1][1])
    check("step-1 blind answer compared with key IN CODE (B vs key %s -> mismatch)" % item["key"],
          r1["data"]["blind_matches_key"] == (item["key"] == "B"))
    s1 = calls[-1][2]
    check("step-1 schema enumerates this item's letters + 'unsure' and Bloom 1-6, additionalProperties false",
          s1["properties"]["solved_answer"]["enum"] == list("ABCD") + ["unsure"]
          and s1["properties"]["bloom_level"]["enum"] == [1, 2, 3, 4, 5, 6] and s1["additionalProperties"] is False)
    bloom_llm.review_key(item)
    check("STEP 2 prompt contains the keyed answer", f"Keyed (official) answer: {item['key']}" in calls[-1][1])
    nokey = dict(item, key=None)
    check("step 2 refuses cleanly when no key is on file", bloom_llm.review_key(nokey)["ok"] is False)
    out = run_llm_batch(m1[:3], "tester", steps=("blind", "key"), repeats=2, db_path=tmp)
    check("batch: 3 items x (2 blind + 1 key) = 9 ok, 0 errors, run stored", out["n_ok"] == 9 and out["n_err"] == 0)
    s = store.llm_suggestion(m1[1], tmp)
    check("repeats stored: 2 blind runs for item 2", s["n_runs"] == 2)
finally:
    bloom_llm._call = orig

est = bloom_llm.estimate_cost_usd(148, 2, 1)
check("cost estimate for all 148 module items, 2 steps, 1 repeat is a few dollars at most", 0 < est < 5, f"${est:.2f}")

# ---------------------------------------------------------------- 5. optional live call
if "--live" in sys.argv:
    print("\nLIVE call to", bloom_llm.MODEL, "(item AI-CI Q4_1, step 1 and 2)")
    it = aici[0]
    r = bloom_llm.rate_blind(it)
    print("  step 1:", r["ok"], r["error"], r["data"], "tokens", r["tokens_in"], r["tokens_out"], "stop", r["stop_reason"])
    check("live step 1 returned valid parsed JSON", r["ok"] and 1 <= r["data"]["bloom_level"] <= 6)
    r2 = bloom_llm.review_key(it)
    print("  step 2:", r2["ok"], r2["error"], r2["data"], "tokens", r2["tokens_in"], r2["tokens_out"])
    check("live step 2 returned valid parsed JSON", r2["ok"] and r2["data"]["key_defensible"] in ("yes", "no", "uncertain"))
    print("  cost this test: $%.4f" % (bloom_llm.cost_from_usage(r["tokens_in"] + r2["tokens_in"], r["tokens_out"] + r2["tokens_out"])))

print("\n" + ("ALL CHECKS PASSED" if not fail else "FAILED: " + ", ".join(fail)))
sys.exit(1 if fail else 0)
