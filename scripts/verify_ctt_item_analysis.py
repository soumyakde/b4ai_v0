"""Hand-computed checks for core/analytics/irt/ctt_item_analysis.py. Run: python scripts/verify_ctt_item_analysis.py"""
import os, sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from core.analytics.irt.ctt_item_analysis import compute_item_analysis, compute_distractor_analysis, kr20, sample_size_note
from core.analytics.irt.reliability_analysis import compute_cronbach_alpha

fail = []
def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok: fail.append(name)

# Hand example: 5 students x 4 items, totals 4,3,2,1,0
M = pd.DataFrame([[1,1,1,1],[1,1,1,0],[1,1,0,0],[1,0,0,0],[0,0,0,0]], columns=["Q1","Q2","Q3","Q4"],
                 index=list("ABCDE"), dtype=float)
import core.analytics.irt.ctt_item_analysis as _c
_c.MIN_N_FOR_R = 3     # the 5-student hand example is below the production minimum of 10
r = compute_item_analysis(M)
it = r["items"].set_index("item")
check("p values .8/.6/.4/.2", np.allclose(it["p"], [.8,.6,.4,.2]))
check("KR-20 = 0.8 (hand: 4/3 * (1 - .8/2))", abs(r["test"]["kr20"] - 0.8) < 1e-12, str(r["test"]["kr20"]))
check("KR-20 equals existing Cronbach alpha function", abs(kr20(M) - compute_cronbach_alpha(M)) < 1e-12)
check("mean total 2, SD sqrt(2.5)", abs(r["test"]["mean_total"] - 2) < 1e-12 and abs(r["test"]["sd_total"] - np.sqrt(2.5)) < 1e-12)
# hand value, Q4: rest scores 3,3,2,1,0 (mean 1.8, var 1.36); cov = 3/5 - .2*1.8 = .24; r = .24/(.4*sqrt(1.36)) = 0.5145
check("Q4 corrected item-rest r = 0.5145 (hand)", abs(it.loc["Q4","r_it"] - 0.24/(0.4*np.sqrt(1.36))) < 1e-9, str(it.loc["Q4","r_it"]))
check("SEM = SD*sqrt(1-KR20)", abs(r["test"]["sem"] - np.sqrt(2.5)*np.sqrt(0.2)) < 1e-12)
# Wilson interval: 8/10 -> known (0.4902, 0.9433)
from core.analytics.irt.ctt_item_analysis import _wilson
lo, hi = _wilson(8, 10)
check("Wilson 95% for 8/10 = (0.490, 0.943)", abs(lo-0.4902) < 5e-4 and abs(hi-0.9433) < 5e-4, f"{lo:.4f},{hi:.4f}")
# missing data: NaN excluded from p, complete cases used for r
M2 = M.copy(); M2.loc["A","Q2"] = np.nan
r2 = compute_item_analysis(M2)
check("NaN handled: Q2 n=4, complete cases=4", r2["items"].set_index("item").loc["Q2","n"] == 4 and r2["n_complete"] == 4)
_c.MIN_N_FOR_R = 10
check("production minimum: r not reported for n<10", np.isnan(compute_item_analysis(M)["items"]["r_it"]).all())
# negative discrimination + below-chance flags (miskeyed item)
rng = np.random.default_rng(1)
ab = rng.normal(size=120)
X = pd.DataFrame({f"Q{i}": (ab + rng.normal(size=120) > 0).astype(float) for i in range(1, 7)})
X["Qbad"] = (ab + rng.normal(size=120) < -0.9).astype(float)     # keyed backwards: high ability -> wrong
rb = compute_item_analysis(X, n_options={"Qbad": 2, "Q1": 2})
fb = rb["items"].set_index("item")
check("miskeyed item flagged NEGATIVE discrimination", "NEGATIVE" in fb.loc["Qbad","flags"], fb.loc["Qbad","flags"])
check("good item not flagged negative", "NEGATIVE" not in fb.loc["Q1","flags"])
check("sample-size note mentions DESCRIPTIVE under 30", "DESCRIPTIVE" in sample_size_note(12) and "DESCRIPTIVE" not in sample_size_note(60))
check("tiny n returns no crash", compute_item_analysis(M.iloc[:2])["error"] is None)
check("empty input gives error string", compute_item_analysis(pd.DataFrame())["error"] is not None)

# Distractor analysis on synthetic canonical rows: 4 options, B never chosen
rows = []
ability = rng.normal(size=100)
for u in range(100):
    for qi in range(1, 5):
        correct = ability[u] + rng.normal() * 0.8 > 0
        if correct: letter = "A"
        else: letter = rng.choice(["C", "D"], p=[0.8, 0.2]) if qi == 1 else rng.choice(["C", "D"])
        rows.append({"user_id": f"u{u}", "instrument_key": "t_mcq", "question_id": f"Q{qi}",
                     "response_value": f"{letter}: text", "item_score": 1.0 if letter == "A" else 0.0})
cd = pd.DataFrame(rows)
da = compute_distractor_analysis(cd, "t_mcq", option_letters={f"Q{i}": list("ABCD") for i in range(1, 5)})
b = da[(da.item == "Q1") & (da.option == "B")].iloc[0]
check("never-chosen option appears with 0% and non-functioning flag", b["share"] == 0 and "non-functioning" in b["flags"], b["flags"])
a = da[(da.item == "Q1") & (da.option == "A")].iloc[0]
check("keyed option inferred as A and its share equals mean item score", a["keyed"] and abs(a["share"] - cd[cd.question_id=="Q1"].item_score.mean()) < 1e-12)
check("keyed option share rises with ability group", a["g5"] > a["g1"], f"g1={a['g1']:.2f} g5={a['g5']:.2f}")
check("shares per item sum to 1", all(abs(g["share"].sum() - 1) < 1e-9 for _, g in da.groupby("item")))
print("\n" + ("ALL CHECKS PASSED" if not fail else "FAILED: " + ", ".join(fail)))
sys.exit(1 if fail else 0)
