"""Unit checks for the post-hoc step of run_repeated_measures (core/analytics/inferential/inferential_tests.py).

Spec: after a significant PRIMARY omnibus test (RM ANOVA if it is the recommended one, else Friedman) with 3+ occasions,
all k(k-1)/2 pairs are compared on the same complete cases (paired t after RM ANOVA, Wilcoxon signed-rank after
Friedman) with Bonferroni adjustment, adjusted p = min(1, p x m). New keys only: result["posthoc"], result["posthoc_note"].
Run: set PYTHONUTF8=1, then python scripts/verify_posthoc.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
from scipy import stats

import core.analytics.inferential.inferential_tests as it

fail = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok:
        fail.append(name)


def canonical(pct):
    """pct: DataFrame users x modules of % correct (whole numbers) -> canonical long frame, 100 items per module."""
    rows = []
    for u, r in pct.iterrows():
        for m, v in r.items():
            k = int(round(v))
            for q in range(100):
                rows.append({"user_id": u, "module_id": m, "instrument_key": "module_content_mcq_assessment",
                             "question_id": f"Q{q}", "item_score": 1.0 if q < k else 0.0})
    return pd.DataFrame(rows)


rng = np.random.default_rng(11)
n = 40
mods = ["module_1", "module_2", "module_3", "module_4"]

# ---- 1. significant, roughly normal -> RM ANOVA primary -> paired t post-hoc
base = rng.normal(50, 12, n)
norm = pd.DataFrame({m: np.clip(np.round(base + 7 * i + rng.normal(0, 6, n)), 0, 100)
                     for i, m in enumerate(mods)}, index=[f"s{i}" for i in range(n)])
r1 = it.run_repeated_measures(canonical(norm), "content_mcq_assessment")
check("1a. no error, existing keys still present",
      r1.get("error") is None and all(k in r1 for k in ("friedman_stat", "p_value", "kendalls_w", "rm_anova", "recommended_test")))
check("1a2. recommendation is RM-ANOVA for this data", str(r1.get("recommended_test", "")).startswith("RM-ANOVA"),
      str(r1.get("recommended_test")))
ph = r1.get("posthoc")
check("1b. significant omnibus gives a post-hoc table", isinstance(ph, dict) and len(ph.get("pairs", [])) == 6,
      str(r1.get("posthoc_note")))
if ph:
    p0 = ph["pairs"][0]
    a, b = norm[mods[0]], norm[mods[1]]
    t, p = stats.ttest_rel(b, a)
    check("1c. first pair is the paired t-test on the same complete cases (statistic, raw p)",
          abs(p0["statistic"] - t) < 1e-3 and abs(p0["p_raw"] - p) < 1e-5, f"{p0['statistic']} vs {t:.4f}")
    dz = float((b - a).mean() / (b - a).std(ddof=1))
    check("1d. effect size is Cohen's d_z of (later - earlier)", abs(p0["effect_size"] - dz) < 1e-3, f"{p0['effect_size']} vs {dz:.4f}")
    check("1e. adjusted p = min(1, p x 6) for every pair",
          all(abs(x["p_adj"] - min(1.0, x["p_raw"] * 6)) < 1e-5 for x in ph["pairs"]))
    check("1f. 6 comparisons, adjusted alpha .05/6", ph["n_comparisons"] == 6 and abs(ph["alpha_adjusted"] - 0.05 / 6) < 1e-9)
    check("1g. test after RM ANOVA is the paired t-test", ph["post_test"].lower().startswith("paired t"), ph["post_test"])
    check("1h. n per pair = complete cases", all(x["n"] == len(norm) for x in ph["pairs"]))
    check("1i. effect size name is stated", "d_z" in ph["effect_size_name"])

# ---- 2. significant, non-normal -> Friedman primary -> Wilcoxon post-hoc
skew = pd.DataFrame({m: np.clip(np.round(rng.exponential(12, n) + 15 * i), 0, 100)
                     for i, m in enumerate(mods)}, index=[f"s{i}" for i in range(n)])
r2 = it.run_repeated_measures(canonical(skew), "content_mcq_assessment")
check("2a. recommendation is Friedman for this skewed data", str(r2.get("recommended_test", "")).startswith("Friedman"),
      str(r2.get("recommended_test")))
ph2 = r2.get("posthoc")
check("2b. significant Friedman gives a post-hoc table", isinstance(ph2, dict) and len(ph2.get("pairs", [])) == 6,
      str(r2.get("posthoc_note")))
if ph2:
    q = ph2["pairs"][0]
    w = stats.wilcoxon(skew[mods[1]], skew[mods[0]])
    check("2c. first pair is Wilcoxon signed-rank (raw p equals scipy)", abs(q["p_raw"] - w.pvalue) < 1e-5, f"{q['p_raw']} vs {w.pvalue:.6f}")
    d = (skew[mods[1]] - skew[mods[0]]).values
    d = d[d != 0]
    rk = stats.rankdata(np.abs(d))
    rrb = (rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum()
    check("2d. effect size is the rank-biserial correlation of (later - earlier)", abs(q["effect_size"] - rrb) < 1e-3,
          f"{q['effect_size']} vs {rrb:.4f}")
    check("2e. adjusted p = min(1, p x 6)", all(abs(x["p_adj"] - min(1.0, x["p_raw"] * 6)) < 1e-5 for x in ph2["pairs"]))
    check("2f. test after Friedman is Wilcoxon", ph2["post_test"].lower().startswith("wilcoxon"), ph2["post_test"])
    check("2g. effect size name is stated", "rank-biserial" in ph2["effect_size_name"])

# ---- 3. not significant -> no post-hoc, with a reason
flat = pd.DataFrame({m: np.clip(np.round(rng.normal(50, 12, n)), 0, 100) for m in mods},
                    index=[f"s{i}" for i in range(n)])
r3 = it.run_repeated_measures(canonical(flat), "content_mcq_assessment")
check("3a. non-significant omnibus gives no post-hoc table", r3.get("posthoc") is None)
check("3b. ...and says why", "not significant" in str(r3.get("posthoc_note", "")).lower(), str(r3.get("posthoc_note")))

# ---- 4. helper edge cases
helper = getattr(it, "_posthoc_pairs", None)
check("4a. helper _posthoc_pairs exists", helper is not None)
if helper is not None:
    z = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [1, 2, 3, 4, 5, 6], "c": [2, 3, 4, 5, 6, 8]})
    out = helper(z, "wilcoxon", 0.05)
    ab = next(x for x in out["pairs"] if x["time_a"] == "a" and x["time_b"] == "b")
    check("4b. all-zero differences (identical occasions) -> p = 1, no crash, no effect size",
          ab["p_raw"] == 1.0 and ab["p_adj"] == 1.0 and ab["effect_size"] is None, str(ab))
    big = helper(pd.DataFrame(rng.normal(size=(12, 6))), "paired_t", 0.05)
    check("4c. 6 occasions -> 15 pairs, adjusted alpha .05/15 = .0033",
          big["n_comparisons"] == 15 and abs(big["alpha_adjusted"] - 0.05 / 15) < 1e-12)
    check("4d. adjusted p never exceeds 1", all(x["p_adj"] <= 1.0 for x in big["pairs"]))
    ties = pd.DataFrame({"a": [1, 1, 2, 2, 3, 3, 4, 5], "b": [2, 2, 3, 3, 4, 4, 5, 6], "c": [3, 3, 4, 4, 5, 5, 6, 7]})
    tt = helper(ties, "wilcoxon", 0.05)
    check("4e. tied differences do not crash and give finite p", all(0 <= x["p_raw"] <= 1 for x in tt["pairs"]))

# ---- 5. display: table + caption + help under the Friedman / RM-ANOVA result; note when absent; report sections
from streamlit.testing.v1 import AppTest


def app_shown():
    import streamlit_app.dashboards.teacher_dashboard as td
    ph = {"post_test": "Wilcoxon signed-rank test", "correction": "Bonferroni", "n_comparisons": 3, "alpha": 0.05,
          "alpha_adjusted": 0.05 / 3, "effect_size_name": "matched-pairs rank-biserial correlation",
          "primary_test": "Friedman test", "primary_p": 0.01,
          "pairs": [{"time_a": "module_1", "time_b": "module_2", "n": 30, "statistic": 12.0, "p_raw": 0.004,
                     "p_adj": 0.012, "effect_size": 0.61, "significant": True}]}
    td._render_posthoc({"posthoc": ph, "posthoc_note": "x"})
    td._render_posthoc({"posthoc": None, "posthoc_note": "Primary omnibus test (Friedman test, p = 0.5000) is not significant"})


at = AppTest.from_function(app_shown, default_timeout=60).run()
cap = " ".join(str(c.value) for c in at.caption)
check("5a. post-hoc card renders without an exception", not at.exception, str([e.value for e in at.exception]))
check("5b. table has the adjusted-p column", any("p (Bonferroni-adjusted)" in list(d.value.columns) for d in at.dataframe))
check("5c. caption names the test, the correction and the number of comparisons",
      "Wilcoxon" in cap and "Bonferroni" in cap and "3 pairs" in cap, cap[:200])
import streamlit_app.dashboards.teacher_dashboard as td
h = td._STAT_HELP["posthoc"]
check("5e. help text covers why / Bonferroni (.05/15 = .0033) / how to read / small samples / Hahs-Vaughn",
      all(w in h for w in ("Why this step exists", "Bonferroni", ".0033", "How to read", "Small samples", "Hahs-Vaughn")))
check("5f. when absent, the reason is shown", "not significant" in cap)
secs = td._posthoc_report_sections({"posthoc": {"post_test": "Paired t-test", "correction": "Bonferroni", "n_comparisons": 1,
                                                "alpha": 0.05, "alpha_adjusted": 0.05, "effect_size_name": "Cohen's d_z",
                                                "primary_test": "RM-ANOVA", "primary_p": 0.01,
                                                "pairs": [{"time_a": "a", "time_b": "b", "n": 9, "statistic": 3.0, "p_raw": 0.01,
                                                           "p_adj": 0.01, "effect_size": 1.0, "significant": True}]}}, "X")
check("5g. report gets a post-hoc section when present, none when absent",
      len(secs) == 1 and secs[0]["table"] is not None and td._posthoc_report_sections({"posthoc": None}, "X") == [])
pdf = td._build_pdf(secs, "t")
check("5h. the post-hoc section builds into a PDF", pdf[:4] == b"%PDF")


print("\n%d check(s) failed" % len(fail) if fail else "\nAll checks passed")
sys.exit(1 if fail else 0)
