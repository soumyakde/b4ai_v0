"""
verify_glmm_item_difficulty.py

Validates core/analytics/irt/glmm_item_difficulty.py (Method 2):
  1. unit checks (long format, degenerate items, sample-size warnings, errors),
  2. a simulation with KNOWN item difficulties at n = 30 / 60 / 100 students:
     correlation with truth, RMSE, 95% interval coverage, for the R (lme4) engine,
     the portable Python engine, and two benchmarks (raw -logit(p), existing girth Rasch),
  3. R-vs-Python agreement on every replication and on real pilot data.

Usage (project root, conda env b4ai_v0):  python scripts/verify_glmm_item_difficulty.py [reps]
R sections are skipped, with a notice, if R + lme4 are not installed.
Exit code 0 = all checks passed.
"""
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

from core.analytics.irt.glmm_item_difficulty import (
    fit_item_difficulty_glmm, r_available, sample_size_warnings, to_long)

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
fail = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok:
        fail.append(name)


# ---------------------------------------------------------------- 1. unit checks
M = pd.DataFrame([[1, 0, np.nan], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0]],
                 columns=["Q1", "Q2", "Q3"], index=list("abcdef"), dtype=float)
L = to_long(M)
check("long format keeps only observed responses (17 of 18)", len(L) == 17 and L["y"].isin([0, 1]).all())
check("fewer than 5 students gives a clear error", fit_item_difficulty_glmm(M.iloc[:4])["error"] is not None)
check("empty input gives error", fit_item_difficulty_glmm(pd.DataFrame())["error"] is not None)
w = sample_size_warnings(25, 6, "random")
check("n=25 and 6 items (random) warns on BOTH Linacre minimum and few items", len(w) == 2, str(len(w)))
check("n=120, 12 items: no Linacre-minimum warning", not any("minimum" in x for x in sample_size_warnings(120, 12, "random")))

# ---------------------------------------------------------------- 2. simulation
HAVE_R = r_available()
print("\nR + lme4 available:", HAVE_R, "| replications per cell:", REPS)
try:
    from girth import rasch_mml
    HAVE_GIRTH = True
except Exception:
    HAVE_GIRTH = False

K = 10
b_true = np.linspace(-1.5, 1.5, K)
rows, agree = [], []
t0 = time.time()
for n in (30, 60, 100):
    for rep in range(REPS):
        rng = np.random.default_rng(1000 * n + rep)
        th = rng.normal(0, 1, n)
        Y = (rng.random((n, K)) < 1 / (1 + np.exp(-(th[:, None] - b_true[None, :])))).astype(float)
        Mx = pd.DataFrame(Y, columns=[f"Q{i + 1}" for i in range(K)])
        if (Y.mean(0) == 0).any() or (Y.mean(0) == 1).any():
            continue
        est = {}
        for eng in (["r"] if HAVE_R else []) + ["python"]:
            for mode in ("random", "fixed"):
                r = fit_item_difficulty_glmm(Mx, engine=eng, item_effects=mode)
                if r["error"]:
                    continue
                it = r["items"]
                est[(eng, mode)] = it["difficulty"].values
                cov = np.mean((it["ci_low"].values <= b_true) & (b_true <= it["ci_high"].values))
                d = it["difficulty"].values
                rows.append({"n": n, "method": f"{eng}-{mode}", "corr": np.corrcoef(d, b_true)[0, 1],
                             "rmse": np.sqrt(np.mean((d - b_true) ** 2)),
                             "rmse_c": np.sqrt(np.mean(((d - d.mean()) - (b_true - b_true.mean())) ** 2)), "cover": cov})
        # benchmarks (centred on the true mean, since their origin is arbitrary)
        p = Y.mean(0)
        naive = -np.log(p / (1 - p))
        naive = naive - naive.mean() + b_true.mean()
        rows.append({"n": n, "method": "raw -logit(p)", "corr": np.corrcoef(naive, b_true)[0, 1],
                     "rmse": np.sqrt(np.mean((naive - b_true) ** 2)),
                     "rmse_c": np.sqrt(np.mean(((naive - naive.mean()) - (b_true - b_true.mean())) ** 2)), "cover": np.nan})
        if HAVE_GIRTH:
            try:
                g = rasch_mml(Y.T.astype(int))["Difficulty"]
                g = np.asarray(g) - np.mean(g) + b_true.mean()
                rows.append({"n": n, "method": "girth Rasch (existing tab)", "corr": np.corrcoef(g, b_true)[0, 1],
                             "rmse": np.sqrt(np.mean((g - b_true) ** 2)),
                             "rmse_c": np.sqrt(np.mean(((g - g.mean()) - (b_true - b_true.mean())) ** 2)), "cover": np.nan})
            except Exception:
                pass
        if ("r", "random") in est and ("python", "random") in est:
            agree.append({"n": n, "corr": np.corrcoef(est[("r", "random")], est[("python", "random")])[0, 1],
                          "maxdiff": np.abs(est[("r", "random")] - est[("python", "random")]).max()})
print("simulation time: %.0fs" % (time.time() - t0))
sim = pd.DataFrame(rows)
summ = sim.groupby(["n", "method"]).agg(corr=("corr", "mean"), rmse_centred=("rmse_c", "mean"), rmse_absolute=("rmse", "mean"), coverage=("cover", "mean"),
                                        reps=("corr", "size")).round(3)
pd.set_option("display.width", 160)
print("\nRecovery of KNOWN item difficulties (true SD of difficulties = 1.0, 10 items):")
print(summ.to_string())

py = sim[sim["method"] == "python-random"].groupby("n")["corr"].mean()
check("Python engine recovers truth at n=100 (mean corr >= 0.97)", py.get(100, 0) >= 0.97, f"{py.get(100, 0):.3f}")
check("Python engine recovers truth at n=30 (mean corr >= 0.85)", py.get(30, 0) >= 0.85, f"{py.get(30, 0):.3f}")
# Item difficulties are only meaningful relative to each other, so ALL methods are compared after centring
# (an earlier version centred only the benchmark on the truth, which was unfair to the models).
rmse = sim.groupby(["n", "method"])["rmse_c"].mean()
check("python-random centred RMSE within 0.03 logits of raw -logit(p) at n=30 (non-inferiority)",
      rmse[(30, "python-random")] <= rmse[(30, "raw -logit(p)")] + 0.03,
      f"{rmse[(30, 'python-random')]:.3f} vs {rmse[(30, 'raw -logit(p)')]:.3f}")
if HAVE_R:
    ag = pd.DataFrame(agree)
    check("R (lme4) and Python engines agree on difficulties (mean corr >= 0.99)", ag["corr"].mean() >= 0.99,
          f"mean corr {ag['corr'].mean():.4f}, worst max|diff| {ag['maxdiff'].max():.3f} logits")
    cov_r = sim[sim["method"] == "r-random"]["cover"].mean()
    cov_p = sim[sim["method"] == "python-random"]["cover"].mean()
    print(f"      interval coverage (nominal 0.95): lme4 {cov_r:.3f} | python {cov_p:.3f}")
    check("lme4 interval coverage within 0.85-1.00 (conditional-SD intervals are approximate)", 0.85 <= cov_r <= 1.0)

# ---------------------------------------------------------------- 3. real pilot data
src = os.path.join(ROOT, "responses.db")
if os.path.exists(src):
    import core.analytics.datasets.canonical_loader as cl
    from core.analytics.irt.irt_runner import build_binary_response_matrix
    from core.analytics.irt.ctt_item_analysis import list_forms, subset_form
    tmp = os.path.join(tempfile.mkdtemp(), "r.db")
    shutil.copyfile(src, tmp)
    df, _, _ = cl.load_canonical_data(tmp)
    U = sqlite3.connect("file:" + os.path.join(ROOT, "users.db").replace("\\", "/") + "?mode=ro&immutable=1", uri=True)
    stu = {u for u, c, r in U.execute("select username,cohort_id,role from users") if r == "student" and c}
    df = df[df.user_id.isin(stu)]
    print("\nReal pilot data: R vs Python agreement")
    for key in ("precourse_pre_aici_assessment", "module1_content_mcq_assessment", "module2_content_mcq_assessment"):
        m, _ = build_binary_response_matrix(df, key)
        sub = subset_form(m, list_forms(m), "Form 1")
        pr = fit_item_difficulty_glmm(sub, engine="python")
        if pr["error"]:
            check(f"{key}: python fit", False, pr["error"])
            continue
        if HAVE_R:
            rr = fit_item_difficulty_glmm(sub, engine="r", item_effects=pr["mode"])
            c = np.corrcoef(rr["items"]["difficulty"], pr["items"]["difficulty"])[0, 1]
            md = np.abs(rr["items"]["difficulty"].values - pr["items"]["difficulty"].values).max()
            check(f"{key} (Form 1, {pr['n_persons']}x{pr['n_items']}, {pr['mode']}): R vs Python corr >= 0.98",
                  c >= 0.98, f"corr {c:.4f}, max|diff| {md:.3f} logits, sigma_person R {rr['sigma_person']:.2f} / Py {pr['sigma_person']:.2f}")
        else:
            check(f"{key}: python fit runs", True, f"{pr['n_persons']}x{pr['n_items']}, mode {pr['mode']}")

print("\n" + ("ALL CHECKS PASSED" if not fail else "FAILED: " + ", ".join(fail)))
sys.exit(1 if fail else 0)
