"""
rerun_lak27_rai_analyses.py -- prints every RAI-dependent result of the LAK27 paper (read-only; temp copy of the DB).

  A  sample sizes and RAI summary          B  Table 3: model comparison (AIC/BIC/LRT), primary sample
  C  Table 4: coefficients (M3b REML) + fixed-slope M3   D  Table 5: Cohort 1 supplementary model (all 7 modules)
  E  Table 6: repeated-measures correlations with Benjamini-Hochberg FDR

Primary sample = BuffaloPrep + NWACC Cohort 2D, Modules 1-6, listwise deletion AFTER person-mean centring (dashboard order).
Used 2026-10-08 to re-run the paper after the SIMS double-reversal fix (see PROJECT_STATUS.md). Run before and after
a scoring change and compare.      Usage:  python scripts/rerun_lak27_rai_analyses.py
"""
import os
import shutil
import sqlite3
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

import core.analytics.datasets.canonical_loader as cl
from core.analytics.correlational.correlation_engine import *  # noqa: F401,F403
from core.analytics.correlational.correlation_engine import DEFAULT_SCCCES_COMPOSITE_MAP as CM

tmp = os.path.join(tempfile.mkdtemp(), "r.db")
shutil.copyfile(os.path.join(ROOT, "responses.db"), tmp)
df, _, _ = cl.load_canonical_data(tmp)
U = sqlite3.connect("file:" + os.path.join(ROOT, "users.db").replace("\\", "/") + "?mode=ro&immutable=1", uri=True)
coh = {u: c for u, c in U.execute("select username,cohort_id from users")}
SCK, SI = "b4ai_sccces_survey", "b4ai_sims_survey"


def frames(sub):
    fr = {}
    for p, cons in CM.items():
        x = compute_composite_scores(sub, SCK, {p: cons})
        fr[p] = x[x.composite == p]
    fr["RAI"] = compute_rai(sub, SI)
    return fr


def fmt_params(b, keep=None):
    out = []
    for t, p in b["params"].items():
        if keep and t not in keep:
            continue
        z = p.get("z")
        pv = p.get("p")
        out.append(f"    {t:44s} est={p['estimate']:8.3f} se={p['se']:7.3f} z={(z if z is not None else float('nan')):6.2f} p={(pv if pv is not None else float('nan')):.4f}")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------- primary
sub = df[df.user_id.map(coh).isin(["BuffaloPrep", "amherstyouthandrec2D"])]
fr = frames(sub)
preds = list(fr)
L = build_person_module_dataset(sub, fr)
L = L[L.module_num <= 6]
print("A. PRIMARY SAMPLE (BuffaloPrep + Cohort 2D, Modules 1-6)")
rai = fr["RAI"].dropna(subset=["rai"]).rai
print(f"   student-module rows {len(L)}, students {L.user_id.nunique()}, complete rows {len(L.dropna())}, students complete {L.dropna().user_id.nunique()}")
print(f"   RAI summary (student x module): n={len(rai)} mean={rai.mean():.2f} SD={rai.std():.2f} min={rai.min():.2f} max={rai.max():.2f}")
r = run_mixed_model(person_mean_center(L, preds), "pct_correct", within_cols=preds, between_cols=preds)

print("\nB. TABLE 3 model comparison (ML)")
for n in ("M0_null", "M1_module", "M2_within", "M3_full", "M3b_random_slope"):
    b = r["blocks"][n]
    print(f"   {n:18s} AIC={b.get('aic'):.2f} BIC={b.get('bic'):.2f}")
for d in r["lrt"]:
    print(f"   LRT {d['from_block']} -> {d['to_block']}: chi2={d['lr_stat']:.2f} p={d['p_value']:.4f}")
print("   best by AIC:", r.get("best_block_by_aic"), "| best by BIC:", r.get("best_block_by_bic"))

print("\nC. TABLE 4 coefficients, M3b random slope, REML")
print(fmt_params(r["blocks"]["M3b_random_slope_reml"]))
print("   fixed-slope M3 (ML), within and between terms:")
print(fmt_params(r["blocks"]["M3_full"], keep=[t for t in r["blocks"]["M3_full"]["params"] if "_within" in t or "_between" in t]))

# ------------------------------------------------------------------------------------------- cohort 1
c1 = df[df.user_id.map(coh) == "amherstyouthandrec1D"]
fr1 = frames(c1)
L1 = build_person_module_dataset(c1, fr1).dropna()
preds1 = list(fr1)
r1 = run_mixed_model(person_mean_center(L1, preds1), "pct_correct", within_cols=preds1, between_cols=preds1)
print(f"\nD. TABLE 5 Cohort 1 supplementary (n students={L1.user_id.nunique()}, obs={len(L1)}); best BIC={r1.get('best_block_by_bic')} best AIC={r1.get('best_block_by_aic')}")
print("   M2 (within block):")
print(fmt_params(r1["blocks"]["M2_within"]))

# ------------------------------------------------------------------------------------------- rm_corr
rm = run_repeated_measures_correlations(L, "pct_correct", preds)
print("\nE. TABLE 6 repeated-measures correlations (>=3 modules), BH-FDR")
for row in rm["fdr"]:
    ci = row["ci95"]
    print(f"   {row['predictor']:34s} r={row['r']:7.3f} df={row['dof']} CI=[{ci[0]:.2f},{ci[1]:.2f}] p_unc={row['p_unc']:.4f} p_FDR={row['p_fdr']:.4f} sig={row['reject_fdr']} n={row['n_subjects']}")
