"""
*** PRE-FIX DIAGNOSTIC ONLY (kept for the record). Valid ONLY while b4ai_sims_scoring.yaml still reverse-scores
*** Q4_1-Q4_4 and Q5_1-Q5_3. Since the 2026-10-08 fix (reverse_questions: []) the stored scores are RAW, so this script's
*** extra "(5 - stored)" flip is itself wrong and its "AS CODED" / "CORRECTED" labels are SWAPPED in meaning (the AS CODED
*** line now holds the correct values). To re-run the paper numbers after the fix use scripts/rerun_lak27_rai_analyses.py;
*** to check direction use scripts/verify_cpi_sims_direction.py.

diagnose_sims_direction_real_data.py -- READ-ONLY impact check of the SIMS double-reversal defect on the pilot data
(see scripts/verify_cpi_sims_direction.py and PROJECT_STATUS.md, 2026-10-08).

Compares, on responses.db (a temp copy is used), the RAI and the CPI_process SIMS component AS CODED versus CORRECTED,
and re-fits the LAK27 primary mixed model (BuffaloPrep + Cohort 2D, Modules 1-6) with each version of RAI.
CORRECTED = subtract the RAW external/amotivation means: RAI = 2*Intr + Ident - (5 - Ext_stored) - 2*(5 - Amo_stored);
CPI SIMS component without the second flip. Nothing is written.   Usage:  python scripts/diagnose_sims_direction_real_data.py
"""
import os,sys,warnings,shutil,tempfile,sqlite3
warnings.filterwarnings("ignore"); ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0,os.getcwd())
import numpy as np, pandas as pd
from scipy import stats
import core.analytics.datasets.canonical_loader as cl
from core.analytics.cpi.cpi_engine import compute_cpi_process
from core.analytics.correlational.correlation_engine import *
from core.analytics.correlational.correlation_engine import DEFAULT_SCCCES_COMPOSITE_MAP as CM
from core.analytics.descriptive.score_aggregator import compute_construct_means
tmp=os.path.join(tempfile.mkdtemp(),"r.db"); shutil.copyfile("responses.db",tmp)
df,_,_=cl.load_canonical_data(tmp)
U=sqlite3.connect("file:users.db?mode=ro&immutable=1",uri=True); coh={u:c for u,c in U.execute("select username,cohort_id from users")}
df=df[df.user_id.map(coh).notna()].copy()
# ---- per student x module construct means (as the code sees them: ALREADY reversed for external/amotivation)
sims=df[df.instrument_key.str.endswith("b4ai_sims_survey")]
cm=compute_construct_means(sims, instrument_keys=list(sims.instrument_key.unique()))
w=cm.pivot_table(index=["user_id","instrument_key"],columns="construct",values="mean_score").dropna()
I,ID,E,A=[w[c] for c in ("intrinsic_motivation","identified_regulation","external_regulation","amotivation")]
rai_code = 2*I+ID-E-2*A                       # what compute_rai does today (E, A already reversed)
rai_ok   = 2*I+ID-(5-E)-2*(5-A)               # correct: subtract the RAW external/amotivation
print("student x module SIMS rows:",len(w))
print("RAI as coded vs corrected:  Spearman rho = %.3f ; mean coded %.2f, corrected %.2f"%(stats.spearmanr(rai_code,rai_ok)[0],rai_code.mean(),rai_ok.mean()))
print("   coded RAI vs intrinsic mean: rho = %.3f   | corrected RAI vs intrinsic mean: rho = %.3f"%(stats.spearmanr(rai_code,I)[0],stats.spearmanr(rai_ok,I)[0]))
print("   coded RAI vs (reversed) amotivation mean: rho = %.3f | corrected RAI vs (reversed) amotivation: rho = %.3f"%(stats.spearmanr(rai_code,A)[0],stats.spearmanr(rai_ok,A)[0]))
# CPI_process SIMS part per student
p=compute_cpi_process(df)
sims_code=p.set_index("user_id").sims_mean.astype(float)
norm=lambda x:(x-1)/3
mine=pd.DataFrame({"I":I,"ID":ID,"E":E,"A":A}).groupby(level=0).mean()
sims_ok=((norm(mine.I)+norm(mine.ID)+norm(mine.E)+norm(mine.A))/4)       # no second flip
j=pd.concat([sims_code.rename("code"),sims_ok.rename("ok")],axis=1).dropna()
print("\nCPI_process SIMS component per student (n=%d): coded vs corrected Spearman rho = %.3f ; mean coded %.3f vs corrected %.3f"%(len(j),stats.spearmanr(j.code,j.ok)[0],j.code.mean(),j.ok.mean()))
print("   students whose SIMS part moves by more than 0.10:",int((abs(j.code-j.ok)>0.10).sum()),"of",len(j))
# ---- the LAK27 primary model with corrected RAI
SCK,SI="b4ai_sccces_survey","b4ai_sims_survey"
sub=df[df.user_id.map(coh).isin(["BuffaloPrep","amherstyouthandrec2D"])]
fr={}
for pn,cons in CM.items():
    x=compute_composite_scores(sub,SCK,{pn:cons}); fr[pn]=x[x.composite==pn]
r_code=compute_rai(sub,SI)
r_ok=r_code.copy()
cm2=compute_construct_means(sub[sub.instrument_key.str.endswith(SI)],instrument_keys=list(sub[sub.instrument_key.str.endswith(SI)].instrument_key.unique()))
pv=cm2.pivot_table(index=["user_id","module_id"],columns="construct",values="mean_score")
pv["rai"]=2*pv.intrinsic_motivation+pv.identified_regulation-(5-pv.external_regulation)-2*(5-pv.amotivation)
pv.loc[pv[["intrinsic_motivation","identified_regulation","external_regulation","amotivation"]].isna().any(axis=1),"rai"]=np.nan
r_ok=pv.reset_index()[["user_id","module_id","rai"]]
def fit(rai_frame,label):
    f=dict(fr); f["RAI"]=rai_frame
    L=build_person_module_dataset(sub,f); L=L[L.module_num<=6]; preds=list(f)
    r=run_mixed_model(person_mean_center(L,preds),"pct_correct",within_cols=preds,between_cols=preds)
    b=r["blocks"]["M3b_random_slope_reml"]["params"]
    print("  %-10s RAI within  est=%7.3f p=%.3f | RAI between est=%7.3f p=%.3f | Attention_Culture between est=%.2f p=%.4f"%(label,b["RAI_within"]["estimate"],b["RAI_within"]["p"],b["RAI_between"]["estimate"],b["RAI_between"]["p"],b["Attention_Culture_between"]["estimate"],b["Attention_Culture_between"]["p"]))
print("\nLAK27 primary model (BuffaloPrep + Cohort 2D, Modules 1-6), random-slope REML:")
fit(r_code,"AS CODED"); fit(r_ok,"CORRECTED")
