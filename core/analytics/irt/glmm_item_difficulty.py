"""
core/analytics/irt/glmm_item_difficulty.py

Method 2 of the three item-difficulty options (see PROJECT_STATUS.md, "Item-difficulty
feature plan"): a crossed random-effects logistic model, i.e. the Rasch model written as a
generalized linear mixed model (GLMM):

    logit P(y_pi = 1) = beta0 + u_p + v_i ,   u_p ~ N(0, sigma_person^2)
                                              v_i ~ N(0, sigma_item^2)   ("random items")
or, with items as fixed effects (no shrinkage):
    logit P(y_pi = 1) = beta_i + u_p

Item difficulty b_i = -(beta0 + v_i)  (or -beta_i).  Scale: logits, with the average
person ability fixed at 0; HIGHER b = HARDER.

Why a GLMM and not just the existing Rasch tab: it works on long-format data, so unbalanced /
missing responses need no special handling; with random items the estimates are shrunk toward
the mean (stabilising small samples) and each item gets a standard error.

Two engines
-----------
* "r"      R lme4::glmer (Laplace approximation) through Rscript. Reference engine.
           Only available where R + lme4 are installed (the researcher's machine). The
           production Docker image does NOT contain R.
* "python" statsmodels BinomialBayesMixedGLM (variational Bayes). Portable; runs on Railway.
           It is an approximation (VB posterior means/SDs; known to understate uncertainty),
           so it is validated against lme4 in scripts/verify_glmm_item_difficulty.py.
* "auto"   R if available, else Python. The engine used is always reported.

References
----------
De Boeck, P., Bakker, M., Zwitser, R., Nivard, M., Hofman, A., Tuerlinckx, F., & Partchev, I.
    (2011). The estimation of item response models with the lmer function from the lme4
    package in R. Journal of Statistical Software, 39(12), 1-28.
Doran, H., Bates, D., Bliese, P., & Dowling, M. (2007). Estimating the multilevel Rasch model:
    With the lme4 package. Journal of Statistical Software, 20(2), 1-18.
Linacre, J. M. (1994). Sample size and item calibration stability. Rasch Measurement
    Transactions, 7(4), 328. https://www.rasch.org/rmt/rmt74m.htm  (sample-size table)
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Linacre (1994) table, dichotomous items, "size for most purposes":
#   +/-1 logit 95%: 30 ; +/-1 logit 99%: 50 ; +/-1/2 logit 95%: 100 ; 99%: 150 ; high stakes: 250
LINACRE_TIERS = [(30, "minimum for item difficulties stable within ±1 logit (95%)"),
                 (50, "conservative for ±1 logit (99%)"),
                 (100, "needed for ±½ logit (95%)"),
                 (150, "needed for ±½ logit (99%)"),
                 (250, "high-stakes use")]
FEW_ITEMS = 10          # below this, a random-item variance is poorly estimated


# ----------------------------------------------------------------------------- R helpers
def find_rscript() -> Optional[str]:
    env = os.environ.get("B4AI_RSCRIPT")
    if env and os.path.exists(env):
        return env
    cands = sorted(glob.glob(r"C:/Program Files/R/R-*/bin/x64/Rscript.exe"), reverse=True)
    cands += sorted(glob.glob(r"C:/Program Files/R/R-*/bin/Rscript.exe"), reverse=True)
    if cands:
        return cands[0]
    return shutil.which("Rscript")


_R_OK: Dict[str, bool] = {}


def r_available() -> bool:
    """True if Rscript exists and lme4 can be loaded. Cached."""
    exe = find_rscript()
    if not exe:
        return False
    if exe in _R_OK:
        return _R_OK[exe]
    try:
        out = subprocess.run([exe, "--vanilla", "-e", "cat(requireNamespace('lme4',quietly=TRUE))"],
                             capture_output=True, text=True, timeout=60)
        _R_OK[exe] = out.stdout.strip().endswith("TRUE")
    except Exception:
        _R_OK[exe] = False
    return _R_OK[exe]


_R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
suppressMessages(library(lme4))
d <- read.csv(args[1], stringsAsFactors = TRUE)
mode <- args[3]
ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
if (mode == "random") {
  m <- suppressWarnings(glmer(y ~ 1 + (1 | person) + (1 | item), data = d, family = binomial, control = ctl))
  b0 <- unname(fixef(m)[1])
  re <- ranef(m, condVar = TRUE)$item
  pv <- attr(re, "postVar")
  items <- data.frame(item = rownames(re), easiness = b0 + re[, 1], se = sqrt(pv[1, 1, ]))
  vc <- as.data.frame(VarCorr(m))
  s_item <- vc$sdcor[vc$grp == "item"]; s_person <- vc$sdcor[vc$grp == "person"]
} else {
  m <- suppressWarnings(glmer(y ~ 0 + item + (1 | person), data = d, family = binomial, control = ctl))
  co <- summary(m)$coefficients
  items <- data.frame(item = sub("^item", "", rownames(co)), easiness = co[, 1], se = co[, 2])
  vc <- as.data.frame(VarCorr(m)); s_item <- NA; s_person <- vc$sdcor[vc$grp == "person"]
}
conv <- is.null(m@optinfo$conv$lme4$messages)
write.csv(items, paste0(args[2], "_items.csv"), row.names = FALSE)
writeLines(sprintf('{"sigma_item": %s, "sigma_person": %s, "converged": %s, "lme4": "%s", "R": "%s"}',
  ifelse(is.na(s_item), "null", format(s_item, digits = 10)), format(s_person, digits = 10),
  ifelse(conv, "true", "false"), as.character(packageVersion("lme4")), R.version.string),
  paste0(args[2], "_meta.json"))
"""


def _fit_r(long: pd.DataFrame, mode: str, timeout: int = 180):
    exe = find_rscript()
    tmp = tempfile.mkdtemp(prefix="b4ai_glmm_")
    try:
        long.to_csv(os.path.join(tmp, "d.csv"), index=False)
        open(os.path.join(tmp, "m.R"), "w", encoding="utf-8").write(_R_SCRIPT)
        pr = subprocess.run([exe, "--vanilla", os.path.join(tmp, "m.R"), os.path.join(tmp, "d.csv"),
                             os.path.join(tmp, "out"), mode], capture_output=True, text=True, timeout=timeout)
        if pr.returncode != 0:
            raise RuntimeError("R failed: " + (pr.stderr or pr.stdout)[-400:])
        items = pd.read_csv(os.path.join(tmp, "out_items.csv"))
        meta = json.load(open(os.path.join(tmp, "out_meta.json"), encoding="utf-8"))
        return items, meta
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------ Python engine
def _fit_python(long: pd.DataFrame, mode: str):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    import statsmodels
    if mode == "random":
        m = BinomialBayesMixedGLM.from_formula(
            "y ~ 1", {"person": "0+C(person)", "item": "0+C(item)"}, long)
        r = m.fit_vb()
        b0 = float(r.fe_mean[0])
        reff = r.random_effects().reset_index()
        reff.columns = ["name", "mean", "sd"]
        it = reff[reff["name"].str.startswith("C(item)")].copy()
        it["item"] = it["name"].str.extract(r"\[(.*)\]$")[0]
        items = pd.DataFrame({"item": it["item"].values, "easiness": b0 + it["mean"].values, "se": it["sd"].values})
        sd = dict(zip(m.vcp_names, np.exp(r.vcp_mean)))
        s_item, s_person = sd.get("item", np.nan), sd.get("person", np.nan)
    else:
        m = BinomialBayesMixedGLM.from_formula("y ~ 0 + C(item)", {"person": "0+C(person)"}, long)
        r = m.fit_vb()
        names = [re.search(r"\[(?:T\.)?(.*)\]$", n).group(1) for n in m.fep_names]
        items = pd.DataFrame({"item": names, "easiness": r.fe_mean, "se": r.fe_sd})
        s_item, s_person = np.nan, float(np.exp(dict(zip(m.vcp_names, r.vcp_mean))["person"]))
    return items, {"sigma_item": None if s_item != s_item else float(s_item),
                   "sigma_person": float(s_person), "converged": True,
                   "statsmodels": statsmodels.__version__}


# ------------------------------------------------------------------------------- public
def to_long(matrix: pd.DataFrame) -> pd.DataFrame:
    """Wide persons x items (1/0/NaN) -> long rows of observed responses only."""
    m = matrix.copy()
    m.index.name = "person"
    long = m.reset_index().melt(id_vars="person", var_name="item", value_name="y").dropna(subset=["y"])
    long["y"] = long["y"].astype(int)
    long["person"] = long["person"].astype(str)
    long["item"] = long["item"].astype(str)
    return long.reset_index(drop=True)


def sample_size_warnings(n_persons: int, n_items: int, mode: str) -> List[str]:
    w: List[str] = []
    if n_persons < 30:
        w.append(f"Only {n_persons} students: below Linacre's (1994) minimum of about 30 for item "
                 "difficulties stable within ±1 logit (95% confidence). Treat as exploratory only.")
    elif n_persons < 50:
        w.append(f"{n_persons} students meets Linacre's (1994) minimum (~30, ±1 logit at 95%) but not "
                 "the 'conservative' 50 (99%). Expect wide intervals.")
    elif n_persons < 100:
        w.append(f"{n_persons} students gives roughly ±1 logit stability; about 100 (95%) to 150 (99%) "
                 "are needed for ±½ logit (Linacre, 1994).")
    if mode == "random" and n_items < FEW_ITEMS:
        w.append(f"Only {n_items} items: the spread of difficulty across items is poorly estimated with so "
                 "few items, so shrinkage is unreliable. Prefer 'items as fixed effects' here.")
    return w


def fit_item_difficulty_glmm(
    matrix: pd.DataFrame,
    engine: str = "auto",
    item_effects: str = "auto",
) -> dict:
    """
    Parameters
    ----------
    matrix : persons x items DataFrame of 1/0/NaN (use ONE test form at a time).
    engine : "auto" | "r" | "python"
    item_effects : "random" (shrinkage) | "fixed" | "auto" (random if >= 10 items, else fixed)

    Returns dict: items (DataFrame), engine, engine_note, mode, sigma_item, sigma_person,
    n_persons, n_items, n_obs, converged, warnings (list), error (None | str)
    """
    out = {"items": pd.DataFrame(), "engine": None, "engine_note": None, "mode": None,
           "sigma_item": None, "sigma_person": None, "n_persons": 0, "n_items": 0, "n_obs": 0,
           "converged": None, "warnings": [], "error": None}
    if matrix is None or matrix.empty:
        out["error"] = "No data."
        return out
    long = to_long(matrix)
    n_p, n_i = long["person"].nunique(), long["item"].nunique()
    out.update(n_persons=n_p, n_items=n_i, n_obs=len(long))
    if n_p < 5 or n_i < 2:
        out["error"] = "Need at least 5 students and 2 items."
        return out
    # an item answered identically by everyone (all right / all wrong) has no finite estimate
    pm = long.groupby("item")["y"].mean()
    degenerate = list(pm[(pm == 0) | (pm == 1)].index)

    mode = item_effects if item_effects in ("random", "fixed") else ("random" if n_i >= FEW_ITEMS else "fixed")
    out["mode"] = mode
    out["warnings"] = sample_size_warnings(n_p, n_i, mode)
    if degenerate:
        out["warnings"].append("Items answered correctly by everyone or by no one have no finite difficulty "
                               "(" + ", ".join(degenerate) + "); with items as fixed effects they are unreliable.")

    use = engine
    if engine == "auto":
        use = "r" if r_available() else "python"
        if use == "python":
            out["engine_note"] = "R/lme4 not available here; used the portable Python engine (variational Bayes approximation)."
    elif engine == "r" and not r_available():
        out["error"] = "R with lme4 is not available on this server."
        return out
    try:
        items, meta = _fit_r(long, mode) if use == "r" else _fit_python(long, mode)
    except Exception as e:  # keep the dashboard alive
        out["error"] = f"Model fitting failed: {e}"
        return out
    out["engine"] = use
    out["sigma_item"], out["sigma_person"] = meta.get("sigma_item"), meta.get("sigma_person")
    out["converged"] = bool(meta.get("converged", True))
    if not out["converged"]:
        out["warnings"].append("The optimizer reported a convergence message; treat estimates with caution.")

    items = items.copy()
    items["difficulty"] = -items["easiness"]
    items["ci_low"] = items["difficulty"] - 1.96 * items["se"]
    items["ci_high"] = items["difficulty"] + 1.96 * items["se"]
    cnt = long.groupby("item")["y"].agg(n="count", p="mean")
    items = items.merge(cnt, left_on="item", right_index=True, how="left")
    items["rank_hardest"] = items["difficulty"].rank(ascending=False, method="min").astype(int)
    items["_k"] = items["item"].map(lambda s: re.sub(r"\d+", lambda m: m.group().zfill(6), str(s)))
    out["items"] = (items.sort_values("_k").drop(columns=["_k", "easiness"])
                    [["item", "n", "p", "difficulty", "se", "ci_low", "ci_high", "rank_hardest"]]
                    .reset_index(drop=True))
    return out
