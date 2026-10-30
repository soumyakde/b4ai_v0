"""
core/analytics/irt/ctt_item_analysis.py

Classical Test Theory (CTT) item analysis -- Method 1 of the three
item-difficulty options (see PROJECT_STATUS.md, "Item-difficulty feature
plan"). Pure pandas/numpy/scipy/statsmodels: no Streamlit, no database,
no model fitting, so it is safe at any sample size and is unit-testable.

What it reports
---------------
Per item
    n               responses to the item
    p               proportion correct ("item difficulty"; HIGHER = EASIER)
    p_ci_low/high   95% Wilson interval for p (uncertainty is shown, never hidden)
    r_it            corrected item-total (item-rest) point-biserial correlation
    r_ci_low/high   95% Fisher-z interval for r_it
    alpha_if_deleted  KR-20 of the remaining items
    chance / p_below_chance   optional: one-sided binomial test that accuracy is
                    BELOW chance (a strong signal the answer key is wrong)
    flags           plain-language review flags
Per test
    KR-20, SEM, mean/SD of total score, mean item difficulty, n complete cases
Per distractor (needs raw option letters)
    % choosing each option overall and in five ability groups (trace table),
    non-functioning-distractor flag (< 5% chosen), and the distractor
    point-biserial of Attali & Fraenkel (2000).

References (verified against the sources named)
-----------------------------------------------
Kuder, G. F., & Richardson, M. W. (1937). The theory of the estimation of test
    reliability. Psychometrika, 2(3), 151-160.                       [KR-20]
Crocker, L., & Algina, J. (1986). Introduction to Classical and Modern Test
    Theory. Holt, Rinehart & Winston.                  [p, r_it, SEM background]
Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. J. Amer. Statist. Assoc., 22(158), 209-212.
Gierl, M. J., Bulut, O., Guo, Q., & Zhang, X. (2017). Developing, analyzing, and
    using distractors for multiple-choice tests in education: A comprehensive
    review. Review of Educational Research, 87(6), 1082-1116.
    -- the < 5% "non-functioning distractor" rule (credited there to Haladyna &
    Downing, 1993), its exception for items answered correctly by > 90%, the
    five-ability-group trace table, and the distractor point-biserial of
    Attali & Fraenkel (2000, J. Educ. Measurement, 37, 77-86).
The "very hard / very easy" and "weak discrimination" cut-offs below are
common rules of thumb, exposed as constants so a researcher can change them;
they are review prompts, not verdicts.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Review-flag thresholds (rules of thumb -- adjustable).
P_HARD = 0.20
P_EASY = 0.90
R_WEAK = 0.20
NONFUNCTIONING_SHARE = 0.05      # Gierl et al. (2017), citing Haladyna & Downing (1993)
BELOW_CHANCE_ALPHA = 0.01
MIN_N_FOR_R = 10                 # below this, item-total r is not reported
SMALL_N = 30                     # below this, show a "descriptive only" warning


def _wilson(k: float, n: float, conf: float = 0.95):
    """Wilson score interval for a proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    ph = k / n
    denom = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / denom
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _fisher_ci(r: float, n: int, conf: float = 0.95):
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - (1 - conf) / 2)
    return (float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se)))


def kr20(matrix: pd.DataFrame) -> float:
    """KR-20 (= Cronbach's alpha for 0/1 items) on a complete-case matrix."""
    k = matrix.shape[1]
    if k < 2 or len(matrix) < 2:
        return float("nan")
    total_var = matrix.sum(axis=1).var(ddof=0)
    if total_var == 0:
        return float("nan")
    pq = (matrix.mean(axis=0) * (1 - matrix.mean(axis=0))).sum()
    return float((k / (k - 1)) * (1 - pq / total_var))


def sample_size_note(n: int) -> Optional[str]:
    """Plain-language caution tied to the sampling arithmetic, not an arbitrary gate."""
    if n < 3:
        return "Fewer than 3 students: nothing can be estimated."
    half = 1.96 * np.sqrt(0.25 / n)
    base = (
        f"With {n} students, an item's percent-correct can be uncertain by about "
        f"±{100 * half:.0f} percentage points (95% interval, worst case)."
    )
    if n < SMALL_N:
        return base + " Treat every statistic below as DESCRIPTIVE only."
    if n < 100:
        return base + " Item-total correlations are unstable at this size: use them to flag items for review, not to rank items."
    return base


def list_forms(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Detect test "forms": groups of students who were given exactly the same set of
    items. Needed because the module assessments changed between cohorts (and were
    shortened mid-pilot), so a single persons-x-items matrix has blocks of
    never-administered items and classical statistics are only defined within a form.

    Returns DataFrame: form, n_students, n_items, items (comma list), sorted largest first.
    """
    if matrix is None or matrix.empty:
        return pd.DataFrame(columns=["form", "n_students", "n_items", "items"])
    pat = matrix.notna().apply(lambda r: tuple(r.values), axis=1)
    rows = []
    for key, idx in pat.groupby(pat).groups.items():
        items = [c for c, on in zip(matrix.columns, key) if on]
        if items:
            rows.append({"n_students": len(idx), "n_items": len(items), "items": ", ".join(items),
                         "_students": list(idx), "_items": items})
    out = pd.DataFrame(rows).sort_values("n_students", ascending=False).reset_index(drop=True)
    out.insert(0, "form", [f"Form {i + 1}" for i in range(len(out))])
    return out


def subset_form(matrix: pd.DataFrame, forms: pd.DataFrame, form: str) -> pd.DataFrame:
    """Rows/columns of `matrix` belonging to one form returned by list_forms()."""
    row = forms.loc[forms["form"] == form].iloc[0]
    return matrix.loc[row["_students"], row["_items"]]


def compute_item_analysis(
    matrix: pd.DataFrame,
    n_options: Optional[Dict[str, int]] = None,
) -> dict:
    """
    Parameters
    ----------
    matrix : persons x items DataFrame of 1.0 / 0.0 / NaN (NaN = not answered or
        item not administered to that student).
    n_options : optional {item_id: number of answer options}, enabling the
        below-chance check.

    Returns
    -------
    dict with keys: items (DataFrame), test (dict), n_persons, n_complete,
    note (sample-size caution), error (None or str).
    """
    out = {"items": pd.DataFrame(), "test": {}, "n_persons": 0, "n_complete": 0,
           "note": None, "error": None}
    if matrix is None or matrix.empty:
        out["error"] = "No data."
        return out
    m = matrix.astype(float)
    out["n_persons"] = int(len(m))
    out["note"] = sample_size_note(len(m))

    complete = m.dropna(axis=0, how="any")
    out["n_complete"] = int(len(complete))
    k = m.shape[1]

    rows = []
    for item in m.columns:
        col = m[item].dropna()
        n_i = int(len(col))
        successes = float(col.sum())
        p = successes / n_i if n_i else np.nan
        lo, hi = _wilson(successes, n_i)

        r = np.nan
        r_lo = r_hi = np.nan
        a_del = np.nan
        if len(complete) >= MIN_N_FOR_R and k >= 2:
            rest = complete.drop(columns=[item]).sum(axis=1)
            x = complete[item]
            if x.std(ddof=0) > 0 and rest.std(ddof=0) > 0:
                r = float(np.corrcoef(x, rest)[0, 1])
                r_lo, r_hi = _fisher_ci(r, len(complete))
            if k > 2:
                a_del = kr20(complete.drop(columns=[item]))

        chance = pb = np.nan
        if n_options and item in n_options and n_options[item] >= 2:
            chance = 1.0 / n_options[item]
            if n_i:
                pb = float(stats.binomtest(int(round(successes)), n_i, chance,
                                           alternative="less").pvalue)

        flags: List[str] = []
        if np.isfinite(p):
            if p < P_HARD:
                flags.append("very hard (p < %.2f)" % P_HARD)
            elif p > P_EASY:
                flags.append("very easy (p > %.2f)" % P_EASY)
        if np.isfinite(r):
            if r < 0:
                flags.append("NEGATIVE discrimination: check the answer key")
            elif r < R_WEAK:
                flags.append("weak discrimination (r < %.2f)" % R_WEAK)
        if np.isfinite(pb) and pb < BELOW_CHANCE_ALPHA:
            flags.append("accuracy BELOW chance: check the answer key")

        rows.append({
            "item": item, "n": n_i, "p": p, "p_ci_low": lo, "p_ci_high": hi,
            "r_it": r, "r_ci_low": r_lo, "r_ci_high": r_hi,
            "alpha_if_deleted": a_del, "chance": chance, "p_below_chance": pb,
            "flags": "; ".join(flags),
        })
    items = pd.DataFrame(rows)
    out["items"] = items

    test = {"k_items": k, "mean_item_p": float(items["p"].mean()) if len(items) else np.nan}
    if len(complete) >= 2:
        tot = complete.sum(axis=1)
        a = kr20(complete)
        test.update({
            "n_complete": int(len(complete)), "mean_total": float(tot.mean()),
            "sd_total": float(tot.std(ddof=1)), "kr20": a,
            "sem": float(tot.std(ddof=1) * np.sqrt(max(0.0, 1 - a))) if np.isfinite(a) else np.nan,
        })
    else:
        test.update({"n_complete": int(len(complete)), "kr20": np.nan, "sem": np.nan})
    out["test"] = test
    return out


def _letter(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return (s.split(":")[0].strip() if ":" in s else s) or None


def compute_distractor_analysis(
    canonical_df: pd.DataFrame,
    instrument_key: str,
    n_groups: int = 5,
    option_letters: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Option-level analysis for one multiple-choice instrument.

    The keyed option for each item is inferred from the data (the letter chosen by
    responses with item_score == 1), so it stays correct when an item's key changed
    between cohorts' versions. Ability = the student's score on all OTHER items
    (rest score), split into `n_groups` ordinal groups (Gierl et al., 2017, Fig. 2).

    `option_letters` ({item_id: ['A','B',...]}) lets options NOBODY chose appear as
    0% rows (the clearest non-functioning distractors); without it only options
    that were chosen at least once are listed.

    Returns one row per (item, option): share overall, share in each ability
    group (g1 = lowest), keyed?, distractor point-biserial, flags.
    """
    mask = ((canonical_df["instrument_key"] == instrument_key)
            | canonical_df["instrument_key"].str.endswith("_" + instrument_key))
    d = canonical_df[mask & canonical_df["item_score"].isin([0.0, 1.0])].copy()
    if d.empty:
        return pd.DataFrame()
    d["letter"] = d["response_value"].map(_letter)
    d = d[d["letter"].notna()]
    score = d.pivot_table(index="user_id", columns="question_id", values="item_score", aggfunc="first")
    total = score.sum(axis=1, min_count=1)

    rows = []
    for q, g in d.groupby("question_id"):
        g = g.drop_duplicates("user_id").set_index("user_id")
        correct = g.loc[g["item_score"] == 1.0, "letter"]
        key = correct.mode().iloc[0] if len(correct) else None
        rest = (total.reindex(g.index) - g["item_score"])
        valid = rest.notna()
        g, rest = g[valid], rest[valid]
        if len(g) < 2:
            continue
        ranks = rest.rank(method="first")
        grp = pd.qcut(ranks, q=min(n_groups, len(g)), labels=False) + 1
        n = len(g)
        p_key = float((g["letter"] == key).mean()) if key else np.nan
        counts = g["letter"].value_counts()
        all_opts = list(counts.index)
        for extra in (option_letters or {}).get(q, []):
            if extra not in all_opts:
                all_opts.append(extra)
        for opt in all_opts:
            cnt = int(counts.get(opt, 0))
            share = cnt / n
            row = {"item": q, "option": opt, "keyed": opt == key, "n": n, "share": share}
            for j in range(1, n_groups + 1):
                gi = g[grp == j]
                row[f"g{j}"] = float((gi["letter"] == opt).mean()) if len(gi) else np.nan
            pb = np.nan
            if key and opt != key:
                sub = g[g["letter"].isin([opt, key])]
                if len(sub) >= 5 and sub["letter"].nunique() == 2:
                    x = (sub["letter"] == opt).astype(float)
                    y = rest.reindex(sub.index)
                    if x.std(ddof=0) > 0 and y.std(ddof=0) > 0:
                        pb = float(np.corrcoef(x, y)[0, 1])
            row["distractor_r"] = pb
            flags = []
            if opt != key:
                if share < NONFUNCTIONING_SHARE and not (p_key > P_EASY):
                    flags.append("non-functioning (< 5% chose it)")
                if np.isfinite(pb) and pb > 0:
                    flags.append("attracts stronger students than the keyed option: review wording/key")
            row["flags"] = "; ".join(flags)
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_k"] = out["item"].map(lambda s: re.sub(r"\d+", lambda m: m.group().zfill(6), str(s)))
    out = out.sort_values(["_k", "option"]).drop(columns="_k").reset_index(drop=True)
    return out
