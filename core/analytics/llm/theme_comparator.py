# core/analytics/llm/theme_comparator.py
"""
Theme Comparator
================
Computes quantitative agreement between ITA theme sets produced by
different models and/or temperatures.

Answers two research questions from De Paoli (2024):
    1. Do different LLMs produce consistent themes? (model comparison)
    2. Do higher temperatures produce different themes? (temp comparison)

Metrics:
    Cosine similarity  — semantic overlap between theme descriptions
    Jaccard similarity — lexical overlap between theme name token sets
    Agreement score    — weighted composite of both

Public API:
-----------
compare_runs(run_a, run_b)
    Pairwise comparison of two theme lists.
    Returns similarity matrix + aggregate scores.

compare_all_runs(runs_dict)
    Compare all pairs in a dict of {label: themes}.
    Returns full comparison table for dashboard display.

get_best_match(theme, candidate_themes)
    Find the closest matching theme from a list.
    Used for side-by-side display alignment.

align_themes(themes_a, themes_b)
    Align two theme lists by best semantic match.
    Returns aligned DataFrame for side-by-side comparison.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# Embedding helper (reuses deduplicator's model cache if loaded)
# -----------------------------------------------------------------------

# Module-level model cache — loaded once, reused across all calls
_MODEL_CACHE = None

def _get_model():
    """Load sentence-transformers model once, cache at module level."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    try:
        # Prefer deduplicator's cache if already loaded in this session
        try:
            from core.analytics.llm.deduplicator import _get_model as _dedup_get
            _MODEL_CACHE = _dedup_get()
            return _MODEL_CACHE
        except ImportError:
            pass
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE = SentenceTransformer("all-MiniLM-L6-v2")
        return _MODEL_CACHE
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )


def _embed(texts: List[str]) -> np.ndarray:
    """Encode texts using cached sentence-transformers model."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True,
                        show_progress_bar=False)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# -----------------------------------------------------------------------
# Jaccard similarity on token sets
# -----------------------------------------------------------------------

def _tokenize(text: str) -> set:
    """Lowercase word tokens, strip punctuation."""
    import re
    return set(re.findall(r"[a-z]+", text.lower()))


def _jaccard(text_a: str, text_b: str) -> float:
    """
    Jaccard similarity between token sets of two strings.
    J(A,B) = |A ∩ B| / |A ∪ B|
    """
    a, b = _tokenize(text_a), _tokenize(text_b)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# -----------------------------------------------------------------------
# Theme text builder
# -----------------------------------------------------------------------

def _theme_to_text(theme: Dict[str, Any]) -> str:
    """
    Combine theme name + description for embedding.
    Matches the code_to_text pattern in deduplicator.
    """
    name = str(theme.get("name", "")).strip()
    desc = str(theme.get("description", "")).strip()
    if desc:
        return f"{name}. {desc}"
    return name


def _validate_themes(themes: List[Dict[str, Any]], label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    for i, t in enumerate(themes):
        if not isinstance(t, dict):
            raise ValueError(f"{prefix}Theme {i} must be a dict.")
        if "name" not in t:
            raise ValueError(
                f"{prefix}Theme {i} missing 'name' field. "
                f"Each theme must have at least 'name' and 'description'."
            )


# -----------------------------------------------------------------------
# Public function 1: compare_runs
# -----------------------------------------------------------------------

def compare_runs(
    run_a: List[Dict[str, Any]],
    run_b: List[Dict[str, Any]],
    label_a: str = "Run A",
    label_b: str = "Run B",
    cosine_weight: float = 0.7,
    jaccard_weight: float = 0.3,
) -> Dict[str, Any]:
    """
    Pairwise comparison of two theme lists.

    Computes:
        - Pairwise cosine similarity matrix (n_a × n_b)
        - Pairwise Jaccard similarity matrix (n_a × n_b)
        - Weighted agreement score per theme in run_a
        - Overall agreement score (mean of best-match scores)
        - Interpretation label

    Parameters
    ----------
    run_a, run_b : list of theme dicts
        Each dict must have 'name'. 'description' improves similarity.
    label_a, label_b : str
        Display labels for the two runs.
    cosine_weight, jaccard_weight : float
        Weights for composite score. Must sum to 1.0.

    Returns
    -------
    dict:
        label_a, label_b,
        n_themes_a, n_themes_b,
        cosine_matrix     pd.DataFrame  (n_a × n_b)
        jaccard_matrix    pd.DataFrame  (n_a × n_b)
        agreement_matrix  pd.DataFrame  (n_a × n_b, weighted composite)
        best_matches      list of dicts  per theme in run_a:
            {theme_a, best_match_b, cosine, jaccard, agreement}
        overall_agreement float  mean of best-match agreement scores
        interpretation    str    "strong"|"moderate"|"weak"|"poor"
        error             str|None
    """
    result: Dict[str, Any] = {
        "label_a": label_a,
        "label_b": label_b,
        "n_themes_a": len(run_a),
        "n_themes_b": len(run_b),
        "error": None,
    }

    try:
        _validate_themes(run_a, label_a)
        _validate_themes(run_b, label_b)

        if not run_a or not run_b:
            result["error"] = (
                f"Both runs must have at least one theme. "
                f"Got: {label_a}={len(run_a)}, {label_b}={len(run_b)}"
            )
            return result

        names_a = [t["name"] for t in run_a]
        names_b = [t["name"] for t in run_b]
        texts_a = [_theme_to_text(t) for t in run_a]
        texts_b = [_theme_to_text(t) for t in run_b]

        # Embed all texts together for efficiency
        all_texts  = texts_a + texts_b
        all_embeds = _embed(all_texts)
        embeds_a   = all_embeds[:len(texts_a)]
        embeds_b   = all_embeds[len(texts_a):]

        na, nb = len(run_a), len(run_b)

        # Build matrices
        cosine_mat   = np.zeros((na, nb))
        jaccard_mat  = np.zeros((na, nb))

        for i in range(na):
            for j in range(nb):
                cosine_mat[i, j]  = _cosine_sim(embeds_a[i], embeds_b[j])
                jaccard_mat[i, j] = _jaccard(texts_a[i], texts_b[j])

        agreement_mat = (
            cosine_weight  * cosine_mat +
            jaccard_weight * jaccard_mat
        )

        # DataFrames for display
        result["cosine_matrix"]    = pd.DataFrame(
            np.round(cosine_mat,   4), index=names_a, columns=names_b)
        result["jaccard_matrix"]   = pd.DataFrame(
            np.round(jaccard_mat,  4), index=names_a, columns=names_b)
        result["agreement_matrix"] = pd.DataFrame(
            np.round(agreement_mat,4), index=names_a, columns=names_b)

        # Best match per theme in run_a
        best_matches = []
        best_scores  = []

        for i, theme_a in enumerate(run_a):
            best_j    = int(np.argmax(agreement_mat[i]))
            best_cos  = round(float(cosine_mat[i, best_j]),   4)
            best_jac  = round(float(jaccard_mat[i, best_j]),  4)
            best_agr  = round(float(agreement_mat[i, best_j]),4)
            best_scores.append(best_agr)

            best_matches.append({
                "theme_a":    theme_a["name"],
                "best_match_b": run_b[best_j]["name"],
                "cosine":     best_cos,
                "jaccard":    best_jac,
                "agreement":  best_agr,
            })

        result["best_matches"] = best_matches

        overall = float(np.mean(best_scores)) if best_scores else 0.0
        result["overall_agreement"] = round(overall, 4)
        result["interpretation"]    = _interpret_agreement(overall)

    except Exception as e:
        result["error"] = str(e)

    return result


# -----------------------------------------------------------------------
# Public function 2: compare_all_runs
# -----------------------------------------------------------------------

def compare_all_runs(
    runs_dict: Dict[str, List[Dict[str, Any]]],
    cosine_weight: float = 0.7,
    jaccard_weight: float = 0.3,
) -> Dict[str, Any]:
    """
    Compare all pairs in a dict of {label: themes}.

    Typical usage:
        runs = {
            "Claude T=0":   claude_t0_themes,
            "Claude T=0.5": claude_t05_themes,
            "Gemini T=0":   gemini_t0_themes,
            "GPT T=0":      gpt_t0_themes,
        }
        result = compare_all_runs(runs)

    Parameters
    ----------
    runs_dict : dict
        Keys = display labels, values = theme lists.

    Returns
    -------
    dict:
        labels          list of str
        pairwise        dict of {(label_a, label_b): compare_runs result}
        summary_matrix  pd.DataFrame  (labels × labels, overall_agreement)
        error           str|None
    """
    labels = list(runs_dict.keys())
    n      = len(labels)

    if n < 2:
        return {
            "labels": labels,
            "pairwise": {},
            "summary_matrix": pd.DataFrame(),
            "error": "Need at least 2 runs to compare.",
        }

    pairwise = {}
    summary  = np.zeros((n, n))
    np.fill_diagonal(summary, 1.0)  # self-similarity = 1.0

    for i in range(n):
        for j in range(i + 1, n):
            la, lb = labels[i], labels[j]
            r = compare_runs(
                runs_dict[la], runs_dict[lb],
                label_a=la, label_b=lb,
                cosine_weight=cosine_weight,
                jaccard_weight=jaccard_weight,
            )
            pairwise[(la, lb)] = r
            if not r.get("error"):
                score = r["overall_agreement"]
                summary[i, j] = score
                summary[j, i] = score  # symmetric

    summary_df = pd.DataFrame(
        np.round(summary, 4),
        index=labels,
        columns=labels,
    )

    return {
        "labels":         labels,
        "pairwise":       pairwise,
        "summary_matrix": summary_df,
        "error":          None,
    }


# -----------------------------------------------------------------------
# Public function 3: get_best_match
# -----------------------------------------------------------------------

def get_best_match(
    theme: Dict[str, Any],
    candidate_themes: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], float]:
    """
    Find the closest matching theme from a list of candidates.

    Parameters
    ----------
    theme : dict
    candidate_themes : list of dicts

    Returns
    -------
    (best_match_theme, agreement_score) tuple
    """
    if not candidate_themes:
        return {}, 0.0

    _validate_themes([theme])
    _validate_themes(candidate_themes)

    query_text = _theme_to_text(theme)
    cand_texts = [_theme_to_text(t) for t in candidate_themes]

    all_embeds = _embed([query_text] + cand_texts)
    query_emb  = all_embeds[0]
    cand_embs  = all_embeds[1:]

    scores = []
    for i, cand_emb in enumerate(cand_embs):
        cos = _cosine_sim(query_emb, cand_emb)
        jac = _jaccard(query_text, cand_texts[i])
        scores.append(0.7 * cos + 0.3 * jac)

    best_idx = int(np.argmax(scores))
    return candidate_themes[best_idx], round(float(scores[best_idx]), 4)


# -----------------------------------------------------------------------
# Public function 4: align_themes
# -----------------------------------------------------------------------

def align_themes(
    themes_a: List[Dict[str, Any]],
    themes_b: List[Dict[str, Any]],
    label_a: str = "Run A",
    label_b: str = "Run B",
) -> pd.DataFrame:
    """
    Align two theme lists by best semantic match for side-by-side display.

    For each theme in themes_a, finds the best matching theme in themes_b.
    The result is a DataFrame with one row per theme in themes_a.

    Parameters
    ----------
    themes_a, themes_b : list of theme dicts
    label_a, label_b : str

    Returns
    -------
    pd.DataFrame with columns:
        f"{label_a} Theme",
        f"{label_a} Description",
        f"{label_b} Best Match",
        f"{label_b} Description",
        "Agreement Score",
        "Interpretation"
    """
    _validate_themes(themes_a, label_a)
    _validate_themes(themes_b, label_b)

    if not themes_a or not themes_b:
        return pd.DataFrame()

    rows = []
    for theme_a in themes_a:
        best_b, score = get_best_match(theme_a, themes_b)
        rows.append({
            f"{label_a} Theme":       theme_a.get("name", ""),
            f"{label_a} Description": theme_a.get("description", ""),
            f"{label_b} Best Match":  best_b.get("name", ""),
            f"{label_b} Description": best_b.get("description", ""),
            "Agreement Score":        score,
            "Interpretation":         _interpret_agreement(score),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Internal: agreement interpretation
# -----------------------------------------------------------------------

def _interpret_agreement(score: float) -> str:
    """
    Interpret an agreement score as a qualitative label.

    Scale:
        ≥ 0.80  strong    — themes are highly consistent
        ≥ 0.65  moderate  — themes partially overlap
        ≥ 0.50  weak      — some relationship but substantial differences
        < 0.50  poor      — themes are largely inconsistent
    """
    if score >= 0.80:
        return "strong"
    if score >= 0.65:
        return "moderate"
    if score >= 0.50:
        return "weak"
    return "poor"
