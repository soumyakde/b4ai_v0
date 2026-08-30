# core/analytics/llm/theme_comparator.py
"""
Theme Comparator
================
Computes quantitative agreement between ITA theme sets produced by
different models and/or temperatures.

Answers two research questions from De Paoli (2024):
    1. Do different LLMs produce consistent themes? (model comparison)
    2. Do higher temperatures produce different themes? (temp comparison)
(The two research questions above are De Paoli's; the agreement formula,
matching algorithm, and thresholds below are original to this codebase,
not adapted from his paper -- his own comparison method is qualitative,
with no numeric metric at all.)

Metrics:
    Cosine similarity  — semantic overlap between theme descriptions
    Jaccard similarity — lexical overlap between theme name token sets
    Agreement score    — weighted composite of both

Matching (compare_runs, compare_all_runs):
    Theme pairs are found via the Hungarian algorithm (Kuhn, 1955) --
    the optimal one-to-one assignment between run_a and run_b's themes,
    not a greedy "each theme claims its nearest neighbor independently"
    approach (which previously let multiple run_a themes claim the same
    run_b theme, silently inflating agreement). A pair scoring below
    MATCH_FLOOR is flagged matched=False but its real score still counts
    toward the average -- a weak match is real information, not something
    to discard. If the two runs have different theme counts, the surplus
    themes get no counterpart at all (matched=False, best_match_b=None),
    which is itself evidence the runs didn't converge on the same
    structure, not just content-level disagreement.

    compare_all_runs additionally fixed a hidden asymmetry: it used to
    compute compare_runs(A, B) once and mirror that single score into
    both directions of the summary matrix, silently assuming symmetry
    that Hungarian matching on a rectangular (unequal theme count) matrix
    does not guarantee. It now computes both directions and averages them.

Embedding backend:
    Default is the local sentence-transformers model (all-MiniLM-L6-v2,
    free, no API calls). compare_runs/compare_all_runs also accept
    embedding_backend="openai" (text-embedding-3-small) as an optional,
    architecturally distinct second encoder for a robustness check -- per
    MTEB (Muennighoff et al., 2023), no single embedding model dominates
    across tasks, so a result that only holds under one specific encoder's
    geometry is a weaker claim than one that holds under two. This is
    deliberately NOT wired into the Teacher Dashboard UI: unlike the
    default backend, it costs real OpenAI API money per call, so it's a
    callable available for methods-paper/validation use rather than a
    button any everyday dashboard use could trigger unexpectedly.

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
from scipy.optimize import linear_sum_assignment

# Similarity floor below which an assigned pair is flagged "matched: False"
# in compare_runs' output. A provisional default, not yet empirically
# calibrated -- Phase 6 of the ongoing methodology work will calibrate
# this against a human-coded subset via ROC/Youden's J rather than leave
# it as a guessed constant indefinitely.
MATCH_FLOOR = 0.3


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


def _embed_openai(texts: List[str], model_id: str = "text-embedding-3-small") -> np.ndarray:
    """
    Encode texts via OpenAI's embeddings API -- a second, architecturally
    distinct embedding backend used only for robustness checks (see module
    docstring). Costs real API money per call, unlike the default local
    backend. Raises RuntimeError with a clear message if OPENAI_API_KEY
    isn't configured; callers (compare_runs) already catch and surface
    exceptions as result["error"], so no extra handling is needed here.
    """
    from core.analytics.llm.llm_clients import _load_keys
    api_key = _load_keys().get("openai")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found -- required for embedding_backend="
            "'openai'. Add it to .env or st.secrets, or use the default "
            "'minilm' backend instead."
        )
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model_id, input=texts)
    return np.array([d.embedding for d in response.data])


def _embed(texts: List[str], backend: str = "minilm") -> np.ndarray:
    """
    Encode texts using the requested embedding backend.

    backend : "minilm" (default, local, free) | "openai" (robustness
        check only -- see module docstring).
    """
    if backend == "openai":
        return _embed_openai(texts)
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
    embedding_backend: str = "minilm",
) -> Dict[str, Any]:
    """
    Pairwise comparison of two theme lists.

    Computes:
        - Pairwise cosine similarity matrix (n_a × n_b)
        - Pairwise Jaccard similarity matrix (n_a × n_b)
        - Optimal one-to-one theme matching (Hungarian algorithm, Kuhn 1955)
        - Overall agreement score (mean over run_a themes' matched scores)
        - Interpretation label

    Parameters
    ----------
    run_a, run_b : list of theme dicts
        Each dict must have 'name'. 'description' improves similarity.
    label_a, label_b : str
        Display labels for the two runs.
    cosine_weight, jaccard_weight : float
        Weights for composite score. Must sum to 1.0.
    embedding_backend : str
        "minilm" (default, local, free) or "openai" (text-embedding-3-small,
        a robustness-check backend -- see module docstring; costs real API
        money and requires OPENAI_API_KEY).

    Returns
    -------
    dict:
        label_a, label_b,
        n_themes_a, n_themes_b, embedding_backend,
        cosine_matrix     pd.DataFrame  (n_a × n_b)
        jaccard_matrix    pd.DataFrame  (n_a × n_b)
        agreement_matrix  pd.DataFrame  (n_a × n_b, weighted composite)
        best_matches      list of dicts  per theme in run_a:
            {theme_a, best_match_b, cosine, jaccard, agreement, matched}
            best_match_b/cosine/jaccard are None when run_a has more
            themes than run_b and this one received no assignment at all.
        overall_agreement float  mean of best_matches' agreement scores
            (unmatched/unassigned themes' scores still count -- see
            module docstring on why this isn't survivorship-biased)
        interpretation    str    "strong"|"moderate"|"weak"|"poor"
        error             str|None
    """
    result: Dict[str, Any] = {
        "label_a": label_a,
        "label_b": label_b,
        "n_themes_a": len(run_a),
        "n_themes_b": len(run_b),
        "embedding_backend": embedding_backend,
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
        all_embeds = _embed(all_texts, backend=embedding_backend)
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

        # Optimal one-to-one assignment (Hungarian algorithm) instead of a
        # greedy per-row argmax -- prevents multiple run_a themes silently
        # claiming the same run_b theme. linear_sum_assignment minimizes
        # cost, so negate the agreement matrix to maximize agreement.
        # On a rectangular matrix it returns min(na, nb) pairs; any run_a
        # row not in row_ind (only possible when na > nb) gets no
        # assignment at all -- handled explicitly below, not silently
        # dropped.
        row_ind, col_ind = linear_sum_assignment(-agreement_mat)
        assigned = dict(zip(row_ind.tolist(), col_ind.tolist()))

        best_matches = []
        best_scores  = []

        for i, theme_a in enumerate(run_a):
            if i in assigned:
                j = assigned[i]
                best_cos = round(float(cosine_mat[i, j]),    4)
                best_jac = round(float(jaccard_mat[i, j]),   4)
                best_agr = round(float(agreement_mat[i, j]), 4)
                best_scores.append(best_agr)
                best_matches.append({
                    "theme_a":      theme_a["name"],
                    "best_match_b": run_b[j]["name"],
                    "cosine":       best_cos,
                    "jaccard":      best_jac,
                    "agreement":    best_agr,
                    "matched":      best_agr >= MATCH_FLOOR,
                })
            else:
                # na > nb: genuinely no counterpart for this theme.
                best_scores.append(0.0)
                best_matches.append({
                    "theme_a":      theme_a["name"],
                    "best_match_b": None,
                    "cosine":       None,
                    "jaccard":      None,
                    "agreement":    0.0,
                    "matched":      False,
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
    embedding_backend: str = "minilm",
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
    embedding_backend : str
        See compare_runs() -- "minilm" (default) or "openai".

    Returns
    -------
    dict:
        labels          list of str
        pairwise        dict of {(label_a, label_b): compare_runs result}
            -- contains BOTH directions for every unordered pair (i.e.
            both (A, B) and (B, A) are present, each a real, independently
            computed compare_runs() call, not one mirrored into the other)
        summary_matrix  pd.DataFrame  (labels × labels) -- off-diagonal
            cells are the AVERAGE of both directions' overall_agreement,
            not a single direction copied into both cells. compare_runs'
            Hungarian matching on a rectangular (unequal theme-count)
            matrix is not guaranteed symmetric, so computing only one
            direction and mirroring it (the previous behavior) silently
            assumed a symmetry that doesn't always hold.
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
            r_ab = compare_runs(
                runs_dict[la], runs_dict[lb],
                label_a=la, label_b=lb,
                cosine_weight=cosine_weight,
                jaccard_weight=jaccard_weight,
                embedding_backend=embedding_backend,
            )
            r_ba = compare_runs(
                runs_dict[lb], runs_dict[la],
                label_a=lb, label_b=la,
                cosine_weight=cosine_weight,
                jaccard_weight=jaccard_weight,
                embedding_backend=embedding_backend,
            )
            pairwise[(la, lb)] = r_ab
            pairwise[(lb, la)] = r_ba

            scores = [
                r["overall_agreement"] for r in (r_ab, r_ba) if not r.get("error")
            ]
            if scores:
                score = float(np.mean(scores))
                summary[i, j] = score
                summary[j, i] = score  # symmetric by construction (averaged)

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
