# core/analytics/llm/deduplicator.py
"""
Embedding-Based Code Deduplicator
==================================
Removes semantically redundant codes from Phase 2 of ITA using
sentence-transformers (all-MiniLM-L6-v2).

No API calls — runs locally, no cost per deduplication.

Anti-hallucination strategy (De Paoli):
    Each code carries its dataframe chunk index.
    When merging duplicates, the code with the LOWEST index is kept
    so the pipeline can always trace back to the source chunk.

Public API:
-----------
deduplicate_codes(codes, threshold=0.85)
    Remove semantically redundant codes from a list.

cluster_codes(codes, threshold=0.85)
    Group codes into semantic clusters without removing any.
    Returns clusters with the representative + members.

compute_similarity_matrix(codes)
    Return pairwise cosine similarity matrix as DataFrame.
    Used for visualisation in the dashboard.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# Model loading — lazy, cached at module level after first load
# -----------------------------------------------------------------------
_MODEL = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    """
    Load sentence-transformers model once, cache for subsequent calls.
    Raises RuntimeError if sentence-transformers is not installed.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_MODEL_NAME)
        return _MODEL
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )


def _embed(texts: List[str]) -> np.ndarray:
    """
    Encode a list of texts to embeddings.
    Returns ndarray of shape (n_texts, embedding_dim).
    """
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.
    Returns ndarray of shape (n, n), values in [-1, 1].
    """
    # L2-normalise each row
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms
    return normed @ normed.T


def _code_to_text(code: Dict[str, Any]) -> str:
    """
    Combine code name + description into a single string for embedding.
    This gives the model more semantic signal than name alone.
    """
    name = str(code.get("name", "")).strip()
    desc = str(code.get("description", "")).strip()
    if desc:
        return f"{name}. {desc}"
    return name


# -----------------------------------------------------------------------
# Public function 1: deduplicate_codes
# -----------------------------------------------------------------------

def deduplicate_codes(
    codes: List[Dict[str, Any]],
    threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    Remove semantically redundant codes using cosine similarity.

    Parameters
    ----------
    codes : list of dicts
        Each code dict must have:
            "name"        str   — 3-word code name (De Paoli)
            "description" str   — 4-line description
            "quote"       str   — supporting quote from data
            "chunk_index" int   — dataframe row index (anti-hallucination)

        Optional fields are preserved unchanged:
            "model"       str   — which LLM produced this code
            any other keys

    threshold : float
        Cosine similarity above which two codes are considered duplicates.
        Default 0.85 — conservative, preserves nuance.
        Range: 0.0 (keep everything) to 1.0 (keep only identical).
        Recommended range: 0.80–0.90 for ITA.

    Returns
    -------
    list of dicts
        Deduplicated codes. When duplicates are found, the code with
        the LOWEST chunk_index is kept (De Paoli anti-hallucination).
        All original fields are preserved on the kept code.
        A "merged_count" field is added indicating how many duplicates
        were absorbed.

    Raises
    ------
    ValueError
        If codes is empty or any code is missing required fields.
    RuntimeError
        If sentence-transformers is not installed.
    """
    if not codes:
        return []

    _validate_codes(codes)

    if len(codes) == 1:
        result = codes[0].copy()
        result["merged_count"] = 0
        return [result]

    # Sort by chunk_index ascending — lowest index wins ties
    sorted_codes = sorted(
        codes,
        key=lambda c: int(c.get("chunk_index", 0))
    )

    texts = [_code_to_text(c) for c in sorted_codes]
    embeddings = _embed(texts)
    sim_matrix = _cosine_similarity_matrix(embeddings)

    n = len(sorted_codes)
    kept    = [True] * n  # True = this code survives
    absorbed_by = {}      # index → index of the code that absorbed it

    for i in range(n):
        if not kept[i]:
            continue
        for j in range(i + 1, n):
            if not kept[j]:
                continue
            if sim_matrix[i, j] >= threshold:
                # j is a duplicate of i — i has lower chunk_index (kept)
                kept[j] = False
                absorbed_by[j] = i

    # Build output — kept codes get a merged_count
    merge_counts = {i: 0 for i in range(n)}
    for j, i in absorbed_by.items():
        merge_counts[i] += 1

    result = []
    for i, code in enumerate(sorted_codes):
        if kept[i]:
            out = code.copy()
            out["merged_count"] = merge_counts[i]
            result.append(out)

    return result


# -----------------------------------------------------------------------
# Public function 2: cluster_codes
# -----------------------------------------------------------------------

def cluster_codes(
    codes: List[Dict[str, Any]],
    threshold: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    Group codes into semantic clusters without removing any.

    Lower threshold than deduplicate_codes (0.75 default) — we want
    to see related clusters, not just near-duplicates.

    Parameters
    ----------
    codes : list of dicts
    threshold : float
        Similarity threshold for grouping.

    Returns
    -------
    list of cluster dicts, each with:
        "representative"  dict   — the code with the lowest chunk_index
        "members"         list   — all codes in this cluster (incl. representative)
        "cluster_size"    int
        "avg_similarity"  float  — mean pairwise similarity within cluster
    """
    if not codes:
        return []

    _validate_codes(codes)

    if len(codes) == 1:
        return [{
            "representative": codes[0].copy(),
            "members":        [codes[0].copy()],
            "cluster_size":   1,
            "avg_similarity": 1.0,
        }]

    sorted_codes = sorted(codes, key=lambda c: int(c.get("chunk_index", 0)))
    texts        = [_code_to_text(c) for c in sorted_codes]
    embeddings   = _embed(texts)
    sim_matrix   = _cosine_similarity_matrix(embeddings)

    n          = len(sorted_codes)
    assigned   = [-1] * n  # cluster id for each code
    cluster_id = 0

    for i in range(n):
        if assigned[i] != -1:
            continue
        assigned[i] = cluster_id
        for j in range(i + 1, n):
            if assigned[j] != -1:
                continue
            if sim_matrix[i, j] >= threshold:
                assigned[j] = cluster_id
        cluster_id += 1

    # Build cluster dicts
    clusters_map: Dict[int, List[int]] = {}
    for idx, cid in enumerate(assigned):
        clusters_map.setdefault(cid, []).append(idx)

    result = []
    for cid, indices in clusters_map.items():
        members = [sorted_codes[i] for i in indices]
        # Representative = lowest chunk_index (already sorted)
        representative = members[0]

        # Average pairwise similarity within cluster
        if len(indices) == 1:
            avg_sim = 1.0
        else:
            sims = [
                sim_matrix[indices[a], indices[b]]
                for a in range(len(indices))
                for b in range(a + 1, len(indices))
            ]
            avg_sim = float(np.mean(sims)) if sims else 1.0

        result.append({
            "representative": representative.copy(),
            "members":        [m.copy() for m in members],
            "cluster_size":   len(members),
            "avg_similarity": round(avg_sim, 4),
        })

    # Sort by cluster size descending
    result.sort(key=lambda c: c["cluster_size"], reverse=True)
    return result


# -----------------------------------------------------------------------
# Public function 3: compute_similarity_matrix
# -----------------------------------------------------------------------

def compute_similarity_matrix(
    codes: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Compute pairwise cosine similarity matrix for dashboard visualisation.

    Parameters
    ----------
    codes : list of dicts

    Returns
    -------
    pd.DataFrame
        Square matrix indexed and columned by code names.
        Values: cosine similarity in [-1, 1], rounded to 4 dp.
    """
    if not codes:
        return pd.DataFrame()

    _validate_codes(codes)

    names      = [c.get("name", f"code_{i}") for i, c in enumerate(codes)]
    texts      = [_code_to_text(c) for c in codes]
    embeddings = _embed(texts)
    sim_matrix = _cosine_similarity_matrix(embeddings)

    return pd.DataFrame(
        np.round(sim_matrix, 4),
        index=names,
        columns=names,
    )


# -----------------------------------------------------------------------
# Internal validation
# -----------------------------------------------------------------------

def _validate_codes(codes: List[Dict[str, Any]]) -> None:
    """Fail fast on malformed code dicts."""
    for i, code in enumerate(codes):
        if not isinstance(code, dict):
            raise ValueError(
                f"Code at index {i} must be a dict, got {type(code).__name__}"
            )
        if "name" not in code:
            raise ValueError(
                f"Code at index {i} missing required field 'name'. "
                f"Each code must have: name, description, quote, chunk_index."
            )
        if "chunk_index" not in code:
            raise ValueError(
                f"Code '{code.get('name', i)}' missing 'chunk_index'. "
                f"chunk_index is required for De Paoli anti-hallucination."
            )
