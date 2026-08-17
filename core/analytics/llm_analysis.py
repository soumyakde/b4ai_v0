"""
core/analytics/llm_analysis.py

LLM-assisted qualitative analysis engine for Basics4AI.

Features
--------
✅ Swappable providers  : Groq/Llama3.1-70B (free) · Claude Haiku (paid) · Demo/Mock
✅ Run-mode subsetting  : Demo (1+1) · Small (2+2) · Full (all data)
✅ SQLite result cache  : keyed on SHA-256 hash of input texts —
                          zero cost on repeat calls unless data changes
✅ Cache invalidation   : automatic when underlying responses change
✅ Cost estimation      : per-provider pricing table

Provider setup
--------------
Keys are read from Streamlit secrets first, then os.environ as fallback.

Streamlit secrets  (.streamlit/secrets.toml):
    GROQ_API_KEY      = "gsk_..."   # free — console.groq.com
    ANTHROPIC_API_KEY = "sk-..."    # paid  — console.anthropic.com

pip installs:
    pip install groq          # Groq / Llama 3.1
    pip install anthropic     # Claude Haiku (only if using paid tier)

Demo/Mock requires no key and costs $0 — use for UI/pipeline testing.
"""

import hashlib
import json
import sqlite3
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------
_DB_PATH = Path(__file__).resolve().parents[2] / "research.db"

# ---------------------------------------------------------------------
#  Provider pricing  (USD per 1K tokens, updated Mar 2025)
# ---------------------------------------------------------------------
PROVIDER_PRICING: dict[str, dict] = {
    "groq_llama3": {
        "label":         "GPT-OSS 120B via Groq — free tier ✅",
        "input_per_1k":  0.0,
        "output_per_1k": 0.0,
        "model":         "openai/gpt-oss-120b",
        "free":          True,
        "key_name":      "GROQ_API_KEY",
        "llm_client_key": "groq",
    },
    "claude_haiku": {
        "label":         "Claude Haiku 3.5 (Anthropic — cheapest paid)",
        "input_per_1k":  0.0008,
        "output_per_1k": 0.004,
        "model":         "claude-haiku-4-5-20251001",
        "free":          False,
        "key_name":      "ANTHROPIC_API_KEY",
        "llm_client_key": "claude",
    },
}

# ---------------------------------------------------------------------
#  Run modes
# ---------------------------------------------------------------------
RUN_MODES: dict[str, dict] = {
    "demo": {
        "label":        "Demo  — 1 reflection + 1 interview (proves pipeline, ~$0)",
        "n_reflections": 1,
        "n_interviews":  1,
    },
    "small": {
        "label":        "Small — 2 reflections + 2 interviews (minimal cost)",
        "n_reflections": 2,
        "n_interviews":  2,
    },
    "full": {
        "label":        "Full  — all data (production run, requires cost confirmation)",
        "n_reflections": None,   # None = no limit
        "n_interviews":  None,
    },
}

# System prompt for thematic analysis
_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert qualitative researcher assisting with thematic analysis
    of student reflections and interviews about AI literacy concepts.

    Your task:
    1. Identify 2–4 recurring themes in the provided text.
    2. For each theme, provide:
       - A short theme label (3–5 words)
       - A one-sentence description
       - One representative quote from the text (verbatim, under 25 words)
    3. End with a one-paragraph synthesis (3–5 sentences) suitable for a
       research report.

    Respond ONLY in valid JSON with this exact structure:
    {
      "themes": [
        {
          "label": "...",
          "description": "...",
          "quote": "..."
        }
      ],
      "synthesis": "..."
    }
""").strip()

# ---------------------------------------------------------------------
#  API key resolver  (Streamlit secrets → os.environ fallback)
# ---------------------------------------------------------------------
def _get_api_key(key_name: str) -> str:
    """
    Read an API key in priority order:
      1. st.secrets        (Streamlit secrets.toml — production path)
      2. os.environ        (already-loaded env vars, including dotenv)
      3. .env file         (explicit dotenv load — safety net if app.py
                            didn't call load_dotenv() yet)
      4. secrets.toml      (direct parse — Python 3.10 compatible)

    Raises RuntimeError with a clear message if none of the four sources
    has the key.
    """
    # 1. Streamlit secrets
    try:
        import streamlit as st
        val = st.secrets.get(key_name)
        if val:
            return val
    except Exception:
        pass

    # 2. os.environ (covers vars set by load_dotenv in app.py)
    import os
    val = os.environ.get(key_name)
    if val:
        return val

    # 3. Explicit dotenv load (safety net — idempotent if already loaded)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        val = os.environ.get(key_name)
        if val:
            return val
    except ImportError:
        pass

    # 4. Direct secrets.toml parse — no tomllib, works on Python 3.10+
    search = Path(__file__).resolve()
    for _ in range(6):
        candidate = search / ".streamlit" / "secrets.toml"
        if candidate.exists():
            with open(candidate, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith(key_name) and "=" in line:
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
            break
        search = search.parent

    raise RuntimeError(
        f"API key '{key_name}' not found in st.secrets, os.environ, "
        f".env, or .streamlit/secrets.toml.\n"
        f"Add it to your .env file:\n"
        f"    {key_name}=your-key-here\n"
        f"or to .streamlit/secrets.toml:\n"
        f'    {key_name} = "your-key-here"'
    )


def _cache_db() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH)


def _init_cache() -> None:
    """Create the llm_result_cache table if it does not exist."""
    with _cache_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_result_cache (
                input_hash  TEXT PRIMARY KEY,
                provider    TEXT NOT NULL,
                run_mode    TEXT NOT NULL,
                n_texts     INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        conn.commit()


def _compute_hash(texts: list[str]) -> str:
    """SHA-256 of sorted, concatenated texts — deterministic regardless of order."""
    combined = "\n---\n".join(sorted(texts))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _cache_get(input_hash: str) -> dict | None:
    _init_cache()
    with _cache_db() as conn:
        row = conn.execute(
            "SELECT result_json, provider, run_mode, n_texts, created_at "
            "FROM llm_result_cache WHERE input_hash = ?",
            (input_hash,),
        ).fetchone()
    if row:
        return {
            "result":     json.loads(row[0]),
            "provider":   row[1],
            "run_mode":   row[2],
            "n_texts":    row[3],
            "created_at": row[4],
            "from_cache": True,
        }
    return None


def _cache_set(input_hash: str, provider: str, run_mode: str,
               n_texts: int, result: dict) -> None:
    _init_cache()
    with _cache_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO llm_result_cache
               (input_hash, provider, run_mode, n_texts, result_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (input_hash, provider, run_mode, n_texts,
             json.dumps(result), datetime.utcnow().isoformat()),
        )
        conn.commit()


def clear_cache() -> int:
    """Delete all cached results. Returns row count deleted."""
    _init_cache()
    with _cache_db() as conn:
        n = conn.execute("DELETE FROM llm_result_cache").rowcount
        conn.commit()
    return n


def get_cache_entries() -> list[dict]:
    """Return summary of all cached entries (for display in dashboard)."""
    _init_cache()
    with _cache_db() as conn:
        rows = conn.execute(
            "SELECT input_hash, provider, run_mode, n_texts, created_at "
            "FROM llm_result_cache ORDER BY created_at DESC"
        ).fetchall()
    return [
        {
            "hash":       r[0][:12] + "…",
            "provider":   r[1],
            "run_mode":   r[2],
            "n_texts":    r[3],
            "created_at": r[4],
        }
        for r in rows
    ]

# ---------------------------------------------------------------------
#  Cost estimation
# ---------------------------------------------------------------------
_TOKENS_PER_WORD = 1.35
_AVG_WORDS       = 120
_OUTPUT_TOKENS   = 500
_SYSTEM_TOKENS   = len(_SYSTEM_PROMPT.split()) + 50


def estimate_cost(n_texts: int, provider_key: str) -> dict:
    pricing = PROVIDER_PRICING.get(provider_key, PROVIDER_PRICING["demo_mock"])
    input_per_call  = int(_AVG_WORDS * _TOKENS_PER_WORD) + _SYSTEM_TOKENS
    total_input     = input_per_call * n_texts
    total_output    = _OUTPUT_TOKENS * n_texts
    cost = (
        (total_input  / 1000) * pricing["input_per_1k"]
        + (total_output / 1000) * pricing["output_per_1k"]
    )
    return {
        "n_texts":       n_texts,
        "total_input":   total_input,
        "total_output":  total_output,
        "est_cost_usd":  round(cost, 5),
        "free":          pricing["free"],
        "provider_label": pricing["label"],
    }

# ---------------------------------------------------------------------
#  Provider call — unified via llm_clients
# ---------------------------------------------------------------------
def _call_provider(provider_key: str, text: str) -> dict:
    """
    Route to the correct LLM via llm_clients.call_model.
    Raises ValueError on unknown provider_key.
    """
    from core.analytics.llm.llm_clients import call_model

    pinfo      = PROVIDER_PRICING[provider_key]
    client_key = pinfo["llm_client_key"]   # "groq" | "claude"
    model_id   = pinfo["model"]

    result = call_model(
        model=client_key,
        prompt=text,
        system=_SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=600,
        model_id=model_id,
    )

    if result["error"]:
        raise RuntimeError(result["error"])

    raw = result["text"].strip()
    return json.loads(raw)

# ---------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------
def run_analysis(
    reflections: list[str],
    interviews:  list[str],
    provider_key: str = "groq_llama3",   # Groq is the default
    run_mode_key: str = "demo",
    force_rerun:  bool = False,
) -> dict[str, Any]:
    """
    Run LLM-assisted thematic analysis.

    Parameters
    ----------
    reflections   : list of raw reflection text strings
    interviews    : list of raw interview text strings
    provider_key  : one of PROVIDER_PRICING keys
    run_mode_key  : one of RUN_MODES keys
    force_rerun   : if True, bypass cache and always call the LLM

    Returns
    -------
    dict with keys: themes, synthesis, from_cache, provider, run_mode,
                    n_reflections_used, n_interviews_used, input_hash
    """
    # 1. Subset data according to run mode
    mode = RUN_MODES[run_mode_key]
    n_r  = mode["n_reflections"]
    n_i  = mode["n_interviews"]
    subset_reflections = reflections[:n_r] if n_r else reflections
    subset_interviews  = interviews[:n_i]  if n_i else interviews
    all_texts = subset_reflections + subset_interviews

    if not all_texts:
        return {"error": "No text data available for analysis."}

    # 2. Check cache
    input_hash = _compute_hash(all_texts)
    if not force_rerun:
        cached = _cache_get(input_hash)
        if cached:
            cached["n_reflections_used"] = len(subset_reflections)
            cached["n_interviews_used"]  = len(subset_interviews)
            cached["input_hash"]         = input_hash
            return cached

    # 3. Build combined prompt text
    combined_text = (
        "## Student Reflections\n\n"
        + "\n\n---\n\n".join(
            f"[Reflection {i+1}]\n{t}" for i, t in enumerate(subset_reflections)
        )
        + "\n\n## Interviews\n\n"
        + "\n\n---\n\n".join(
            f"[Interview {i+1}]\n{t}" for i, t in enumerate(subset_interviews)
        )
    )

    # 4. Call LLM via unified provider routing
    try:
        result = _call_provider(provider_key, combined_text)
    except json.JSONDecodeError as e:
        return {"error": f"LLM returned non-JSON output: {e}"}
    except Exception as e:
        return {"error": f"LLM call failed ({provider_key}): {e}"}

    # 5. Store in cache
    _cache_set(input_hash, provider_key, run_mode_key, len(all_texts), result)

    return {
        **result,
        "from_cache":           False,
        "provider":             provider_key,
        "run_mode":             run_mode_key,
        "n_reflections_used":   len(subset_reflections),
        "n_interviews_used":    len(subset_interviews),
        "input_hash":           input_hash,
    }
