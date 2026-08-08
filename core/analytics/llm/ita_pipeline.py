# core/analytics/llm/ita_pipeline.py
"""
Inductive Thematic Analysis Pipeline
=====================================
Orchestrates Braun & Clarke (2006) six phases using LLM APIs.

Design:
    - Phase-by-phase execution with DB persistence after each phase
    - Sequential model runs (one model at a time)
    - Resumable: re-running a completed phase overwrites its results
    - All prompts follow De Paoli (2024) anti-hallucination conventions
    - Fixed system prompt injects Basics4AI curriculum context

Fixed system prompt (Phase 0 context):
    Injected into every LLM call. Describes the Basics4AI programme,
    participants, and the inductive TA methodology being followed.

Phases:
    1  Familiarise — chunk transcripts into ~2500-token pieces
    2  Generate codes — inductively infer 3 codes per chunk
    2b Deduplicate — embedding-based removal of redundant codes
    3  Search themes — group codes into N themes
    4  Review themes — re-run Phase 3 at higher temperature
    5  Define themes — name and summarise final themes
    6  Write report  — produce narrative analysis report

Public API:
-----------
run_phase1(transcripts, chunk_size=2500)
run_phase2(chunks, model, temperature, n_codes=3, run_id=None)
run_phase2_dedup(codes, threshold=0.85)
run_phase3(codes, model, temperature, n_themes, run_id=None)
run_phase4(codes, model, temperature=1.0, n_themes=None, run_id=None)
run_phase5(themes, codes, model, temperature, run_id=None)
run_phase6(themes, codes, model, temperature, run_id=None)

save_phase_result(run_id, phase, output)
load_phase_result(run_id, phase)
create_run(model, temperature, source_type, created_by)
get_run(run_id)
list_runs(created_by=None)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import sqlite3
import json
import uuid
import re
import math

# -----------------------------------------------------------------------
# DB path
# -----------------------------------------------------------------------
def _find_db() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "responses.db"
        if candidate.exists():
            return candidate
    return here.parents[min(3, len(here.parents)-1)] / "responses.db"

_DB_PATH = _find_db()


def _get_conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or _DB_PATH)


# -----------------------------------------------------------------------
# Lazy import helper — works both in project (core.analytics) and in
# test environments where the module path may not be on sys.path
# -----------------------------------------------------------------------

def _lazy_import(module_name: str, attr: str):
    """
    Import attr from module_name.
    Tries standard import first; falls back to file-based loading
    by searching parent directories for the module file.
    """
    import importlib, importlib.util as _ilu, sys as _sys

    # Try standard import (works in full project environment)
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attr)
    except (ImportError, ModuleNotFoundError):
        pass

    # File-based fallback: search parent dirs for core/analytics/llm/X.py
    rel = module_name.replace(".", "/") + ".py"
    search_roots = list(Path(__file__).resolve().parents)
    # Also search sys.path roots (covers test runner working directories)
    for sp in _sys.path:
        p = Path(sp)
        if p not in search_roots:
            search_roots.append(p)

    candidate = next(
        (root / rel for root in search_roots
         if (root / rel).exists()), None
    )
    if candidate is None:
        # Last resort: look for just the filename in same directory
        fname = module_name.split(".")[-1] + ".py"
        candidate = next(
            (root / fname for root in search_roots
             if (root / fname).exists()), None
        )

    if candidate is None:
        raise ImportError(
            f"Cannot find module '{module_name}'. "
            f"Ensure core/analytics/llm/ is on the Python path."
        )

    spec = _ilu.spec_from_file_location(module_name, candidate)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


# -----------------------------------------------------------------------
# Schema initialisation
# -----------------------------------------------------------------------

def _init_schema(db_path: Optional[Path] = None) -> None:
    """
    Create ita_runs and ita_results tables if not present.
    Also adds any missing columns to existing tables (safe migration).
    """
    conn = _get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ita_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT UNIQUE NOT NULL,
            created_by  TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            model       TEXT NOT NULL,
            temperature REAL NOT NULL,
            source_type TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'created',
            phase_reached INTEGER DEFAULT 0,
            notes       TEXT
        );

        CREATE TABLE IF NOT EXISTS ita_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL,
            phase       INTEGER NOT NULL,
            output_json TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            UNIQUE(run_id, phase)
        );
    """)

    # Migration 1: add notes column to ita_runs if missing
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(ita_runs)").fetchall()
    }
    if "notes" not in existing_cols:
        conn.execute("ALTER TABLE ita_runs ADD COLUMN notes TEXT")

    # Migration 3: add cohort_scope column (2026-08-08) — records which
    # cohort(s) this run was filtered to, e.g. "All cohorts" or a
    # comma-joined list, so Report Generation can show it in the run picker.
    if "cohort_scope" not in existing_cols:
        conn.execute("ALTER TABLE ita_runs ADD COLUMN cohort_scope TEXT")

    # Migration 2: ensure UNIQUE index on ita_results(run_id, phase)
    # Required for ON CONFLICT upsert. Table may exist without this index
    # if created before this migration was added.
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_ita_results_run_phase
           ON ita_results(run_id, phase)"""
    )

    conn.commit()
    conn.close()


# -----------------------------------------------------------------------
# Fixed system prompt — Basics4AI curriculum context
# -----------------------------------------------------------------------

SYSTEM_PROMPT = """You are a qualitative research assistant supporting an inductive thematic analysis of data from the Basics4AI programme — a 7-module curriculum teaching AI literacy to young people aged 10–14 years in informal learning settings (after-school programs, community centres, libraries). The curriculum uses plugged and unplugged games and activities grounded in cognitive- and context-based learning approaches. Modules cover: what AI is, how AI learns, how AI works, goal-based problem-solving, natural vs. artificial agents' problem-solving processes, real-world problem-solving with constraints and uncertainties, natural language processing, and AI in society. Data sources include semi-structured interviews and end-of-module reflection notes collected from participants aged 10–14 years. You are performing inductive thematic analysis following Braun and Clarke (2006) without any pre-existing coding framework. All codes and themes must be grounded strictly in the data provided. Ensure all sources for codes generated are saved for the ability to traceback to the usernames to whom the quotes are to be attributed to, for transparency purposes. Do not introduce concepts not present in the data. Note: Research with this age group commonly finds that young people hold misconceptions about AI — for example, conflating AI with robots, attributing human-like emotions or consciousness to AI, or believing AI learns entirely on its own without human input. You are not directed to code for these specifically, but this contextual awareness may help you interpret ambiguous participant language accurately."""


# -----------------------------------------------------------------------
# Token estimation (rough: 1 token ≈ 4 chars for English text)
# -----------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# -----------------------------------------------------------------------
# Phase 1: Chunk transcripts
# -----------------------------------------------------------------------

def run_phase1(
    transcripts: List[Dict[str, Any]],
    chunk_size: int = 2500,
) -> List[Dict[str, Any]]:
    """
    Phase 1 — Familiarise with data.

    Splits each transcript into chunks of approximately chunk_size tokens.
    Following De Paoli (2024): chunks of ~2500 tokens prevent hitting
    model context limits in Phase 2.

    Parameters
    ----------
    transcripts : list of dicts
        Each dict: {"participant_id": str, "content": str, ...}
    chunk_size : int
        Target token size per chunk. Default 2500.

    Returns
    -------
    list of dicts:
        {
          "chunk_index":     int   global index across all transcripts
          "participant_id":  str
          "content":         str   chunk text
          "char_count":      int
          "token_estimate":  int
          "source_chunk":    int   chunk number within this transcript
        }
    """
    if not transcripts:
        return []

    chunks = []
    global_index = 0

    for transcript in transcripts:
        pid     = transcript.get("participant_id", "unknown")
        content = str(transcript.get("content", "")).strip()

        if not content:
            continue

        # Split into sentences first (preserve natural boundaries)
        sentences = re.split(r"(?<=[.!?])\s+", content)

        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = _estimate_tokens(sentence)

            # If single sentence exceeds chunk_size, split by words
            if sent_tokens > chunk_size:
                words      = sentence.split()
                word_chunk = []
                word_tokens = 0
                for word in words:
                    wt = _estimate_tokens(word)
                    if word_tokens + wt > chunk_size and word_chunk:
                        text = " ".join(word_chunk)
                        chunks.append(_make_chunk(
                            global_index, pid, text,
                            len(chunks)  # source_chunk
                        ))
                        global_index += 1
                        word_chunk  = [word]
                        word_tokens = wt
                    else:
                        word_chunk.append(word)
                        word_tokens += wt
                if word_chunk:
                    current_chunk.extend(word_chunk)
                    current_tokens += word_tokens
                continue

            if current_tokens + sent_tokens > chunk_size and current_chunk:
                text = " ".join(current_chunk)
                chunks.append(_make_chunk(
                    global_index, pid, text,
                    len([c for c in chunks if c["participant_id"] == pid])
                ))
                global_index  += 1
                current_chunk  = [sentence]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sent_tokens

        # Flush remaining
        if current_chunk:
            text = " ".join(current_chunk)
            chunks.append(_make_chunk(
                global_index, pid, text,
                len([c for c in chunks if c["participant_id"] == pid])
            ))
            global_index += 1

    return chunks


def _make_chunk(
    chunk_index: int,
    participant_id: str,
    text: str,
    source_chunk: int,
) -> Dict[str, Any]:
    return {
        "chunk_index":    chunk_index,
        "participant_id": participant_id,
        "content":        text,
        "char_count":     len(text),
        "token_estimate": _estimate_tokens(text),
        "source_chunk":   source_chunk,
    }


# -----------------------------------------------------------------------
# Phase 2: Generate initial codes
# -----------------------------------------------------------------------

_PHASE2_PROMPT = """\
Read the following interview/reflection excerpt carefully.
This excerpt is from participant: {participant_id}

Identify exactly {n_codes} of the most relevant codes in the text.
For each code:
1. Provide a name in no more than 3 words
2. Provide a meaningful description in no more than 4 sentences, written as a single paragraph with no line breaks
3. Include one meaningful quote from the text, no longer than 2 sentences — the quote must be attributed to the participant identified above

The excerpt index is: {chunk_index}

Format your response as valid JSON only, with this exact structure:
{{
  "codes": [
    {{
      "name": "code name here",
      "description": "description here",
      "quote": "verbatim quote from text here",
      "participant_id": "{participant_id}",
      "chunk_index": {chunk_index}
    }}
  ]
}}

Do not include any text before or after the JSON.
Do not use markdown code fences.

Excerpt:
{text}"""


def run_phase2(
    chunks: List[Dict[str, Any]],
    model: str,
    temperature: float,
    n_codes: int = 3,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Phase 2 — Generate initial codes from chunks.

    Calls the LLM once per chunk. Each call asks for n_codes codes
    formatted as JSON. Following De Paoli (2024): chunk_index is
    included in the prompt to prevent hallucination.

    Parameters
    ----------
    chunks : list of dicts from run_phase1()
    model : str  "claude"|"gemini"|"gpt"
    temperature : float
    n_codes : int  codes per chunk, default 3 (De Paoli limit)
    run_id : str or None  if provided, saves results to DB
    db_path : Path or None

    Returns
    -------
    dict:
        codes         list of code dicts (all chunks combined)
        n_chunks      int
        n_codes_raw   int  before deduplication
        errors        list of chunk-level error strings
        model         str
        temperature   float
        phase         int  2
    """
    call_model = _lazy_import("core.analytics.llm.llm_clients", "call_model")

    result = {
        "phase":       2,
        "model":       model,
        "temperature": temperature,
        "codes":       [],
        "n_chunks":    len(chunks),
        "n_codes_raw": 0,
        "errors":      [],
    }

    all_codes = []

    for chunk in chunks:
        prompt = _PHASE2_PROMPT.format(
            n_codes=n_codes,
            chunk_index=chunk["chunk_index"],
            participant_id=chunk.get("participant_id", "unknown"),
            text=chunk["content"],
        )

        response = call_model(
            model, prompt,
            system=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=3000,   # raised from 1500 — Llama verbosity requires headroom
        )

        if response["error"]:
            result["errors"].append(
                f"Chunk {chunk['chunk_index']}: {response['error']}"
            )
            continue

        parsed = _parse_json_response(response["text"])
        if parsed is None:
            result["errors"].append(
                f"Chunk {chunk['chunk_index']}: JSON parse failed. "
                f"Raw: {response['text'][:200]}"
            )
            continue

        codes = parsed.get("codes", [])
        for code in codes:
            # Ensure chunk_index is set (anti-hallucination)
            code["chunk_index"]    = chunk["chunk_index"]
            code["participant_id"] = chunk["participant_id"]
            code["model"]          = model
            all_codes.append(code)

    result["codes"]       = all_codes
    result["n_codes_raw"] = len(all_codes)

    if run_id:
        save_phase_result(run_id, 2, result, db_path)

    return result


# -----------------------------------------------------------------------
# Phase 2b: Deduplicate codes
# -----------------------------------------------------------------------

def run_phase2_dedup(
    codes: List[Dict[str, Any]],
    threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Phase 2b — Deduplicate codes using embedding similarity.

    Parameters
    ----------
    codes : list of code dicts from run_phase2()
    threshold : float  default 0.85

    Returns
    -------
    dict:
        codes_dedup      list  deduplicated codes
        n_before         int
        n_after          int
        n_removed        int
        threshold        float
    """
    deduplicate_codes = _lazy_import(
        "core.analytics.llm.deduplicator", "deduplicate_codes"
    )

    n_before = len(codes)
    deduped  = deduplicate_codes(codes, threshold=threshold)
    n_after  = len(deduped)

    # Compute removed codes so the dashboard can display them.
    # A code is "removed" if its (chunk_index, name) key is not in the
    # deduplicated set.  chunk_index alone is not unique (multiple codes
    # per chunk), but (chunk_index, name) is stable across serialisation.
    kept_keys = {
        (c.get("chunk_index"), c.get("name", "").strip().lower())
        for c in deduped
    }
    removed_codes = [
        c for c in codes
        if (c.get("chunk_index"), c.get("name", "").strip().lower())
        not in kept_keys
    ]

    return {
        "codes_dedup":   deduped,
        "removed_codes": removed_codes,   # NEW — list of removed code dicts
        "n_before":      n_before,
        "n_after":       n_after,
        "n_removed":     n_before - n_after,
        "threshold":     threshold,
    }


# -----------------------------------------------------------------------
# Phase 3: Search for themes
# -----------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 prompt — maps to De Paoli (2024) Phase 4 ("group topics into themes")
#
# De Paoli methodology note
# ─────────────────────────
# De Paoli's Phase 3 asks the LLM to identify UNIQUE topics from a flat list
# (deduplication at the code level). In this implementation, deduplication is
# handled upstream in Phase 2b using sentence-transformers cosine similarity
# (threshold=0.85 by default), which is more reliable than asking an LLM to
# judge uniqueness.  Phase 3 here therefore begins directly from De Paoli's
# Phase 4: grouping the deduplicated codes into themes.
#
# Traceability
# ────────────
# Each theme returns code_indices (chunk-level indices linking back to the
# original transcript chunks) and quotes (verbatim participant text) so every
# theme can be traced to source data — fulfilling De Paoli's anti-hallucination
# requirement.
# ─────────────────────────────────────────────────────────────────────────────

_PHASE3_PROMPT = """Below is a list of codes identified from interview and reflection data.
Each code has a number (its index), a name, a description, a supporting quote, and the participant username the quote is attributed to.

Your task:
Determine how all the codes can be grouped into exactly {n_themes} significant themes.
A code may belong to more than one group.
For each group provide:
  - a name (maximum 5 words)
  - a clear description (2-3 sentences, single paragraph, no line breaks)
  - the list of code indices that belong to this group
  - up to 3 representative attributed quotes drawn directly from the codes in the group,
    each with the participant_id of the person who said it

Format your response as valid JSON only:
{{
  "themes": [
    {{
      "name": "theme name here",
      "description": "theme description here",
      "code_indices": [0, 3, 7],
      "attributed_quotes": [
        {{"participant_id": "username_here", "quote": "verbatim participant quote"}}
      ]
    }}
  ]
}}

Do not include any text before or after the JSON.
Do not use markdown code fences.
Quotes must be taken verbatim from the participant quotes listed below — do not paraphrase or invent quotes.
participant_id values must be copied exactly from the codes list below — do not guess or invent usernames.

Codes:
{codes_list}"""


def run_phase3(
    codes: List[Dict[str, Any]],
    model: str,
    temperature: float,
    n_themes: int,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
    _max_tokens: int = 4000,
) -> Dict[str, Any]:
    """
    Phase 3 — Search for themes.

    Groups codes into n_themes using a single LLM call.
    Following De Paoli: codes are passed with their indices to
    prevent hallucination.

    Parameters
    ----------
    codes : list of code dicts (deduplicated from Phase 2b)
    model : str
    temperature : float
    n_themes : int  target number of themes
    run_id : str or None
    db_path : Path or None

    Returns
    -------
    dict:
        themes       list of theme dicts
        n_themes     int
        n_codes_in   int
        model        str
        temperature  float
        phase        int  3
        error        str|None
    """
    call_model = _lazy_import("core.analytics.llm.llm_clients", "call_model")

    result = {
        "phase":       3,
        "model":       model,
        "temperature": temperature,
        "n_codes_in":  len(codes),
        "n_themes":    n_themes,
        "themes":      [],
        "error":       None,
    }

    if not codes:
        result["error"] = "No codes provided to Phase 3."
        return result

    # Build codes list string with index + participant_id + quote for attribution
    codes_lines = []
    for code in codes:
        idx   = code.get("chunk_index", 0)
        name  = code.get("name", "")
        desc  = code.get("description", "")
        quote = code.get("quote", "")
        pid   = code.get("participant_id", "unknown")
        line  = f"{idx}: '{name}': {desc}  [participant: {pid}]"
        if quote:
            line += f'  [Quote: "{quote}"]'
        codes_lines.append(line)
    codes_list = "\n".join(codes_lines)

    prompt = _PHASE3_PROMPT.format(
        n_themes=n_themes,
        codes_list=codes_list,
    )

    response = call_model(
        model, prompt,
        system=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=_max_tokens,
    )

    if response["error"]:
        result["error"] = response["error"]
        return result

    parsed = _parse_json_response(response["text"])
    if parsed is None:
        result["error"] = (
            f"JSON parse failed. Raw: {response['text'][:300]}"
        )
        return result

    themes = parsed.get("themes", [])
    for i, t in enumerate(themes):
        t["model"]       = model
        t["temperature"] = temperature
        t["theme_index"] = i

    result["themes"]   = themes
    result["n_themes"] = len(themes)

    if run_id:
        save_phase_result(run_id, 3, result, db_path)

    return result


# -----------------------------------------------------------------------
# Phase 4: Review themes
# -----------------------------------------------------------------------

def run_phase4(
    codes: List[Dict[str, Any]],
    model: str,
    temperature: float = 1.0,
    n_themes: Optional[int] = None,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Phase 4 — Review themes at higher temperature.

    Re-runs Phase 3 with elevated temperature to check consistency.
    De Paoli (2024) suggests T=1.0 for Phase 4 to reveal
    overlooked themes and test validity.

    Parameters
    ----------
    codes : list of code dicts (same as Phase 3 input)
    model : str
    temperature : float  default 1.0 (De Paoli recommendation)
    n_themes : int or None  if None, uses same count as Phase 3
    run_id : str or None
    db_path : Path or None

    Returns
    -------
    Same structure as run_phase3(), with phase=4.
    """
    if n_themes is None:
        # Data-driven default: scale with the number of codes, bounded 5–13.
        # De Paoli (2024) uses 11 as a fixed reference; this formula adapts
        # to smaller or larger code lists. The UI slider lets researchers
        # override to any target value, including De Paoli's recommended 11.
        n_themes = max(5, min(13, len(codes) // 3))

    result = run_phase3(
        codes, model, temperature, n_themes,
        run_id=None, db_path=None,
        _max_tokens=6000,   # Phase 4 at T=1.0 generates longer responses
    )
    result["phase"] = 4

    if run_id:
        save_phase_result(run_id, 4, result, db_path)

    return result


# -----------------------------------------------------------------------
# Phase 5: Define and name themes
# -----------------------------------------------------------------------

_PHASE5_PROMPT = """\
Below is a list of topics (codes) that belong to a theme.
Each topic has a name, a description, and a supporting quote.

Using all the topics in the list, provide:
1. A summary (exactly 2 sentences) capturing what these topics mean together
2. A name for the theme (maximum 5 words)

Format your response as valid JSON only:
{{
  "name": "theme name here",
  "summary": "two sentence summary here"
}}

Do not include any text before or after the JSON.
Do not use markdown code fences.

Topics:
{topics_list}"""


def run_phase5(
    themes: List[Dict[str, Any]],
    codes: List[Dict[str, Any]],
    model: str,
    temperature: float,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Phase 5 — Define and name themes.

    For each theme from Phase 3/4, passes its underlying codes
    (with quotes) to the model for a 2-sentence summary and name.
    Following De Paoli: model has no memory of previous phases,
    so the full code context is re-provided.

    Parameters
    ----------
    themes : list of theme dicts from run_phase3() or run_phase4()
    codes  : list of code dicts (for quote lookup)
    model  : str
    temperature : float
    run_id : str or None
    db_path : Path or None

    Returns
    -------
    dict:
        themes_defined  list of theme dicts with name + summary added
        n_themes        int
        model           str
        temperature     float
        phase           int  5
        errors          list
    """
    call_model = _lazy_import("core.analytics.llm.llm_clients", "call_model")

    result = {
        "phase":          5,
        "model":          model,
        "temperature":    temperature,
        "themes_defined": [],
        "n_themes":       len(themes),
        "errors":         [],
    }

    # Build code index for lookup
    code_index = {c.get("chunk_index", i): c for i, c in enumerate(codes)}

    for theme in themes:
        theme_codes = _get_theme_codes(theme, codes)
        if not theme_codes:
            result["errors"].append(
                f"Theme '{theme.get('name','')}': no codes found."
            )
            continue

        # Build topics list with name, description, quote
        topics_lines = []
        for c in theme_codes:
            name  = c.get("name", "")
            desc  = c.get("description", "")
            quote = c.get("quote", "")
            topics_lines.append(
                f"- Name: {name}\n  Description: {desc}\n  Quote: \"{quote}\""
            )
        topics_list = "\n".join(topics_lines)

        prompt = _PHASE5_PROMPT.format(topics_list=topics_list)

        response = call_model(
            model, prompt,
            system=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=500,
        )

        if response["error"]:
            result["errors"].append(
                f"Theme '{theme.get('name','')}': {response['error']}"
            )
            defined_theme = theme.copy()
            defined_theme["summary"] = None
            result["themes_defined"].append(defined_theme)
            continue

        parsed = _parse_json_response(response["text"])
        if parsed is None:
            result["errors"].append(
                f"Theme '{theme.get('name','')}': JSON parse failed."
            )
            defined_theme = theme.copy()
            defined_theme["summary"] = None
            result["themes_defined"].append(defined_theme)
            continue

        defined_theme = theme.copy()
        defined_theme["name"]    = parsed.get("name", theme.get("name",""))
        defined_theme["summary"] = parsed.get("summary", "")
        defined_theme["model"]   = model
        result["themes_defined"].append(defined_theme)

    if run_id:
        save_phase_result(run_id, 5, result, db_path)

    return result


# -----------------------------------------------------------------------
# Phase 6: Write report
# -----------------------------------------------------------------------

_PHASE6_PROMPT = """\
You are writing the results section of a qualitative research report.
The analysis used inductive thematic analysis following Braun and \
Clarke (2006), applied to interview and reflection data from the \
Basics4AI AI literacy programme for young people aged 10–14 years.

Below are the final themes identified, each with a name, a summary, \
and the codes that compose it.

Write a coherent narrative report of approximately 400–600 words that:
1. Introduces the thematic analysis findings
2. Discusses each theme with its key insights
3. Includes relevant quotes from participants where available
4. Concludes with an overall interpretation of what the themes reveal \
about participants' experiences of the Basics4AI programme

Write in academic English suitable for a social science journal.
Do not use bullet points — write in continuous prose paragraphs.

Themes:
{themes_text}"""


def run_phase6(
    themes: List[Dict[str, Any]],
    codes: List[Dict[str, Any]],
    model: str,
    temperature: float,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Phase 6 — Produce the report.

    Generates a narrative academic report from the final themes.

    Parameters
    ----------
    themes : list of theme dicts from run_phase5()
    codes  : list of code dicts (for quotes)
    model  : str
    temperature : float
    run_id : str or None
    db_path : Path or None

    Returns
    -------
    dict:
        report_text  str   narrative report
        n_themes     int
        model        str
        temperature  float
        phase        int   6
        error        str|None
    """
    call_model = _lazy_import("core.analytics.llm.llm_clients", "call_model")

    result = {
        "phase":       6,
        "model":       model,
        "temperature": temperature,
        "n_themes":    len(themes),
        "report_text": None,
        "error":       None,
    }

    if not themes:
        result["error"] = "No themes provided to Phase 6."
        return result

    # Build themes text with codes and quotes
    themes_lines = []
    for t in themes:
        name    = t.get("name", "Unnamed theme")
        summary = t.get("summary", t.get("description", ""))
        t_codes = _get_theme_codes(t, codes)
        quotes  = [
            c.get("quote","") for c in t_codes if c.get("quote","").strip()
        ][:3]  # max 3 quotes per theme

        themes_lines.append(f"\nTheme: {name}")
        themes_lines.append(f"Summary: {summary}")
        if quotes:
            themes_lines.append("Supporting quotes:")
            for q in quotes:
                themes_lines.append(f'  - "{q}"')

    themes_text = "\n".join(themes_lines)
    prompt      = _PHASE6_PROMPT.format(themes_text=themes_text)

    response = call_model(
        model, prompt,
        system=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=2000,
    )

    if response["error"]:
        result["error"] = response["error"]
        return result

    result["report_text"] = response["text"].strip()

    if run_id:
        save_phase_result(run_id, 6, result, db_path)

    return result


# -----------------------------------------------------------------------
# Run management
# -----------------------------------------------------------------------

def create_run(
    model: str,
    temperature: float,
    source_type: str,
    created_by: str,
    notes: str = "",
    db_path: Optional[Path] = None,
    cohort_scope: str = "All cohorts",
) -> str:
    """
    Create a new ITA run record in the DB.

    Parameters
    ----------
    cohort_scope : str — "All cohorts" if no cohort filter was applied,
        otherwise a comma-joined list of the cohort(s) this run was
        scoped to. Shown in Report Generation's run picker so runs from
        different cohorts can be told apart and compared.

    Returns
    -------
    str  run_id (UUID)
    """
    _init_schema(db_path)
    run_id = str(uuid.uuid4())
    now    = datetime.utcnow().isoformat()
    conn   = _get_conn(db_path)
    conn.execute(
        """INSERT INTO ita_runs
           (run_id, created_by, created_at, model, temperature,
            source_type, status, phase_reached, notes, cohort_scope)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (run_id, created_by, now, model, temperature,
         source_type, "created", 0, notes, cohort_scope)
    )
    conn.commit()
    conn.close()
    return run_id


def save_phase_result(
    run_id: str,
    phase: int,
    output: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> None:
    """
    Save a phase result to ita_results. Overwrites if already exists.
    Also updates phase_reached in ita_runs.
    """
    _init_schema(db_path)
    now  = datetime.utcnow().isoformat()
    conn = _get_conn(db_path)

    # Serialise — drop non-serialisable keys
    safe = _make_serialisable(output)

    conn.execute(
        """INSERT INTO ita_results (run_id, phase, output_json, created_at)
           VALUES (?,?,?,?)
           ON CONFLICT(run_id, phase) DO UPDATE SET
               output_json=excluded.output_json,
               created_at=excluded.created_at""",
        (run_id, phase, json.dumps(safe), now)
    )
    conn.execute(
        """UPDATE ita_runs
           SET phase_reached = MAX(phase_reached, ?),
               status = CASE WHEN ? = 6 THEN 'complete' ELSE 'running' END
           WHERE run_id = ?""",
        (phase, phase, run_id)
    )
    conn.commit()
    conn.close()


def load_phase_result(
    run_id: str,
    phase: int,
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load a saved phase result from the DB.

    Returns None if not found.
    """
    _init_schema(db_path)
    conn = _get_conn(db_path)
    row  = conn.execute(
        "SELECT output_json FROM ita_results WHERE run_id=? AND phase=?",
        (run_id, phase)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return json.loads(row[0])


def get_run(
    run_id: str,
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return run metadata dict or None if not found."""
    _init_schema(db_path)
    conn = _get_conn(db_path)
    row  = conn.execute(
        """SELECT run_id, created_by, created_at, model, temperature,
                  source_type, status, phase_reached, notes
           FROM ita_runs WHERE run_id=?""",
        (run_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(zip(
        ["run_id","created_by","created_at","model","temperature",
         "source_type","status","phase_reached","notes"], row
    ))


def list_runs(
    created_by: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return list of run metadata dicts, newest first."""
    _init_schema(db_path)
    conn = _get_conn(db_path)
    if created_by:
        rows = conn.execute(
            """SELECT run_id, created_by, created_at, model, temperature,
                      source_type, status, phase_reached, notes, cohort_scope
               FROM ita_runs WHERE created_by=?
               ORDER BY created_at DESC""",
            (created_by,)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT run_id, created_by, created_at, model, temperature,
                      source_type, status, phase_reached, notes, cohort_scope
               FROM ita_runs ORDER BY created_at DESC"""
        ).fetchall()
    conn.close()
    cols = ["run_id","created_by","created_at","model","temperature",
            "source_type","status","phase_reached","notes","cohort_scope"]
    return [dict(zip(cols, r)) for r in rows]


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _sanitise_json_strings(text: str) -> str:
    """
    Replace literal newlines and carriage returns inside JSON string values
    with a single space.

    Llama 3.x interprets prompt instructions like "4 lines" literally and
    embeds real \n characters inside JSON string values. Python's json.loads
    correctly rejects these as invalid control characters (RFC 8259 §7).
    The truncation-repair code also fails on them because the string-tracking
    loop loses sync at the bare newline.

    This sanitiser runs before any parse attempt and is a no-op on clean JSON.
    """
    result  = []
    in_str  = False
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            result.append(ch)
            continue
        if ch == "\\" and in_str:
            escaped = True
            result.append(ch)
            continue
        if ch == '"':
            in_str = not in_str
            result.append(ch)
            continue
        if in_str and ch in ("\n", "\r"):
            result.append(" ")   # strip literal newline — keep string on one line
            continue
        result.append(ch)
    return "".join(result)


def _parse_json_response(text: str) -> Optional[Dict]:
    """
    Parse JSON from LLM response text.
    Handles markdown code fences, leading/trailing text,
    and truncated responses (model hit max_tokens mid-JSON).
    Also handles literal newlines inside string values (Llama quirk).
    """
    if not text:
        return None

    # Pre-process: remove literal newlines inside JSON strings (Llama 3.x quirk)
    text = _sanitise_json_strings(text)

    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.strip()

    # Find first { and last }
    start = text.find("{")
    if start == -1:
        return None

    end = text.rfind("}")

    # Try clean parse first
    if end != -1:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass

    # Truncation repair: response was cut off before closing braces.
    # Count unclosed braces and arrays, then append the required closers.
    fragment = text[start:]
    depth_brace  = 0
    depth_bracket = 0
    in_string = False
    escape    = False

    for ch in fragment:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1

    # Trim trailing incomplete string or comma
    repaired = fragment.rstrip().rstrip(",").rstrip()

    # Close any open string
    if in_string:
        repaired += '"'

    # Close arrays then objects in reverse order
    repaired += "]" * max(0, depth_bracket)
    repaired += "}" * max(0, depth_brace)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _get_theme_codes(
    theme: Dict[str, Any],
    codes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Retrieve the codes belonging to a theme.
    Uses code_indices if present, otherwise returns all codes.
    """
    indices = theme.get("code_indices")
    if indices is None:
        return codes

    index_set = set(indices)
    return [
        c for c in codes
        if c.get("chunk_index") in index_set
    ]


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert non-JSON-serialisable objects."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()
                if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(i) for i in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)
