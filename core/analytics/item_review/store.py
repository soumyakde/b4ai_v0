"""
core/analytics/item_review/store.py

SQLite persistence for the Item Review feature (Method 3). Mirrors the cpi_store / ita_pipeline
pattern: tables live in responses.db, created idempotently (CREATE TABLE IF NOT EXISTS), no Streamlit.

Tables
------
item_llm_runs      one row per batch/single LLM run (model, prompt_version, cost, who ran it)
item_llm_results   one row per LLM call result (step 'blind' or 'key', repeat index, parsed fields,
                   raw response, tokens). Versioned by model + prompt_version + item_hash.
item_human_ratings one row per (reviewer, item version): the reviewer's BLIND rating (immutable once
                   locked), and, only after the LLM suggestion is revealed, their final decision.

Reviewer-first rule (enforced here, not only in the screen):
  * reveal_llm_suggestion() returns nothing until the reviewer has locked their own rating;
  * a locked blind rating can never be changed (only the 'final' fields can be written afterwards);
  * a final level that differs from the blind level requires a written rationale.
Rationale: showing a machine suggestion first anchors human judgement (automation bias;
Parasuraman & Manzey, 2010, Human Factors, 52(3), 381-410).

An item_hash covers wording + options: if an item is edited, earlier ratings no longer match and
the screen shows the item as unrated for its new wording.
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _find_db() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "responses.db"
        if cand.exists():
            return cand
    return here.parents[min(3, len(here.parents) - 1)] / "responses.db"


_DB_PATH = _find_db()


def _conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    c = sqlite3.connect(db_path or _DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_schema(db_path: Optional[Path] = None) -> None:
    c = _conn(db_path)
    c.executescript("""
        CREATE TABLE IF NOT EXISTS item_llm_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            instrument_key TEXT NOT NULL,
            steps TEXT NOT NULL,
            repeats INTEGER NOT NULL DEFAULT 1,
            n_items INTEGER NOT NULL DEFAULT 0,
            est_cost_usd REAL,
            actual_cost_usd REAL,
            status TEXT NOT NULL DEFAULT 'running'
        );
        CREATE TABLE IF NOT EXISTS item_llm_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            instrument_key TEXT NOT NULL,
            question_id TEXT NOT NULL,
            item_hash TEXT NOT NULL,
            step TEXT NOT NULL,
            repeat_idx INTEGER NOT NULL DEFAULT 0,
            ok INTEGER NOT NULL,
            bloom_level INTEGER,
            solved_answer TEXT,
            solve_confidence TEXT,
            blind_matches_key INTEGER,
            needs_figure INTEGER,
            rationale TEXT,
            key_defensible TEXT,
            other_arguably_correct TEXT,
            implausible_distractors TEXT,
            comments TEXT,
            error TEXT,
            raw_response TEXT,
            model TEXT,
            prompt_version TEXT,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_item_llm_results_item
            ON item_llm_results (instrument_key, question_id, item_hash, step);
        CREATE TABLE IF NOT EXISTS item_human_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument_key TEXT NOT NULL,
            question_id TEXT NOT NULL,
            item_hash TEXT NOT NULL,
            rater TEXT NOT NULL,
            blind_level INTEGER NOT NULL,
            blind_rationale TEXT,
            blind_locked_at TEXT NOT NULL,
            final_level INTEGER,
            final_rationale TEXT,
            decision TEXT,
            decided_at TEXT,
            key_decision TEXT,
            key_note TEXT,
            UNIQUE (instrument_key, question_id, item_hash, rater)
        );
    """)
    c.commit()
    c.close()


# ------------------------------------------------------------------------------ LLM side
def create_run(created_by: str, model: str, prompt_version: str, instrument_key: str, steps: str,
               repeats: int, n_items: int, est_cost: float, db_path=None) -> str:
    rid = uuid.uuid4().hex[:12]
    c = _conn(db_path)
    c.execute("INSERT INTO item_llm_runs (run_id, created_by, created_at, model, prompt_version, instrument_key,"
              " steps, repeats, n_items, est_cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?)",
              (rid, created_by, _now(), model, prompt_version, instrument_key, steps, repeats, n_items, est_cost))
    c.commit()
    c.close()
    return rid


def finish_run(run_id: str, actual_cost: float, status: str = "done", db_path=None) -> None:
    c = _conn(db_path)
    c.execute("UPDATE item_llm_runs SET actual_cost_usd=?, status=? WHERE run_id=?", (actual_cost, status, run_id))
    c.commit()
    c.close()


def save_llm_result(run_id: str, item: Dict, step: str, repeat_idx: int, res: Dict, model: str,
                    prompt_version: str, db_path=None) -> None:
    d = res.get("data") or {}
    c = _conn(db_path)
    c.execute(
        "INSERT INTO item_llm_results (run_id, instrument_key, question_id, item_hash, step, repeat_idx, ok,"
        " bloom_level, solved_answer, solve_confidence, blind_matches_key, needs_figure, rationale,"
        " key_defensible, other_arguably_correct, implausible_distractors, comments, error, raw_response,"
        " model, prompt_version, tokens_in, tokens_out, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, item["instrument_key"], item["question_id"], item["item_hash"], step, repeat_idx,
         1 if res.get("ok") else 0, d.get("bloom_level"), d.get("solved_answer"), d.get("solve_confidence"),
         (None if d.get("blind_matches_key") is None else int(d["blind_matches_key"])),
         (None if d.get("needs_figure") is None else int(d["needs_figure"])), d.get("rationale"),
         d.get("key_defensible"), json.dumps(d.get("other_arguably_correct")) if "other_arguably_correct" in d else None,
         json.dumps(d.get("implausible_distractors")) if "implausible_distractors" in d else None, d.get("comments"),
         res.get("error"), res.get("raw"), model, prompt_version, res.get("tokens_in", 0), res.get("tokens_out", 0), _now()))
    c.commit()
    c.close()


def _latest_results(item: Dict, step: str, db_path=None, model: Optional[str] = None) -> List[sqlite3.Row]:
    c = _conn(db_path)
    q = ("SELECT * FROM item_llm_results WHERE instrument_key=? AND question_id=? AND item_hash=? "
         "AND step=? AND ok=1")
    args: list = [item["instrument_key"], item["question_id"], item["item_hash"], step]
    if model:
        q += " AND model=?"
        args.append(model)
    rows = c.execute(q + " ORDER BY created_at DESC, id DESC", args).fetchall()
    c.close()
    if not rows:
        return []
    last_run = rows[0]["run_id"]
    return [r for r in rows if r["run_id"] == last_run]


def llm_suggestion(item: Dict, db_path=None, model: Optional[str] = None) -> Optional[Dict]:
    """
    Latest successful LLM suggestion for this item version (blind step + key step) or None.
    With model=None this is the latest run of ANY model; pass a model id for one model.
    INTERNAL: callers showing it to a reviewer must use reveal_llm_suggestions().
    """
    blind = _latest_results(item, "blind", db_path, model)
    key = _latest_results(item, "key", db_path, model)
    if not blind and not key:
        return None
    out: Dict[str, Any] = {"n_runs": len(blind)}
    if blind:
        lv = [r["bloom_level"] for r in blind]
        out.update(levels=lv, level_median=float(statistics.median(lv)), level_min=min(lv), level_max=max(lv),
                   solved_answers=[r["solved_answer"] for r in blind],
                   blind_matches_key=[r["blind_matches_key"] for r in blind],
                   needs_figure=any(r["needs_figure"] for r in blind),
                   rationale=blind[0]["rationale"], model=blind[0]["model"], prompt_version=blind[0]["prompt_version"],
                   run_id=blind[0]["run_id"], solve_confidence=blind[0]["solve_confidence"])
    if key:
        r = key[0]
        out["key_review"] = {
            "key_defensible": r["key_defensible"], "comments": r["comments"],
            "other_arguably_correct": json.loads(r["other_arguably_correct"] or "[]"),
            "implausible_distractors": json.loads(r["implausible_distractors"] or "[]"),
            "model": r["model"], "prompt_version": r["prompt_version"]}
    return out


def llm_models_for_item(item: Dict, db_path=None) -> List[str]:
    c = _conn(db_path)
    rows = c.execute("SELECT DISTINCT model FROM item_llm_results WHERE instrument_key=? AND question_id=? "
                     "AND item_hash=? AND ok=1 ORDER BY model",
                     (item["instrument_key"], item["question_id"], item["item_hash"])).fetchall()
    c.close()
    return [r["model"] for r in rows]


def llm_suggestions_by_model(item: Dict, db_path=None) -> Dict[str, Dict]:
    """{model id: suggestion} for every model that has a stored result for this item version. INTERNAL."""
    out = {}
    for m in llm_models_for_item(item, db_path):
        sug = llm_suggestion(item, db_path, model=m)
        if sug:
            out[m] = sug
    return out


def consensus_level(suggestions: Dict[str, Dict]) -> Optional[Dict]:
    """Median of each model's median level, with the spread across models. None if no Bloom suggestions."""
    meds = {m: s["level_median"] for m, s in suggestions.items() if s.get("levels")}
    if not meds:
        return None
    vals = sorted(meds.values())
    return {"median": float(statistics.median(vals)), "min": min(vals), "max": max(vals), "n_models": len(vals),
            "per_model": meds}


# ----------------------------------------------------------------------------- human side
def get_human(rater: str, item: Dict, db_path=None) -> Optional[Dict]:
    c = _conn(db_path)
    r = c.execute("SELECT * FROM item_human_ratings WHERE instrument_key=? AND question_id=? AND item_hash=? "
                  "AND rater=?", (item["instrument_key"], item["question_id"], item["item_hash"], rater)).fetchone()
    c.close()
    return dict(r) if r else None


def lock_blind_rating(rater: str, item: Dict, level: int, rationale: str, db_path=None) -> Dict:
    """Save the reviewer's own rating BEFORE any LLM suggestion is shown. Cannot be changed later."""
    if level not in range(1, 7):
        raise ValueError("Bloom level must be 1-6.")
    if get_human(rater, item, db_path):
        raise ValueError("This rating is already locked and cannot be changed.")
    c = _conn(db_path)
    c.execute("INSERT INTO item_human_ratings (instrument_key, question_id, item_hash, rater, blind_level,"
              " blind_rationale, blind_locked_at) VALUES (?,?,?,?,?,?,?)",
              (item["instrument_key"], item["question_id"], item["item_hash"], rater, int(level),
               (rationale or "").strip(), _now()))
    c.commit()
    c.close()
    return get_human(rater, item, db_path)


def reveal_llm_suggestion(rater: str, item: Dict, db_path=None) -> Optional[Dict]:
    """Latest LLM suggestion (any model), ONLY if this reviewer has already locked their own rating."""
    if not get_human(rater, item, db_path):
        return None
    return llm_suggestion(item, db_path)


def reveal_llm_suggestions(rater: str, item: Dict, db_path=None) -> Dict[str, Dict]:
    """{model: suggestion} for ALL models, ONLY after this reviewer has locked their own rating."""
    if not get_human(rater, item, db_path):
        return {}
    return llm_suggestions_by_model(item, db_path)


def save_final_decision(rater: str, item: Dict, final_level: int, rationale: str, db_path=None) -> Dict:
    h = get_human(rater, item, db_path)
    if not h:
        raise ValueError("Lock your own rating first.")
    if final_level not in range(1, 7):
        raise ValueError("Bloom level must be 1-6.")
    sugs = llm_suggestions_by_model(item, db_path)
    model_levels = {int(round(x["level_median"])) for x in sugs.values() if x.get("levels")}
    if final_level == h["blind_level"]:
        decision = "kept"
    else:
        if not (rationale or "").strip():
            raise ValueError("A written rationale is required when the final level differs from your own first rating.")
        decision = "adopted_llm" if final_level in model_levels else "changed"
    c = _conn(db_path)
    c.execute("UPDATE item_human_ratings SET final_level=?, final_rationale=?, decision=?, decided_at=? "
              "WHERE instrument_key=? AND question_id=? AND item_hash=? AND rater=?",
              (int(final_level), (rationale or "").strip(), decision, _now(), item["instrument_key"],
               item["question_id"], item["item_hash"], rater))
    c.commit()
    c.close()
    return get_human(rater, item, db_path)


def save_key_decision(rater: str, item: Dict, decision: str, note: str, db_path=None) -> None:
    if decision not in ("key_ok", "needs_review"):
        raise ValueError("decision must be 'key_ok' or 'needs_review'.")
    if not get_human(rater, item, db_path):
        raise ValueError("Lock your own rating first.")
    c = _conn(db_path)
    c.execute("UPDATE item_human_ratings SET key_decision=?, key_note=? WHERE instrument_key=? AND question_id=? "
              "AND item_hash=? AND rater=?", (decision, (note or "").strip(), item["instrument_key"],
                                              item["question_id"], item["item_hash"], rater))
    c.commit()
    c.close()


def summary_table(items: List[Dict], rater: str, db_path=None) -> pd.DataFrame:
    """One row per item: reviewer ratings + LLM suggestion (shown only for items the reviewer has locked)."""
    rows = []
    for it in items:
        h = get_human(rater, it, db_path)
        by_model = llm_suggestions_by_model(it, db_path) if h else {}   # never expose LLM levels for unlocked items
        cons = consensus_level(by_model) if by_model else None
        s = llm_suggestion(it, db_path) if h else None
        kr = (s or {}).get("key_review") or {}
        for m_ in by_model.values():
            if m_.get("key_review"):
                kr = m_["key_review"]
                break
        row_models = {f"llm[{m}]": (round(x["level_median"], 1) if x.get("levels") else None)
                      for m, x in by_model.items()}
        rows.append({**row_models,
            "item": it["question_id"], "in_current_selection": it.get("in_current_selection"),
            "depends_on_figure": it["depends_on_figure"], "item_hash": it["item_hash"],
            "my_first_rating": h["blind_level"] if h else None,
            "my_final_rating": (h["final_level"] if h and h["final_level"] else None),
            "decision": h["decision"] if h else None,
            "llm_level_median": cons["median"] if cons else None,
            "llm_models": cons["n_models"] if cons else 0,
            "llm_level_spread": (f"{cons['min']:g}–{cons['max']:g}" if cons else None),
            "llm_blind_matches_key": (None if not s or not s.get("blind_matches_key") else
                                      all(v == 1 for v in s["blind_matches_key"] if v is not None)),
            "llm_key_defensible": kr.get("key_defensible"),
            "my_key_decision": h["key_decision"] if h else None,
        })
    return pd.DataFrame(rows)
