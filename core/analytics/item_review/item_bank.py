"""
core/analytics/item_review/item_bank.py

Loads assessment items (text + options + answer key) for the Item Review screen (Method 3 of the
item-difficulty options). Pure Python: no Streamlit, no database, no LLM.

Sources
-------
* Module content MCQs : content_dev/module{n}_question_bank.json  (the files the live app loads,
                        see modules/definitions/module{n}_definition.py) + key from
                        streamlit_app/surveys/module{n}_content_mcq_assessment_scoring.yaml
* AI-CI, Misconceptions: streamlit_app/surveys/pre_*_assessment.yaml (+ *_scoring.yaml); the post-test
                        uses the same items.
* "@cohort1" versions  : Modules 1-4 as Cohort 1 (NWACC 1D) took them on 2026-06-29, BEFORE the banks were
                        rewritten on 5-8 July. Snapshots of exactly the items Cohort 1 answered, taken from git
                        history, in content_dev/legacy_cohort1/; keys from the legacy scoring YAMLs.
                        (Modules 5-7 were not changed after that date: Cohort 1's items are identical to today's.)

Each item is a dict:
    instrument_key, instrument_label, module_n (int | None), question_id, text,
    options [{"label": "A", "text": "..."}], key (letter / value or None),
    item_hash (changes whenever the wording or options change), depends_on_figure (bool),
    in_current_selection (bool | None; module banks only: is it in the CURRENT .env selection)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[3]

# Revised Bloom's taxonomy, cognitive-process dimension (Anderson & Krathwohl, 2001).
BLOOM_LEVELS: Dict[int, Dict[str, str]] = {
    1: {"label": "Remember", "gloss": "recall a fact, term or definition that was taught"},
    2: {"label": "Understand", "gloss": "explain, interpret, classify or summarise an idea in own words"},
    3: {"label": "Apply", "gloss": "use a rule or procedure in a new but familiar situation"},
    4: {"label": "Analyze", "gloss": "break something into parts, compare, find how parts relate or why"},
    5: {"label": "Evaluate", "gloss": "judge, critique or justify using criteria or evidence"},
    6: {"label": "Create", "gloss": "design, plan or produce something new"},
}

_FIGURE = re.compile(r"\b(image|picture|diagram|photo|photograph|figure|chart|graph|illustration|"
                     r"shown (below|above)|look at|the (tree|map|table) (below|above))\b", re.I)

COHORT1_SUFFIX = "@cohort1"
MODULE_LABELS = {n: f"Module {n} — Content MCQ" for n in range(1, 8)}
SURVEY_INSTRUMENTS = {
    "precourse_pre_aici_assessment": ("AI Conceptual Inventory (pre = post items)", "pre_aici_assessment"),
    "precourse_pre_ai_misconceptions_assessment": ("AI Misconceptions (pre = post items)", "pre_ai_misconceptions_assessment"),
}


def item_hash(text: str, options: List[Dict[str, str]]) -> str:
    payload = json.dumps({"t": text.strip(), "o": [(o["label"], o["text"].strip()) for o in options]},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def list_instruments() -> Dict[str, str]:
    out = {f"module{n}_content_mcq_assessment": lab for n, lab in MODULE_LABELS.items()
           if (ROOT / "content_dev" / f"module{n}_question_bank.json").exists()}
    out.update({k: v[0] for k, v in SURVEY_INSTRUMENTS.items()})
    for n in range(1, 5):
        if (ROOT / "content_dev" / "legacy_cohort1" / f"module{n}_question_bank_cohort1_20260629.json").exists():
            out[f"module{n}_content_mcq_assessment{COHORT1_SUFFIX}"] = f"Module {n} — Cohort 1 version (as taken 2026-06-29)"
    return out


def _parse_option(raw: str, idx: int) -> Dict[str, str]:
    m = re.match(r"^\s*([A-Za-z])\s*[\)\.:]\s*(.*)$", raw, re.S)
    if m:
        return {"label": m.group(1).upper(), "text": m.group(2).strip()}
    return {"label": chr(ord("A") + idx), "text": raw.strip()}


def _read_env_ids(module_n: int) -> Optional[List[str]]:
    """Question IDs in the CURRENT .env selection (QUIZ_MODE=research), else None."""
    val = os.environ.get(f"QUIZ_QUESTION_IDS_MODULE_{module_n}")
    if val is None:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith(f"QUIZ_QUESTION_IDS_MODULE_{module_n}="):
                    val = line.split("=", 1)[1].strip()
                    break
    if not val:
        return None
    return [x.strip() for x in val.split(",") if x.strip()]


def _key_map(scoring_yaml: str) -> Dict[str, str]:
    p = ROOT / "streamlit_app" / "surveys" / scoring_yaml
    if not p.exists():
        return {}
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {str(k): str(v).strip() for k, v in (d.get("correct_answers") or {}).items()}


def _legacy_key_map(module_n: int) -> Dict[str, str]:
    fs = sorted((ROOT / "streamlit_app" / "surveys").glob(f"module{module_n}_content_mcq_assessment_scoring_legacy_until_*.yaml"))
    if not fs:
        return {}
    d = yaml.safe_load(fs[0].read_text(encoding="utf-8")) or {}
    return {str(k): str(v).strip() for k, v in (d.get("correct_answers") or {}).items()}


def load_item_bank(instrument_key: str) -> List[Dict]:
    items: List[Dict] = []
    ml = re.match(r"^module(\d+)_content_mcq_assessment" + re.escape(COHORT1_SUFFIX) + r"$", instrument_key)
    if ml:
        n = int(ml.group(1))
        raw = json.loads((ROOT / "content_dev" / "legacy_cohort1" /
                          f"module{n}_question_bank_cohort1_20260629.json").read_text(encoding="utf-8"))
        keys = _legacy_key_map(n)
        for q in raw:
            opts = [_parse_option(o, i) for i, o in enumerate(q["options"])]
            text = q["question"].strip()
            items.append({
                "instrument_key": instrument_key, "instrument_label": f"Module {n} — Cohort 1 version", "module_n": n,
                "question_id": str(q["id"]), "text": text, "options": opts,
                "key": keys.get(str(q["id"])), "item_hash": item_hash(text, opts),
                "depends_on_figure": bool(_FIGURE.search(text + " " + " ".join(o["text"] for o in opts))),
                "in_current_selection": None,
            })
        return items
    m = re.match(r"^module(\d+)_content_mcq_assessment$", instrument_key)
    if m:
        n = int(m.group(1))
        raw = json.loads((ROOT / "content_dev" / f"module{n}_question_bank.json").read_text(encoding="utf-8"))
        keys = _key_map(f"module{n}_content_mcq_assessment_scoring.yaml")
        live = _read_env_ids(n)
        for q in raw:
            opts = [_parse_option(o, i) for i, o in enumerate(q["options"])]
            text = q["question"].strip()
            items.append({
                "instrument_key": instrument_key, "instrument_label": MODULE_LABELS[n], "module_n": n,
                "question_id": str(q["id"]), "text": text, "options": opts,
                "key": keys.get(str(q["id"])), "item_hash": item_hash(text, opts),
                "depends_on_figure": bool(_FIGURE.search(text + " " + " ".join(o["text"] for o in opts))),
                "in_current_selection": (str(q["id"]) in live) if live else None,
            })
        return items

    if instrument_key in SURVEY_INSTRUMENTS:
        label, base = SURVEY_INSTRUMENTS[instrument_key]
        d = yaml.safe_load((ROOT / "streamlit_app" / "surveys" / f"{base}.yaml").read_text(encoding="utf-8"))
        keys = _key_map(f"{base}_scoring.yaml")
        for q in d.get("questions", []):
            opts = []
            for i, o in enumerate(q.get("options") or []):
                o = str(o)
                if re.match(r"^\s*[A-Za-z]\s*:", o):
                    lab, txt = o.split(":", 1)
                    opts.append({"label": lab.strip().upper(), "text": txt.strip()})
                else:                                   # True/False, Yes/No: the answer IS the label
                    opts.append({"label": o.strip(), "text": o.strip()})
            text = str(q["text"]).strip()
            items.append({
                "instrument_key": instrument_key, "instrument_label": label, "module_n": None,
                "question_id": str(q["id"]), "text": text, "options": opts,
                "key": keys.get(str(q["id"])), "item_hash": item_hash(text, opts),
                "depends_on_figure": bool(_FIGURE.search(text)), "in_current_selection": None,
            })
        return items
    raise KeyError(f"Unknown instrument: {instrument_key}")


def module_context(module_n: Optional[int]) -> str:
    """Module goal + learning objectives text for the LLM prompt (empty if unavailable)."""
    if not module_n:
        return ""
    p = ROOT / "content_dev" / "learning_objectives.yaml"
    if not p.exists():
        return ""
    lo = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("learning_objectives", {})
    m = lo.get(f"module_{module_n}") or {}
    if not m:
        return ""
    objs = "\n".join("- " + str(o).strip() for o in (m.get("learning_objectives") or []))
    concepts = ", ".join(str(c) for c in (m.get("core_ai_concepts") or []))
    return (f"Module {module_n}: {m.get('title', '')}\nGoal: {str(m.get('module_goal', '')).strip()}\n"
            f"Core concepts: {concepts}\nLearning objectives:\n{objs}")
