"""
core/analytics/item_review/bloom_llm.py

Method 3 of the item-difficulty options: LLM-assisted cognitive-demand (Bloom) rating of
assessment items, ALWAYS reviewed by a teacher/researcher (see store.py / the Item Review tab).

Two separate calls per item, in this order (decided with the researcher, 2026-10-05):
  Step 1  BLIND  : the model sees only the item text + options (NO answer key). It first answers
                   the item itself, then rates the Bloom level. Answering first lets us compare its
                   blind answer with the keyed answer in code (a disagreement flags a possible
                   key problem, which is how the AI-CI Q4_1 mis-key would have surfaced).
  Step 2  KEY    : the model sees the keyed answer and gives ADVISORY flags: is the key defensible,
                   is another option arguably correct, are any distractors implausible.

Privacy: only item wording / options / module learning objectives are sent. Never student data.

Model / API notes (checked against the claude-api skill, 2026-10-05)
  * Model claude-sonnet-5-5. Sonnet 5.5 rejects non-default sampling parameters (so NO temperature;
    the older call_claude() wrapper, which always sends one, cannot be used for it) and forced
    tool_choice; thinking runs adaptively. Structured output: output_config.format json_schema.
  * No server-side fallback is enabled: a refused/failed call is simply stored as an error and the
    human reviewer rates the item unaided.
  * Because temperature cannot be fixed, repeated calls can differ; the `repeats` option (default 1)
    stores each run so the screen can show the spread.
  * Prices ($2 / $10 per million input / output tokens) are the skill's cached 2026-09-25 values;
    the screen labels cost as an ESTIMATE.

Bloom's revised taxonomy: Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). A Taxonomy for Learning,
Teaching, and Assessing. Longman. Caution carried into the UI: Bloom level describes demanded thinking,
not difficulty (Kibble & Johnson, 2011, Adv. Physiol. Educ., 35, 396-401, found no correlation between
item cognitive level and student scores).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.analytics.item_review.item_bank import BLOOM_LEVELS, module_context

MODEL = "claude-sonnet-5-5"
PROMPT_VERSION = "bloom_v1"
PRICE_IN_PER_MTOK = 2.0
PRICE_OUT_PER_MTOK = 10.0
EST_TOKENS_IN = 1100       # per call, rough
EST_TOKENS_OUT = 900       # includes adaptive-thinking tokens, rough


def estimate_cost_usd(n_items: int, steps: int = 2, repeats: int = 1) -> float:
    calls = n_items * steps * repeats
    return calls * (EST_TOKENS_IN * PRICE_IN_PER_MTOK + EST_TOKENS_OUT * PRICE_OUT_PER_MTOK) / 1e6


def cost_from_usage(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * PRICE_IN_PER_MTOK + tokens_out * PRICE_OUT_PER_MTOK) / 1e6


_LEVEL_TEXT = "\n".join(f"{n}. {v['label']}: {v['gloss']}" for n, v in BLOOM_LEVELS.items())

SYSTEM_STEP1 = f"""You assist a curriculum researcher who is describing the cognitive demand of multiple-choice \
items from Basics4AI, an AI-literacy curriculum for learners in Grades 4-8.

Use the revised Bloom's taxonomy, cognitive-process dimension:
{_LEVEL_TEXT}

Rate the level of thinking the item AS WRITTEN requires from a Grade 4-8 learner who has completed the \
module. This is a description of the thinking demanded, NOT of how hard the item is: recalling a taught fact \
is "Remember" even if the topic sounds advanced, and an item that asks the learner to classify a new example, \
interpret a result, or reason about why something happens sits higher. When an item fits two adjacent levels, \
choose the lower one unless the item clearly requires more.

Work in this order: (1) answer the item yourself, using only the item text and options (you are NOT given an \
answer key); (2) rate its Bloom level; (3) give a rationale of at most two sentences that points at the wording \
that decided the level. If the item depends on an image, figure or table that is not shown to you, set \
needs_figure to true and still rate from the text alone. Respond only with the requested JSON."""

SYSTEM_STEP2 = """You assist a curriculum researcher who is checking the answer key of a multiple-choice item \
from Basics4AI, an AI-literacy curriculum for learners in Grades 4-8. You will be given the item, its options, \
and the keyed (official) answer.

Judge: (a) is the keyed answer defensible for a learner who completed the module; (b) is any OTHER option also \
arguably correct (list its letter); (c) which wrong options are implausible, so that almost no learner would \
choose them (list their letters). Be conservative: say "uncertain" rather than "no" when reasonable people could \
differ, and do not invent facts about the curriculum. Your output is advisory; a human makes the decision. \
Respond only with the requested JSON."""


def _letters(item: Dict) -> List[str]:
    return [o["label"] for o in item["options"]]


def schema_step1(item: Dict) -> Dict:
    return {
        "type": "object",
        "properties": {
            "solved_answer": {"type": "string", "enum": _letters(item) + ["unsure"]},
            "solve_confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "bloom_level": {"type": "integer", "enum": [1, 2, 3, 4, 5, 6]},
            "rationale": {"type": "string"},
            "needs_figure": {"type": "boolean"},
        },
        "required": ["solved_answer", "solve_confidence", "bloom_level", "rationale", "needs_figure"],
        "additionalProperties": False,
    }


def schema_step2(item: Dict) -> Dict:
    letters = {"type": "array", "items": {"type": "string", "enum": _letters(item)}}
    return {
        "type": "object",
        "properties": {
            "key_defensible": {"type": "string", "enum": ["yes", "no", "uncertain"]},
            "other_arguably_correct": letters,
            "implausible_distractors": letters,
            "comments": {"type": "string"},
        },
        "required": ["key_defensible", "other_arguably_correct", "implausible_distractors", "comments"],
        "additionalProperties": False,
    }


def _item_block(item: Dict) -> str:
    opts = "\n".join(f"{o['label']}) {o['text']}" for o in item["options"])
    return f"Item {item['question_id']}:\n{item['text']}\n\nOptions:\n{opts}"


def user_step1(item: Dict) -> str:
    ctx = module_context(item.get("module_n"))
    head = (f"Curriculum context for this module (what learners were taught):\n{ctx}\n\n" if ctx else
            "Context: this item is part of a general pre/post assessment of AI knowledge for Grades 4-8 "
            "learners who took the Basics4AI curriculum.\n\n")
    return head + _item_block(item)


def user_step2(item: Dict, key: str) -> str:
    ctx = module_context(item.get("module_n"))
    head = f"Curriculum context:\n{ctx}\n\n" if ctx else ""
    return head + _item_block(item) + f"\n\nKeyed (official) answer: {key}"


def _call(system: str, user: str, schema: Dict, max_tokens: int = 4000, effort: str = "medium") -> Dict[str, Any]:
    out = {"ok": False, "data": None, "raw": None, "error": None, "stop_reason": None,
           "tokens_in": 0, "tokens_out": 0}
    try:
        from core.analytics.llm.llm_clients import _load_keys
        key = _load_keys().get("anthropic")
        if not key:
            out["error"] = "ANTHROPIC_API_KEY not found (add it to .env or st.secrets)."
            return out
        import anthropic
        client = anthropic.Anthropic(api_key=key, max_retries=2, timeout=120.0)
        resp = client.messages.create(
            model=MODEL, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
        out["stop_reason"] = resp.stop_reason
        out["tokens_in"] = int(resp.usage.input_tokens)
        out["tokens_out"] = int(resp.usage.output_tokens)
        if resp.stop_reason == "refusal":
            out["error"] = "The model declined this request (refusal)."
            return out
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        out["raw"] = text
        if resp.stop_reason == "max_tokens":
            out["error"] = "The response was cut off (max_tokens); no rating stored."
            return out
        out["data"] = json.loads(text)          # parse, never string-match
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def rate_blind(item: Dict, effort: str = "medium") -> Dict[str, Any]:
    """Step 1: blind answer + Bloom level. Adds blind_matches_key (computed here, not by the model)."""
    r = _call(SYSTEM_STEP1, user_step1(item), schema_step1(item), effort=effort)
    if r["ok"]:
        d = r["data"]
        key = item.get("key")
        d["blind_matches_key"] = (None if (key is None or d["solved_answer"] == "unsure")
                                  else str(d["solved_answer"]).strip() == str(key).strip())
        if not 1 <= int(d["bloom_level"]) <= 6:
            r.update(ok=False, error="Model returned an out-of-range Bloom level.")
    return r


def review_key(item: Dict, effort: str = "medium") -> Dict[str, Any]:
    """Step 2: advisory key / distractor review (only if the item has a key)."""
    key = item.get("key")
    if key is None:
        return {"ok": False, "data": None, "raw": None, "error": "This item has no answer key on file.",
                "stop_reason": None, "tokens_in": 0, "tokens_out": 0}
    return _call(SYSTEM_STEP2, user_step2(item, key), schema_step2(item), effort=effort)
