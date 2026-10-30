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

Four model providers (multi-model option, 2026-10-05). The same prompts and the same JSON answer
format are used for all of them so their ratings are comparable:
  claude  Anthropic     claude-sonnet-5-5      native structured output (json_schema)
  gpt     OpenAI        gpt-5.5                JSON mode + our own validation
  gemini  Google        gemini-3.5-flash       JSON mime type + our own validation
          (gemini-2.5-pro was tried first: Google no longer serves it to new accounts even though it is listed)
  groq    Groq-hosted   openai/gpt-oss-120b    JSON in prompt + our own validation
          NOTE: "groq" is a hosting company, not a model. On 2026-10-05 it served NO Llama chat model
          (only Meta's small prompt-guard safety models); the open-weight model it hosts is OpenAI's
          gpt-oss. It is labelled honestly in the UI. A Llama model needs another host + key.
Model ids can be overridden with environment variables B4AI_ITEM_MODEL_CLAUDE / _GPT / _GEMINI / _GROQ.
Every stored result records the exact model id, so results from different models are never mixed.

Privacy: only item wording / options / module learning objectives are sent. Never student data.

Claude API notes (claude-api skill, 2026-10-05): Sonnet 5.5 rejects non-default sampling parameters
(NO temperature) and forced tool_choice; thinking runs adaptively. No server-side fallback is enabled.
None of the providers is run at a fixed temperature, so repeated calls can differ; the `repeats`
option stores each run so the screen can show the spread.

Cost: only Claude's price is built in ($2 / $10 per million input / output tokens, the skill's cached
2026-09-25 table). Token counts are stored for every provider; their prices are not assumed.
Groq's free tier has daily token limits (see llm_clients.call_groq notes).

Bloom's revised taxonomy: Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). A Taxonomy for Learning,
Teaching, and Assessing. Longman. Caution carried into the UI: Bloom level describes demanded thinking,
not difficulty (Kibble & Johnson, 2011, Adv. Physiol. Educ., 35, 396-401, found no correlation between
item cognitive level and student scores).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from core.analytics.item_review.item_bank import BLOOM_LEVELS, module_context

PROMPT_VERSION = "bloom_v1"

PROVIDERS: Dict[str, Dict[str, str]] = {
    "claude": {"label": "Claude (Anthropic)", "model": "claude-sonnet-5-5", "key": "anthropic"},
    "gpt": {"label": "ChatGPT (OpenAI)", "model": "gpt-5.5", "key": "openai"},
    "gemini": {"label": "Gemini (Google)", "model": "gemini-3.5-flash", "key": "gemini"},
    "groq": {"label": "Open-weight model via Groq (GPT-OSS 120B; no Llama chat model offered there today)",
             "model": "openai/gpt-oss-120b", "key": "groq"},
}
DEFAULT_PROVIDER = "claude"
MODEL = PROVIDERS["claude"]["model"]            # kept for backward compatibility

PRICE_IN_PER_MTOK = 2.0                         # Claude Sonnet 5.5 only
PRICE_OUT_PER_MTOK = 10.0
EST_TOKENS_IN = 1100                            # per call, rough
EST_TOKENS_OUT = 900                            # includes hidden reasoning tokens, rough


def model_id(provider: str) -> str:
    return os.environ.get(f"B4AI_ITEM_MODEL_{provider.upper()}") or PROVIDERS[provider]["model"]


def provider_available(provider: str) -> bool:
    try:
        from core.analytics.llm.llm_clients import _load_keys
        return bool(_load_keys().get(PROVIDERS[provider]["key"]))
    except Exception:
        return False


def estimate_calls(n_items: int, steps: int = 2, repeats: int = 1) -> int:
    return n_items * steps * repeats


def estimate_tokens(n_items: int, steps: int = 2, repeats: int = 1) -> int:
    return estimate_calls(n_items, steps, repeats) * (EST_TOKENS_IN + EST_TOKENS_OUT)


def estimate_cost_usd(n_items: int, steps: int = 2, repeats: int = 1) -> float:
    """Claude Sonnet 5.5 only (other providers' prices are not assumed)."""
    return estimate_calls(n_items, steps, repeats) * (EST_TOKENS_IN * PRICE_IN_PER_MTOK + EST_TOKENS_OUT * PRICE_OUT_PER_MTOK) / 1e6


def cost_from_usage(tokens_in: int, tokens_out: int, provider: str = "claude") -> float:
    if provider != "claude":
        return 0.0
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


# ------------------------------------------------------------------ schema helpers (non-Claude providers)
def schema_instructions(schema: Dict) -> str:
    """Plain-language JSON format description appended to the system prompt for providers without native schemas."""
    lines = ["Return ONLY one JSON object (no markdown, no text before or after) with exactly these keys:"]
    for k, p in schema["properties"].items():
        if p.get("type") == "array":
            lines.append(f'- "{k}": a JSON array of strings, each one of {p["items"]["enum"]} (use [] if none)')
        elif "enum" in p:
            vals = json.dumps(p["enum"])
            lines.append(f'- "{k}": one of {vals}')
        elif p.get("type") == "boolean":
            lines.append(f'- "{k}": true or false')
        else:
            lines.append(f'- "{k}": a string')
    return "\n".join(lines)


def extract_json(text: str) -> Any:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            return json.loads(t[a:b + 1])
        raise


def validate_against_schema(data: Any, schema: Dict) -> List[str]:
    """Minimal validator for the flat schemas used here (required, additionalProperties, types, enums)."""
    errs: List[str] = []
    if not isinstance(data, dict):
        return ["Answer is not a JSON object."]
    for k in schema["required"]:
        if k not in data:
            errs.append(f"missing key '{k}'")
    for k in data:
        if k not in schema["properties"]:
            errs.append(f"unexpected key '{k}'")
    for k, p in schema["properties"].items():
        if k not in data:
            continue
        v = data[k]
        t = p.get("type")
        if t == "string" and not isinstance(v, str):
            errs.append(f"'{k}' must be a string")
        elif t == "integer" and (isinstance(v, bool) or not isinstance(v, int)):
            errs.append(f"'{k}' must be an integer")
        elif t == "boolean" and not isinstance(v, bool):
            errs.append(f"'{k}' must be true or false")
        elif t == "array":
            if not isinstance(v, list) or any(x not in p["items"]["enum"] for x in v):
                errs.append(f"'{k}' must be a list of allowed letters")
        if "enum" in p and t != "array" and v not in p["enum"]:
            errs.append(f"'{k}' has a value outside the allowed set")
    return errs


# ------------------------------------------------------------------------------------ provider calls
def _blank() -> Dict[str, Any]:
    return {"ok": False, "data": None, "raw": None, "error": None, "stop_reason": None,
            "tokens_in": 0, "tokens_out": 0}


def _call_claude(system: str, user: str, schema: Dict, max_tokens: int, effort: str) -> Dict[str, Any]:
    out = _blank()
    from core.analytics.llm.llm_clients import _load_keys
    key = _load_keys().get("anthropic")
    if not key:
        out["error"] = "ANTHROPIC_API_KEY not found (add it to .env or st.secrets)."
        return out
    import anthropic
    client = anthropic.Anthropic(api_key=key, max_retries=2, timeout=120.0)
    resp = client.messages.create(
        model=model_id("claude"), max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    )
    out["stop_reason"] = resp.stop_reason
    out["tokens_in"], out["tokens_out"] = int(resp.usage.input_tokens), int(resp.usage.output_tokens)
    if resp.stop_reason == "refusal":
        out["error"] = "The model declined this request (refusal)."
        return out
    out["raw"] = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    if resp.stop_reason == "max_tokens":
        out["error"] = "The response was cut off (max_tokens); no rating stored."
        return out
    out["data"] = json.loads(out["raw"])
    out["ok"] = True
    return out


def _call_openai(system: str, user: str, max_tokens: int) -> Dict[str, Any]:
    out = _blank()
    from core.analytics.llm.llm_clients import _load_keys
    key = _load_keys().get("openai")
    if not key:
        out["error"] = "OPENAI_API_KEY not found (add it to .env or st.secrets)."
        return out
    from openai import OpenAI
    r = OpenAI(api_key=key, timeout=120.0, max_retries=2).chat.completions.create(
        model=model_id("gpt"), messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"}, max_completion_tokens=max_tokens)
    ch = r.choices[0]
    out["stop_reason"] = ch.finish_reason
    out["tokens_in"] = int(getattr(r.usage, "prompt_tokens", 0) or 0)
    out["tokens_out"] = int(getattr(r.usage, "completion_tokens", 0) or 0)
    out["raw"] = ch.message.content or ""
    if ch.finish_reason == "length":
        out["error"] = "The response was cut off (token limit); no rating stored."
    return out


def _call_gemini(system: str, user: str, max_tokens: int) -> Dict[str, Any]:
    out = _blank()
    from core.analytics.llm.llm_clients import _load_keys
    key = _load_keys().get("gemini")
    if not key:
        out["error"] = "GEMINI_API_KEY not found (add it to .env or st.secrets)."
        return out
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)      # keep a reference: an inline client is closed before the call ends
    r = client.models.generate_content(
        model=model_id("gemini"), contents=user,
        config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                           max_output_tokens=max_tokens))
    um = r.usage_metadata
    out["tokens_in"] = int(getattr(um, "prompt_token_count", 0) or 0)
    out["tokens_out"] = int((getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0))
    cand = r.candidates[0] if r.candidates else None
    out["stop_reason"] = str(getattr(cand, "finish_reason", "")) if cand else "no_candidate"
    try:
        out["raw"] = r.text or ""
    except Exception:
        out["raw"] = ""
    if not out["raw"]:
        out["error"] = f"No text returned (finish reason: {out['stop_reason']})."
    return out


def _call_groq(system: str, user: str, max_tokens: int) -> Dict[str, Any]:
    out = _blank()
    from core.analytics.llm.llm_clients import _load_keys
    key = _load_keys().get("groq")
    if not key:
        out["error"] = "GROQ_API_KEY not found (add it to .env or st.secrets)."
        return out
    from groq import Groq
    r = Groq(api_key=key, max_retries=2, timeout=120.0).chat.completions.create(
        model=model_id("groq"), messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens)
    ch = r.choices[0]
    out["stop_reason"] = ch.finish_reason
    out["tokens_in"] = int(getattr(r.usage, "prompt_tokens", 0) or 0)
    out["tokens_out"] = int(getattr(r.usage, "completion_tokens", 0) or 0)
    out["raw"] = ch.message.content or ""
    if ch.finish_reason == "length":
        out["error"] = "The response was cut off (token limit); no rating stored."
    return out


def _call(system: str, user: str, schema: Dict, max_tokens: int = 4000, effort: str = "medium",
          provider: str = "claude") -> Dict[str, Any]:
    """Dispatch one call. Never raises. Non-Claude providers get up to 2 attempts (one correction retry)."""
    out = _blank()
    try:
        if provider == "claude":
            return _call_claude(system, user, schema, max_tokens, effort)
        sys_full = system + "\n\n" + schema_instructions(schema)
        fn = {"gpt": _call_openai, "gemini": _call_gemini, "groq": _call_groq}[provider]
        budget = max(max_tokens, 8000)          # reasoning models spend hidden tokens against this budget
        tin = tout = 0
        prompt = user
        for attempt in range(2):
            r = fn(sys_full, prompt, budget)
            tin, tout = tin + r["tokens_in"], tout + r["tokens_out"]
            r["tokens_in"], r["tokens_out"] = tin, tout
            if r["error"]:
                return r
            try:
                data = extract_json(r["raw"])
                errs = validate_against_schema(data, schema)
            except Exception as e:
                data, errs = None, [f"not valid JSON ({e})"]
            if not errs:
                r["data"], r["ok"] = data, True
                return r
            if attempt == 1:
                r["error"] = "Answer did not match the required format: " + "; ".join(errs)
                return r
            prompt = (user + "\n\nYour previous answer was rejected: " + "; ".join(errs) +
                      ". Reply again with ONLY the JSON object in the required format.")
        return r
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


def rate_blind(item: Dict, effort: str = "medium", provider: str = "claude") -> Dict[str, Any]:
    """Step 1: blind answer + Bloom level. Adds blind_matches_key (computed here, not by the model)."""
    r = _call(SYSTEM_STEP1, user_step1(item), schema_step1(item), effort=effort, provider=provider)
    if r["ok"]:
        d = r["data"]
        key = item.get("key")
        d["blind_matches_key"] = (None if (key is None or d["solved_answer"] == "unsure")
                                  else str(d["solved_answer"]).strip() == str(key).strip())
        if not 1 <= int(d["bloom_level"]) <= 6:
            r.update(ok=False, error="Model returned an out-of-range Bloom level.")
    return r


def review_key(item: Dict, effort: str = "medium", provider: str = "claude") -> Dict[str, Any]:
    """Step 2: advisory key / distractor review (only if the item has a key)."""
    key = item.get("key")
    if key is None:
        r = _blank()
        r["error"] = "This item has no answer key on file."
        return r
    return _call(SYSTEM_STEP2, user_step2(item, key), schema_step2(item), effort=effort, provider=provider)
