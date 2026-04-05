# core/analytics/llm/llm_clients.py
"""
LLM API Client Wrappers
=======================
Unified interface for Anthropic (Claude), Google (Gemini), OpenAI (GPT).

Key design decisions:
- Keys loaded from .env (local) with fallback to st.secrets (shared/cloud)
- All three clients share identical call signature
- Errors returned as result dicts — never raised to caller
- No Streamlit imports except in _load_keys() for secrets fallback

Usage:
    from core.analytics.llm.llm_clients import call_model

    result = call_model(
        model="claude",
        prompt="Identify themes in this text...",
        system="You are a qualitative research assistant...",
        temperature=0.0,
        max_tokens=2000,
    )
    if result["error"]:
        print(result["error"])
    else:
        print(result["text"])
"""

from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path
import os

# -----------------------------------------------------------------------
# Key loading — .env first, st.secrets fallback
# -----------------------------------------------------------------------

def _load_keys() -> Dict[str, Optional[str]]:
    """
    Load API keys from environment.

    Priority order:
    1. Already-set environment variables (covers CI, Docker, production)
    2. .env file at project root (local development)
    3. st.secrets (Streamlit Cloud / shared deployment)

    Keys expected:
        ANTHROPIC_API_KEY
        GEMINI_API_KEY
        OPENAI_API_KEY
    """
    # Step 1: try python-dotenv if available
    try:
        from dotenv import load_dotenv
        # Walk up to find .env — handles varying deployment depths
        _here = Path(__file__).resolve()
        _env_path = None
        for _parent in _here.parents:
            _candidate = _parent / ".env"
            if _candidate.exists():
                _env_path = _candidate
                break
        if _env_path:
            load_dotenv(_env_path, override=False)
    except ImportError:
        pass  # dotenv not installed — skip silently

    keys = {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "gemini":    os.environ.get("GEMINI_API_KEY"),
        "openai":    os.environ.get("OPENAI_API_KEY"),
        "groq":      os.environ.get("GROQ_API_KEY"),
    }

    # Step 2: fall back to st.secrets for any missing keys
    missing = [k for k, v in keys.items() if not v]
    if missing:
        try:
            import streamlit as st
            secrets = st.secrets if hasattr(st, "secrets") else {}
            key_map = {
                "anthropic": "ANTHROPIC_API_KEY",
                "gemini":    "GEMINI_API_KEY",
                "openai":    "OPENAI_API_KEY",
                "groq":      "GROQ_API_KEY",
            }
            for k in missing:
                env_key = key_map[k]
                if env_key in secrets:
                    keys[k] = secrets[env_key]
        except Exception:
            pass  # Streamlit not running — skip silently

    return keys


# -----------------------------------------------------------------------
# Result dict contract
# -----------------------------------------------------------------------

def _ok(text: str, model: str, tokens_used: int = 0) -> Dict[str, Any]:
    return {
        "text":        text,
        "model":       model,
        "tokens_used": tokens_used,
        "error":       None,
    }


def _err(message: str, model: str) -> Dict[str, Any]:
    return {
        "text":        None,
        "model":       model,
        "tokens_used": 0,
        "error":       message,
    }


# -----------------------------------------------------------------------
# Claude (Anthropic)
# -----------------------------------------------------------------------

def call_claude(
    prompt: str,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2000,
    model_id: str = "claude-sonnet-4-5",
) -> Dict[str, Any]:
    """
    Call Anthropic Claude API.

    Parameters
    ----------
    prompt : str
        User message content.
    system : str
        System prompt (curriculum context + role definition).
    temperature : float
        0.0 = deterministic, up to 1.0 for creative variation.
        Note: Anthropic clamps max temperature at 1.0.
    max_tokens : int
        Maximum response tokens.
    model_id : str
        Anthropic model identifier.

    Returns
    -------
    dict : {text, model, tokens_used, error}
    """
    keys = _load_keys()
    api_key = keys.get("anthropic")

    if not api_key:
        return _err(
            "ANTHROPIC_API_KEY not found. Add it to .env or st.secrets.",
            "claude"
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Anthropic temperature range: 0.0–1.0
        temp = min(float(temperature), 1.0)

        kwargs: Dict[str, Any] = {
            "model":      model_id,
            "max_tokens": max_tokens,
            "temperature": temp,
            "messages":   [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
        text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens

        return _ok(text, "claude", tokens)

    except Exception as e:
        return _err(f"Claude API error: {e}", "claude")


# -----------------------------------------------------------------------
# Gemini (Google)
# -----------------------------------------------------------------------

def call_gemini(
    prompt: str,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2000,
    model_id: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    Call Google Gemini API via google-genai SDK.

    Parameters
    ----------
    prompt : str
    system : str
        System instruction passed as system_instruction parameter.
    temperature : float
        0.0–2.0.
    max_tokens : int
    model_id : str
        Gemini model identifier.

    Returns
    -------
    dict : {text, model, tokens_used, error}
    """
    keys = _load_keys()
    api_key = keys.get("gemini")

    if not api_key:
        return _err(
            "GEMINI_API_KEY not found. Add it to .env or st.secrets.",
            "gemini"
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        config = types.GenerateContentConfig(
            temperature=float(temperature),
            max_output_tokens=max_tokens,
            system_instruction=system if system else None,
        )

        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config,
        )

        text = response.text
        tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0

        return _ok(text, "gemini", tokens)

    except Exception as e:
        return _err(f"Gemini API error: {e}", "gemini")


# -----------------------------------------------------------------------
# OpenAI (GPT)
# -----------------------------------------------------------------------

def call_openai(
    prompt: str,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2000,
    model_id: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Call OpenAI GPT API.

    Parameters
    ----------
    prompt : str
    system : str
    temperature : float
        0.0–2.0.
    max_tokens : int
    model_id : str

    Returns
    -------
    dict : {text, model, tokens_used, error}
    """
    keys = _load_keys()
    api_key = keys.get("openai")

    if not api_key:
        return _err(
            "OPENAI_API_KEY not found. Add it to .env or st.secrets.",
            "gpt"
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=float(temperature),
            max_tokens=max_tokens,
        )

        text = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0

        return _ok(text, "gpt", tokens)

    except Exception as e:
        return _err(f"OpenAI API error: {e}", "gpt")


# -----------------------------------------------------------------------
# Groq (Llama 3.1 70B — free tier)
# -----------------------------------------------------------------------

def call_groq(
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,           # raised from 2000 — Llama descriptions are verbose
    model_id: str = "llama-3.3-70b-versatile",
) -> Dict[str, Any]:
    """
    Call Groq API (Llama 3.3 70B).

    Free tier limits (as of 2025)
    ------------------------------
    - 100,000 tokens / day  (TPD)
    - 6,000 tokens / minute (TPM)
    If either limit is hit a 429 is returned. This function retries up to
    3 times with exponential backoff before returning an error.

    Design notes
    ------------
    - response_format=json_object is intentionally NOT used. It forces Llama
      to produce valid JSON but ignores the schema in the prompt, causing it
      to use arbitrary key names instead of pipeline-expected keys.
    - max_tokens is set to 4096 (not 2000) because Llama generates verbose
      descriptions; truncation at 2000 tokens produces invalid JSON.
    - Markdown fences are stripped before returning.

    Returns
    -------
    dict : {text, model, tokens_used, error}
    """
    import time

    keys = _load_keys()
    api_key = keys.get("groq")

    if not api_key:
        return _err(
            "GROQ_API_KEY not found. Add it to .env or st.secrets.",
            "groq"
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    max_retries = 3
    for attempt in range(max_retries):
        try:
            from groq import Groq
            client = Groq(api_key=api_key)

            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=float(temperature),
                max_tokens=max_tokens,
                # NO response_format — let prompt schema drive output
            )

            text   = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if response.usage else 0

            # Strip markdown fences if present
            if text.startswith("```"):
                lines = text.splitlines()
                inner = [l for l in lines[1:] if l.strip() != "```"]
                text  = "\n".join(inner).strip()

            return _ok(text, "groq", tokens)

        except Exception as e:
            err_str = str(e)

            # 429 rate limit — wait and retry
            if "429" in err_str and attempt < max_retries - 1:
                # Parse suggested wait time from error if present
                import re as _re
                m = _re.search(r"try again in ([\d.]+)s", err_str)
                wait = float(m.group(1)) + 2 if m else (10 * (attempt + 1))
                wait = min(wait, 60)   # cap at 60s so UI doesn't appear frozen
                time.sleep(wait)
                continue

            # Daily token limit (TPD) — not recoverable by waiting a few seconds
            if "tokens per day" in err_str.lower() or "TPD" in err_str:
                return _err(
                    f"Groq daily token limit reached (100K tokens/day free tier). "
                    f"Wait until tomorrow or upgrade at console.groq.com/settings/billing. "
                    f"Full error: {err_str}",
                    "groq"
                )

            return _err(f"Groq API error: {err_str}", "groq")


# -----------------------------------------------------------------------
# Unified dispatcher
# -----------------------------------------------------------------------

# Default model IDs — override by passing model_id to individual functions
_DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-5",
    "gemini": "gemini-2.5-flash",
    "gpt":    "gpt-4o-mini",
    "groq":   "llama-3.3-70b-versatile",
}

_DISPLAY_NAMES = {
    "claude": "Claude (Anthropic)",
    "gemini": "Gemini (Google)",
    "gpt":    "GPT (OpenAI)",
    "groq":   "Llama 3.3 70B (Groq — free)",
}


def call_model(
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2000,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified model dispatcher.

    Parameters
    ----------
    model : str
        One of: "claude", "gemini", "gpt"
    prompt : str
    system : str
    temperature : float
    max_tokens : int
    model_id : str or None
        Override default model ID. If None, uses _DEFAULT_MODELS[model].

    Returns
    -------
    dict : {text, model, tokens_used, error}
    """
    model = model.lower().strip()

    if model not in _DEFAULT_MODELS:
        return _err(
            f"Unknown model '{model}'. Use one of: {list(_DEFAULT_MODELS.keys())}",
            model
        )

    mid = model_id or _DEFAULT_MODELS[model]

    if model == "claude":
        return call_claude(prompt, system, temperature, max_tokens, mid)
    elif model == "gemini":
        return call_gemini(prompt, system, temperature, max_tokens, mid)
    elif model == "gpt":
        return call_openai(prompt, system, temperature, max_tokens, mid)
    elif model == "groq":
        return call_groq(prompt, system, temperature, max_tokens, mid)


def get_available_models(check_keys: bool = True) -> Dict[str, bool]:
    """
    Return dict of {model_name: available} based on whether API keys exist.

    Parameters
    ----------
    check_keys : bool
        If True, check that keys are present. If False, return all True.

    Returns
    -------
    dict: {"claude": bool, "gemini": bool, "gpt": bool}
    """
    if not check_keys:
        return {m: True for m in _DEFAULT_MODELS}

    keys = _load_keys()
    return {
        "claude": bool(keys.get("anthropic")),
        "gemini": bool(keys.get("gemini")),
        "gpt":    bool(keys.get("openai")),
        "groq":   bool(keys.get("groq")),
    }


def get_display_name(model: str) -> str:
    """Return human-readable model name for UI display."""
    return _DISPLAY_NAMES.get(model.lower(), model)
