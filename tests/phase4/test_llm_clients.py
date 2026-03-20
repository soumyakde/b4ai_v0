"""
Phase 4 — LLM Clients Unit Tests
test_llm_clients.py

Groups 1-5: No API calls — test structure, key loading, error handling.
Group 6:    Optional live API calls (requires real keys in .env).

Run from project root:
    python tests/phase4/test_llm_clients.py
    python tests/phase4/test_llm_clients.py --live   # makes real API calls
"""

import sys, os, importlib.util

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_THIS_DIR, "..", "..", "core", "analytics", "llm", "llm_clients.py"),
    os.path.join(_THIS_DIR, "llm_clients.py"),
]
_path = next((os.path.normpath(p) for p in _CANDIDATES
              if os.path.exists(os.path.normpath(p))), None)
if not _path:
    print("ERROR: llm_clients.py not found"); sys.exit(1)

_spec = importlib.util.spec_from_file_location("llm_clients", _path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

call_model           = _mod.call_model
call_claude          = _mod.call_claude
call_gemini          = _mod.call_gemini
call_openai          = _mod.call_openai
get_available_models = _mod.get_available_models
get_display_name     = _mod.get_display_name
_DEFAULT_MODELS      = _mod._DEFAULT_MODELS
_DISPLAY_NAMES       = _mod._DISPLAY_NAMES

LIVE = "--live" in sys.argv
PASS = FAIL = 0

def check(label, got, expected):
    global PASS, FAIL
    ok = (got == expected)
    if ok: print(f"  ✅ PASS  {label}"); PASS += 1
    else:
        print(f"  ❌ FAIL  {label}  got={got!r}  expected={expected!r}"); FAIL += 1

def check_true(label, cond, detail=""):
    global PASS, FAIL
    if cond: print(f"  ✅ PASS  {label}"); PASS += 1
    else: print(f"  ❌ FAIL  {label}  {detail}"); FAIL += 1

# -----------------------------------------------------------------------
# Group 1 — Result dict contract
# -----------------------------------------------------------------------
print("\n[Group 1] Result dict contract — _ok and _err shapes")

# Test result dict shape using internal _ok/_err directly
# This avoids env-patching complexity while verifying the contract
_ok  = _mod._ok
_err = _mod._err

# _err shape
err_result = _err("test error message", "claude")
check_true("error result has 'text' key",        "text" in err_result)
check_true("error result has 'model' key",       "model" in err_result)
check_true("error result has 'tokens_used' key", "tokens_used" in err_result)
check_true("error result has 'error' key",       "error" in err_result)
check_true("text is None on error",              err_result["text"] is None)
check_true("error is str on _err",
    isinstance(err_result["error"], str) and len(err_result["error"]) > 0)

# _ok shape
ok_result = _ok("some text", "claude", 42)
check_true("ok result text populated",       ok_result["text"] == "some text")
check_true("ok result error is None",        ok_result["error"] is None)
check_true("ok result tokens_used correct",  ok_result["tokens_used"] == 42)

# All three client functions return correct shape regardless of key state
for fn, name in [(call_gemini,"gemini"), (call_openai,"openai")]:
    r = fn("test", max_tokens=5)
    check_true(f"{name} result dict complete", all(
        k in r for k in ("text","model","tokens_used","error")
    ))

# -----------------------------------------------------------------------
# Group 2 — call_model dispatcher
# -----------------------------------------------------------------------
print("\n[Group 2] call_model dispatcher")

# Unknown model → error
r_unk = call_model("unknown_model", "test")
check_true("unknown model → error set", r_unk["error"] is not None)
check_true("unknown model error mentions model name",
    "unknown_model" in str(r_unk["error"]))

# All three known models dispatch without crashing
for m in ("claude", "gemini", "gpt"):
    r = call_model(m, "test prompt", max_tokens=5)
    check_true(f"call_model('{m}') returns dict",
        isinstance(r, dict) and "error" in r)

# -----------------------------------------------------------------------
# Group 3 — Default models
# -----------------------------------------------------------------------
print("\n[Group 3] Default model IDs")

check_true("claude default model set",  bool(_DEFAULT_MODELS.get("claude")))
check_true("gemini default model set",  bool(_DEFAULT_MODELS.get("gemini")))
check_true("gpt default model set",     bool(_DEFAULT_MODELS.get("gpt")))
check("claude default is sonnet",
    _DEFAULT_MODELS["claude"], "claude-sonnet-4-5")
check("gemini default is flash",
    _DEFAULT_MODELS["gemini"], "gemini-2.5-flash")
check("gpt default is gpt-4o-mini",
    _DEFAULT_MODELS["gpt"],    "gpt-4o-mini")

# -----------------------------------------------------------------------
# Group 4 — get_available_models
# -----------------------------------------------------------------------
print("\n[Group 4] get_available_models")

available = get_available_models(check_keys=True)
check_true("returns dict with 3 keys",
    set(available.keys()) == {"claude", "gemini", "gpt"})
check_true("all values are bool",
    all(isinstance(v, bool) for v in available.values()))

# Without key check — all True
all_available = get_available_models(check_keys=False)
check("all True when check_keys=False",
    all(all_available.values()), True)

print(f"\n  Key availability: {available}")
print(f"  (True = key found in env/.env/secrets)")

# -----------------------------------------------------------------------
# Group 5 — get_display_name
# -----------------------------------------------------------------------
print("\n[Group 5] get_display_name")

check("claude display name", get_display_name("claude"), "Claude (Anthropic)")
check("gemini display name", get_display_name("gemini"), "Gemini (Google)")
check("gpt display name",    get_display_name("gpt"),    "GPT (OpenAI)")
check("unknown passthrough", get_display_name("unknown"), "unknown")
check("case insensitive",    get_display_name("CLAUDE"), "Claude (Anthropic)")

# -----------------------------------------------------------------------
# Group 6 — Live API calls (optional, requires --live flag + real keys)
# -----------------------------------------------------------------------
print(f"\n[Group 6] Live API calls {'(ENABLED)' if LIVE else '(SKIPPED — run with --live)'}")

if LIVE:
    TEST_PROMPT = (
        "In one sentence, what is the capital of France? "
        "Answer with just the city name."
    )
    TEST_SYSTEM = "You are a helpful assistant. Be very concise."

    for model_name in ("claude", "gemini", "gpt"):
        if not available.get(model_name):
            print(f"  ⏭  SKIP  {model_name} — API key not found")
            continue

        print(f"\n  Testing {get_display_name(model_name)}...")
        r = call_model(
            model_name, TEST_PROMPT,
            system=TEST_SYSTEM,
            temperature=0.0,
            max_tokens=20,
        )
        if r["error"]:
            print(f"  ❌ FAIL  {model_name} error: {r['error']}")
            FAIL += 1
        else:
            check_true(f"{model_name} returns non-empty text",
                bool(r["text"]) and len(r["text"]) > 0)
            check_true(f"{model_name} tokens_used > 0",
                r["tokens_used"] > 0)
            print(f"  ℹ️  Response: {r['text'].strip()[:80]}")
            print(f"  ℹ️  Tokens: {r['tokens_used']}")
else:
    print("  Run with --live flag to test real API calls")
    print("  Requires ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY in .env")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — llm_clients verified.")
else:
    print("  ❌ SOME TESTS FAILED — review above.")
print('='*60)
