"""
core/analytics/item_review/runner.py

Runs the LLM steps over a list of items for one or more providers and stores every result (success or
error) in item_llm_results, one run row per provider so the exact model id is always recorded. A failed
call is recorded and skipped; it never stops the batch. No Streamlit.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from core.analytics.item_review import bloom_llm, store


def run_llm_batch(
    items: List[Dict],
    created_by: str,
    steps: Sequence[str] = ("blind", "key"),
    repeats: int = 1,
    effort: str = "medium",
    progress: Optional[Callable[[int, int, str], None]] = None,
    db_path=None,
    providers: Sequence[str] = ("claude",),
) -> Dict:
    store.init_schema(db_path)
    steps = tuple(steps)
    providers = tuple(providers) or ("claude",)
    repeats = max(1, min(int(repeats), 5))
    inst = items[0]["instrument_key"] if items else ""
    total = max(1, len(items) * len(providers))
    done = 0
    runs: Dict[str, Dict] = {}
    n_ok = n_err = 0
    spent = est_total = 0.0
    for prov in providers:
        mid = bloom_llm.model_id(prov)
        est = bloom_llm.estimate_cost_usd(len(items), steps=len(steps), repeats=repeats if "blind" in steps else 1) \
            if prov == "claude" else 0.0
        run_id = store.create_run(created_by, mid, bloom_llm.PROMPT_VERSION, inst, ",".join(steps), repeats,
                                  len(items), est, db_path)
        p_ok = p_err = tok_in = tok_out = 0
        p_cost = 0.0
        for item in items:
            if progress:
                progress(done, total, f"{item['question_id']} · {prov}")
            if "blind" in steps:
                for rep in range(repeats):
                    r = bloom_llm.rate_blind(item, effort=effort, provider=prov)
                    store.save_llm_result(run_id, item, "blind", rep, r, mid, bloom_llm.PROMPT_VERSION, db_path)
                    tok_in, tok_out = tok_in + r["tokens_in"], tok_out + r["tokens_out"]
                    p_cost += bloom_llm.cost_from_usage(r["tokens_in"], r["tokens_out"], prov)
                    p_ok, p_err = p_ok + int(r["ok"]), p_err + int(not r["ok"])
            if "key" in steps:
                r = bloom_llm.review_key(item, effort=effort, provider=prov)
                store.save_llm_result(run_id, item, "key", 0, r, mid, bloom_llm.PROMPT_VERSION, db_path)
                tok_in, tok_out = tok_in + r["tokens_in"], tok_out + r["tokens_out"]
                p_cost += bloom_llm.cost_from_usage(r["tokens_in"], r["tokens_out"], prov)
                p_ok, p_err = p_ok + int(r["ok"]), p_err + int(not r["ok"])
            done += 1
        store.finish_run(run_id, p_cost, "done" if p_err == 0 else "done_with_errors", db_path)
        runs[prov] = {"run_id": run_id, "model": mid, "n_ok": p_ok, "n_err": p_err, "tokens_in": tok_in,
                      "tokens_out": tok_out, "cost_usd": p_cost}
        n_ok, n_err, spent, est_total = n_ok + p_ok, n_err + p_err, spent + p_cost, est_total + est
    if progress:
        progress(total, total, "")
    first = runs[providers[0]]["run_id"] if runs else None
    return {"run_id": first, "runs": runs, "n_ok": n_ok, "n_err": n_err, "cost_usd": spent,
            "est_cost_usd": est_total}
