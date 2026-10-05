"""
core/analytics/item_review/runner.py

Runs the Sonnet steps over a list of items and stores every result (success or error) in
item_llm_results. A failed call is recorded and skipped; it never stops the batch. No Streamlit.
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
) -> Dict:
    store.init_schema(db_path)
    steps = tuple(steps)
    repeats = max(1, min(int(repeats), 5))
    est = bloom_llm.estimate_cost_usd(len(items), steps=len(steps), repeats=repeats if "blind" in steps else 1)
    inst = items[0]["instrument_key"] if items else ""
    run_id = store.create_run(created_by, bloom_llm.MODEL, bloom_llm.PROMPT_VERSION, inst,
                              ",".join(steps), repeats, len(items), est, db_path)
    spent, n_ok, n_err = 0.0, 0, 0
    for i, item in enumerate(items):
        if progress:
            progress(i, len(items), item["question_id"])
        if "blind" in steps:
            for rep in range(repeats):
                r = bloom_llm.rate_blind(item, effort=effort)
                store.save_llm_result(run_id, item, "blind", rep, r, bloom_llm.MODEL, bloom_llm.PROMPT_VERSION, db_path)
                spent += bloom_llm.cost_from_usage(r["tokens_in"], r["tokens_out"])
                n_ok, n_err = n_ok + int(r["ok"]), n_err + int(not r["ok"])
        if "key" in steps:
            r = bloom_llm.review_key(item, effort=effort)
            store.save_llm_result(run_id, item, "key", 0, r, bloom_llm.MODEL, bloom_llm.PROMPT_VERSION, db_path)
            spent += bloom_llm.cost_from_usage(r["tokens_in"], r["tokens_out"])
            n_ok, n_err = n_ok + int(r["ok"]), n_err + int(not r["ok"])
    if progress:
        progress(len(items), len(items), "")
    store.finish_run(run_id, spent, "done" if n_err == 0 else "done_with_errors", db_path)
    return {"run_id": run_id, "n_ok": n_ok, "n_err": n_err, "cost_usd": spent, "est_cost_usd": est}
