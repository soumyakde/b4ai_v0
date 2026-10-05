"""
core/analytics/item_review/agreement.py

Agreement statistics for two raters' ordinal Bloom levels (human vs LLM, or human vs human).
Pure numpy; unit-tested in scripts/verify_item_review.py.

Weighted kappa: Cohen (1960) kappa generalised with linear or quadratic disagreement weights
(Cohen, 1968, Psychological Bulletin, 70(4), 213-220). Interpretation bands: Landis & Koch (1977),
Biometrics, 33(1), 159-174 -- shown as a rough guide only; with few items the interval is wide, so
a bootstrap 95% interval is reported.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def weighted_kappa(a: Sequence[int], b: Sequence[int], levels: Sequence[int] = (1, 2, 3, 4, 5, 6),
                   weights: str = "quadratic") -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(a) != len(b):
        return float("nan")
    k = len(levels)
    idx = {l: i for i, l in enumerate(levels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    n = O.sum()
    E = np.outer(O.sum(1), O.sum(0)) / n
    i, j = np.meshgrid(range(k), range(k), indexing="ij")
    W = (np.abs(i - j) / (k - 1)) if weights == "linear" else ((i - j) ** 2 / (k - 1) ** 2)
    denom = (W * E).sum()
    if denom == 0:
        return float("nan")          # both raters used the same single level for every item: kappa undefined
    return float(1 - (W * O).sum() / denom)


def kappa_band(k: float) -> str:
    if k != k:
        return "undefined"
    for cut, name in ((0.8, "almost perfect"), (0.6, "substantial"), (0.4, "moderate"), (0.2, "fair"), (0.0, "slight")):
        if k > cut:
            return name
    return "poor"


def agreement_summary(a: Sequence[int], b: Sequence[int], n_boot: int = 2000, seed: int = 7) -> Dict:
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    out = {"n": n, "exact": float("nan"), "within1": float("nan"), "kappa_w": float("nan"),
           "kappa_lo": float("nan"), "kappa_hi": float("nan"), "band": "undefined"}
    if n == 0:
        return out
    out["exact"] = float(np.mean(a == b))
    out["within1"] = float(np.mean(np.abs(a - b) <= 1))
    out["kappa_w"] = weighted_kappa(a, b)
    out["band"] = kappa_band(out["kappa_w"])
    if n >= 8:
        rng = np.random.default_rng(seed)
        ks = []
        for _ in range(n_boot):
            s = rng.integers(0, n, n)
            k = weighted_kappa(a[s], b[s])
            if k == k:
                ks.append(k)
        if len(ks) > 50:
            out["kappa_lo"], out["kappa_hi"] = float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))
    return out
