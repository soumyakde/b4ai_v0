# Theme Comparison Methodology Guide

*Companion reference for the "Compare Runs" feature in LLM Analysis → Inductive Thematic Analysis (ITA). Explains what it measures, how it works, and — deliberately as much space as the "how" — what it does not tell you.*

---

## 1. What this feature is for

When you run ITA more than once — a different model, a different temperature, or a re-run of the same settings — each run produces its own independent set of themes. "Compare Runs" answers a narrow, specific question: **how similar are the two theme sets, as text?** It does not tell you whether either theme set is a good or faithful analysis of your data. That distinction matters and is revisited throughout this guide.

This directly answers two of De Paoli's (2024) research questions — do different models produce consistent themes, and does temperature affect them — using an original agreement metric built for this app. De Paoli's own paper does not specify a quantitative agreement formula; his comparison method is a qualitative read of theme names and descriptions. The metric described below is this codebase's own contribution, not an adaptation of his.

## 2. How a comparison is computed

For two theme sets (Run A and Run B), each theme is turned into a short piece of text (its name plus description), and every Run A theme is compared against every Run B theme two ways:

- **Cosine similarity** — semantic closeness, via sentence embeddings (`all-MiniLM-L6-v2` by default — see §5 on why this specific choice matters).
- **Jaccard similarity** — lexical overlap of the words used in each theme's name/description.

These combine into one **agreement score** per pair: `0.7 × cosine + 0.3 × Jaccard`. This weighting is this app's own choice, not derived from a validated instrument — treat it as a working default, not an authoritative formula.

### Matching: Hungarian algorithm (Kuhn, 1955)

Given the full agreement-score grid (every Run A theme × every Run B theme), the app needs to decide *which* Run B theme each Run A theme should be compared against. As of this update, it uses the **Hungarian algorithm** — the optimal one-to-one assignment across the whole grid at once (via `scipy.optimize.linear_sum_assignment`).

**Why this changed:** the previous approach let each Run A theme independently pick whichever Run B theme scored highest against *it alone* — a greedy, per-row decision with no global constraint. Concretely, if two different Run A themes were both semantically closest to the *same* Run B theme, both would "match" it, and both would report a correspondingly high agreement score — even though only one of them can genuinely correspond to that theme. This silently inflated apparent agreement whenever a run under- or over-generated a topic relative to the other run. The Hungarian algorithm removes this by finding the best assignment *across the whole set simultaneously*, so no Run B theme can be claimed by more than one Run A theme.

### The `matched` flag and the 0.30 floor

Even with an optimal assignment, some assigned pairs may still be weak — the best available partner isn't necessarily a *good* partner. Any assigned pair scoring below **0.30** is flagged `matched: False` in the results table. Two things to note about this:

- **The 0.30 floor is a provisional placeholder, not an empirically derived cutoff.** It has not yet been calibrated against any ground truth. A planned follow-up (see §6) will set this value properly using ROC analysis and Youden's J statistic against a human-coded subset, once one exists. Until then, treat 0.30 as a rough sorting aid, not a validated threshold.
- **A pair's actual score still counts toward the overall average, even when flagged unmatched.** The flag is informational, not a filter — silently dropping weak matches from the average would inflate the reported agreement by hiding the runs' real disagreements (a form of survivorship bias). If Run A generated more themes than Run B, the surplus Run A themes get *no* assignment at all (`best_match_b: None`) — which is itself real information: it means the two runs didn't even converge on the same number of underlying ideas, a stronger form of disagreement than two themes just being described differently.

### The symmetry fix

The overall comparison between two runs is not perfectly symmetric — matching Run A's themes onto Run B's can, in principle, produce a different bipartite assignment (and therefore a different score) than matching Run B's onto Run A's, especially when the two runs proposed different numbers of themes. When comparing more than two runs at once (`compare_all_runs`), the app now computes **both directions independently** and averages them for the summary table, rather than computing one direction and mirroring it into both cells (the previous behavior, which silently assumed a symmetry that isn't guaranteed). Both raw directional results remain available for inspection, not just the average.

## 3. Reading the results table

| Field | Meaning |
|---|---|
| `theme_a` | The Run A theme being reported on. |
| `best_match_b` | Its Hungarian-assigned Run B counterpart, or `None` if it received no assignment at all. |
| `cosine`, `jaccard` | The two raw similarity components for that specific assigned pair. |
| `agreement` | The weighted composite score for that pair. |
| `matched` | `True` if `agreement ≥ 0.30` (provisional floor — see above), else `False`. |
| `overall_agreement` | The mean of every Run A theme's assigned score (unmatched pairs included, per §2). |
| `interpretation` | A qualitative label — see the caveat immediately below. |

## 4. Why the "strong/moderate/weak/poor" labels deserve skepticism

The interpretation bands (≥0.80 strong, ≥0.65 moderate, ≥0.50 weak, below poor) look like the familiar Landis & Koch (1977) kappa benchmarks, but they are **not** kappa, and these specific cutoffs are not derived from any validation study of this metric — they are this app's own invented convention. This matters because the same criticism leveled at Landis & Koch's own bands applies here with equal force: fixed verbal labels on a continuous statistic have no sound theoretical basis unless they're calibrated against something (Ludbrook, 2002). Prefer reading the raw `overall_agreement` number and, where you can, an actual confidence interval, over trusting the qualitative label at face value.

## 5. What this metric does *not* tell you — the limitations

This is the part worth reading carefully before citing a "Compare Runs" result as evidence of anything.

- **Agreement is not validity.** Two models agreeing with each other means they converged on similar output — it does not mean either one is a faithful or accurate reading of your data. If two models share the same blind spot, agreement goes up while validity goes down. A genuine validity claim needs comparison against human-coded themes, which this feature does not do (see §6, Phase 6).
- **No correction for chance agreement.** Unlike Krippendorff's alpha or Cohen's kappa, this metric has no built-in adjustment for the fact that two theme sets drawn from a narrow, shared-vocabulary domain (e.g., everything here is about "AI," "learning," "engagement") will show non-trivial lexical and semantic overlap by default, regardless of whether the themes are actually the same idea. A high score can partly reflect shared domain vocabulary, not genuine convergence.
- **One embedding model's geometry.** By default, similarity is computed entirely through `all-MiniLM-L6-v2` — a specific, fairly small local model. Per the MTEB benchmark (Muennighoff et al., 2023), no single embedding model dominates across tasks, so a pattern that only holds under one model's specific geometry is a weaker claim than one that holds under two independent ones. A second backend (`embedding_backend="openai"`, using `text-embedding-3-small`) is available in the code as a robustness check, but is **not** exposed in the dashboard — it costs real OpenAI API money per call, so it's meant for a deliberate, one-off validation pass (e.g., for a methods paper), not something a routine dashboard click should trigger. If you want to run this robustness check, it needs to be called directly in code, not through the app.
- **No self-consistency baseline.** A cross-model agreement score is hard to interpret in isolation — is 0.65 good or bad? That depends on how much a *single* model's output varies against itself on repeated runs at the same settings, which nothing here currently measures. Even at temperature 0, LLM outputs are not guaranteed to be identical across runs. Without that same-model baseline, there's no ceiling to compare a cross-model score against.
- **The 0.30 match floor, the 0.7/0.3 cosine-Jaccard weighting, and the strong/moderate/weak/poor bands are all uncalibrated defaults**, not values derived from this app's own data or from any external validation study.

## 6. What's still planned

These limitations aren't being ignored. Some now have built, tested infrastructure sitting ready for real data; others are still sequenced work.

- **Ground-truth upload + reliability infrastructure — built (`core/analytics/validation/ground_truth_validation.py`), synthetic-tested, not yet run on real data.** A documented CSV/XLSX template, a validator/normalizer for uploaded ground-truth files, a Krippendorff's alpha function, and a ROC/Youden's J calibration function all exist and pass a battery of synthetic checks (perfect agreement, near-random agreement, partial rater overlap, perfectly-separable and noisy ROC cases, and the relevant error conditions). None of them has been run against real ITA/DTA output yet — no matching completed run exists to validate against. The Teacher Dashboard's LLM Analysis tab has a "🧪 Validation (Research)" section (visible to any teacher-role user, no role gate, clearly labeled) where the ground-truth upload and preview already work; the actual alpha/ROC computations are explicitly labeled "To Be Developed" there rather than faked against placeholder numbers.
- **Krippendorff's alpha** is built generically (any two-rater, shared-unit-set comparison) and applies cleanly to the DTA deductive pipeline's fixed-codebook coding — but not to this document's free-form ITA theme comparison, where alpha's shared-category-scheme assumption doesn't hold without first reconciling both runs onto one common theme list. It also doesn't yet have matching data: the existing human-coded interview document (see `scripts/convert_interview_codes_docx.py`) uses its own inductive categories, not DTA's actual construct taxonomy (coherency_of_messaging, engagement_with_task, etc.) — a real DTA-taxonomy-coded human dataset doesn't exist yet.
- **ROC/Youden's J calibration** of the 0.30 floor is built and ready, but needs two things that don't exist yet: a completed ITA run on the same transcripts as the human-coded document, and human match/no-match judgments on candidate LLM-theme-vs-human-code pairs (a labeling task, not just a file upload).
- **Same-model repeated-run baseline and factorial decomposition — built (`theme_comparator.decompose_factorial_agreement()`), synthetic-verified, not yet run on real data.** Splits a `compare_all_runs()` result into four groups — same-model/same-temperature (the self-consistency ceiling), same-model/different-temperature (isolates a temperature effect), different-model/same-temperature (isolates a model effect), and different-model/different-temperature (both, least interpretable) — each with a mean agreement and a 95% CI, rather than one convenience matrix that confounds model identity and temperature together (Mizrahi et al., 2024). See §7 below for how to actually run the study this is built to analyze.

## 7. Running the calibration study, when you're ready

This is a real decision about spending API budget, not something to do casually — here's a concrete protocol for when you are ready.

**Design.** Pick 2–3 models already available in this app (Claude, Gemini, GPT, Groq) and 2 temperatures per model (e.g., T=0, matching De Paoli's deterministic default, and T≈0.5–1.0, matching his "reviewing themes" comparison point). For every (model, temperature) cell, run ITA **at least 3 times** — ideally 5 if budget allows — on the *same* transcript set each time, changing nothing else between replicates. Fewer than 3 same-settings replicates means `decompose_factorial_agreement()`'s self-consistency group has no computable confidence interval (n<2), which defeats the purpose.

**Data side.** Reuse whichever transcript set you're already using for pilot analysis — this study is about *model behavior stability*, not about the representativeness of a data sample, so it doesn't need a separate or larger dataset than what you already have.

**Labeling.** Give each run a label that encodes both model and temperature plus a replicate marker (e.g., `"Claude T=0 (r1)"`, `"Claude T=0 (r2)"`, `"Claude T=0.5 (r1)"`, ...), and build a matching `run_metadata` dict: `{label: {"model": ..., "temperature": ...}}`. Both `compare_all_runs()` and `decompose_factorial_agreement()` need this.

**Cost check first.** The LLM Analysis tab already has a cost estimator (`_ita_cost_estimate()`) shown before you run anything — multiply its per-run estimate by your total grid size (e.g., 3 models × 2 temperatures × 3 replicates = 18 runs) before committing.

**Running the analysis.** Once every run has completed through Phase 3, collect each run's `themes` list into a `runs` dict keyed by your labels, call `compare_all_runs(runs)`, then `decompose_factorial_agreement(result, run_metadata)`. The self-consistency ceiling (`same_model_same_temp`) is the number every cross-model or cross-temperature score in the other three groups should actually be judged against — not an assumed 1.0.

## References

- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly, 2*(1–2), 83–97. https://doi.org/10.1002/nav.3800020109
- De Paoli, S. (2024). Performing an inductive thematic analysis of semi-structured interviews with a large language model: An exploration and provocation on the limits of the approach. *Social Science Computer Review, 42*(4), 997–1019. https://doi.org/10.1177/08944393231220483
- Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark. *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics*.
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics, 33*(1), 159–174.
- Ludbrook, J. (2002). Statistical techniques for comparing measurers and methods of measurement: A critical review. *Clinical and Experimental Pharmacology and Physiology, 29*(7), 527–536.
- Hayes, A. F., & Krippendorff, K. (2007). Answering the call for a standard reliability measure for coding data. *Communication Methods and Measures, 1*(1), 77–89.
