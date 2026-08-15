# Railway Test-Environment Checklist — Post-pilot enhancements (2026-08-04, updated 2026-08-15)

**Purpose:** confirm the requested changes (Tasks A-L, plus fixes found while testing) work correctly on the real deployed Railway `test` environment. Underlying logic was already verified directly (hand-computed math checks, full data round-trip tests, live SSH confirmation of the bug fix, direct-Python and AppTest verification) — this is your pass to confirm it looks and behaves right from the actual dashboard.

**URL:** https://basics4ai-staging-test.up.railway.app
**Login:** your own admin account (the `test` environment's database was refreshed from production on 2026-08-04, so your real credentials work here too)

Estimated time: ~110 minutes (Tasks A-D: ~15 min, Task E: ~10 min, Task J: ~15 min, Task K: ~20 min, Task L: ~30 min). Tasks A-K already signed off and live in production — skip unless you want a re-check; Task L (below) is new.

---

## A. "Cognitive Engagement" label (2 min)

1. Log in, open the **Teacher Dashboard**.
2. Go to **📊 Basic Statistics** (should now be the 1st tab — see item D below).
3. Scroll to **Survey Construct Means (Likert 1-4)**.
4. Open the **"Select Survey"** dropdown — confirm it reads **"Cognitive Engagement"**, not "SCCCES (Conceptual Change)".
5. Select it, switch the chart view to **"By Question"** — confirm the chart title reads **"Mean Score Per Question Item — Cognitive Engagement"**.
6. ✅ Pass if: both places show the new label, no "SCCCES" text visible anywhere in this section.

## B. Bland-Altman Diff (Post−Pre) (3 min)

1. Go to **📈 Inferential Statistics** → the Pre vs Post / method-agreement section.
2. For **AI Misconceptions**, expand **"📐 Method agreement — Bland-Altman limits of agreement"**.
3. Open the inner **"Individual differences"** expander — confirm the column header now reads **"Diff (Post−Pre)"**.
4. Spot-check one row: if a student's Post % was higher than their Pre %, that row's Diff value should now be **positive** (previously it would have shown negative under the old Pre−Post convention).
5. Read the interpretation sentence above the table — confirm it says something like *"Post scores exceeded pre on average (gain post-intervention)"* when the bias is positive, and the opposite when negative.
6. Repeat steps 2-5 for **AI Conceptual Inventory (AICI)**.
7. ✅ Pass if: both instruments show "Diff (Post−Pre)", the sign direction matches what you'd expect from a real improving student, and the interpretation sentence's wording matches the sign.

## C.1 — Groq error fixed, + 2 bugs found during your 2026-08-08 test pass, now fixed (5 min)

1. Go to **🤖 LLM Analysis** (should now be the 3rd tab — see item D).
2. Pick either **Inductive Thematic Analysis (ITA)** or **Deductive Thematic Analysis (DTA)**.
3. In the model selection step, select **Groq**, and select at least one data source with available text.
4. Run a small analysis (1-2 texts is enough to confirm it works — keep it small to control cost/quota).
5. ✅ Pass if: you do **not** see "Groq API error: No module named 'groq'" — a real result (or a different, unrelated error like a rate limit) means the fix worked.
6. **Re-test the ITA "sentence-transformers is not installed" crash you hit at Phase 2b (Deduplicating codes):** run an ITA analysis through to completion (any model). Root cause: `sentence-transformers` was present in the last known-good backup's `requirements.txt` but had dropped out of the current repo's — a real regression, now re-added. ✅ Pass if: Phase 2b completes without the `RuntimeError` (first run may take a little longer — it downloads a small ~90MB model from Hugging Face the first time; this repeats after every scheduled redeploy since the container filesystem is ephemeral, which is expected, not a bug).
7. **Re-test the "counts don't update with the cohort filter" bug you found:** in ITA or DTA Step 1, select a cohort filter, then look at the "Reflections/Interviews/Observer transcripts available" metrics in the next step. ✅ Pass if: the metric labels show the cohort name (e.g. "Reflections available (amherstyouthandrec2D)") and the numbers reflect only that cohort — not the total across everyone. Try switching cohorts or clearing the filter and confirm the counts update each time.

## C.2 — Observer/instructor transcripts (4 min)

1. Go to **Admin Dashboard → Research Operations**.
2. Scroll past the existing "Interview Transcript Store" — confirm a new **"📂 Observer/Instructor Transcript Store"** section appears below it, with its own upload/list/delete controls.
3. Upload a small `.txt` test file, assign it a real participant ID, submit.
4. Confirm it appears in the Observer/Instructor store's table — and confirm it does **not** appear in the Interview Transcript Store's table above it (they should stay separate).
5. Go back to **Teacher Dashboard → 🤖 LLM Analysis → (ITA) Step 1 — Select Data Source**.
6. Confirm a third checkbox now exists: **"Observer/instructor transcript(s) (store)"**, and that checking it shows a metric with your uploaded file counted.
7. Delete your test upload from the Observer/Instructor Transcript Store when done (use "Delete a transcript", not "Clear all" — that would remove real data if any exists).
8. ✅ Pass if: the new store is genuinely separate from the interview one, your test file shows up and is countable, and you can clean it up afterward with no error.

*(The underlying bug fix — transcripts now actually persisting to the real database instead of silently vanishing on redeploy — was already verified directly via SSH and doesn't need a UI check, but if you want extra confidence: upload a file here, then check back after a few hours to confirm it's still there.)*

## D. Tab order + IRT note (2 min)

1. On the Teacher Dashboard, confirm the top section-selector reads, left to right: **Basic Statistics, Inferential Statistics, LLM Analysis, Competency Progression, IRT Analysis, Report Generation**.
2. Click into **🔬 IRT Analysis** — confirm a note appears near the top: *"🚧 This section is functional but earmarked for further development."*
3. ✅ Pass if: the order matches exactly and the note is visible.

---

## E. Cohort tagging for interview/observer transcripts (10 min)

**Purpose:** lets you tag pre-pilot interview/observer transcripts (recorded before the platform existed, with no registered-user record) with a cohort_id, so they can be filtered and reported on by cohort just like the rest of your pilot data.

### E.1 — Cohort selector on upload, re-test (2026-08-09) of 3 bugs you found (5 min)

Your first pass found 3 real bugs here — all fixed now:
- The "cohort created" success message never appeared (it was being wiped by an immediate rerun before it could paint) — fixed, and the new cohort is now auto-selected in the dropdown too.
- Changing the bulk "Cohort (applies to all files below)" selector after files were already listed had no effect on the per-file dropdowns (they kept showing "None") — fixed, they now correctly re-sync to the bulk choice.
- No way to clear the form / start a new batch — added a **"🔄 Clear form"** button, and the uploader now also auto-clears itself after a successful save.

1. Go to **Admin Dashboard → Research Operations → Interview Transcript Store**.
2. In the upload section, confirm a **bulk "Cohort" selector** appears above the file list, with your existing cohorts as options plus a **"+ Create new cohort…"** choice.
3. Pick **"+ Create new cohort…"**, type a throwaway test cohort ID (e.g. `TestCohortE`), click **"Create cohort"**. ✅ Pass if: a green success message appears immediately, AND the bulk selector jumps straight to your new cohort (no need to find/reselect it).
4. Upload 2 small `.txt` test files. Leave participant IDs as suggested. **Now change the bulk "Cohort" selector to something else** (or back to "— none —"). ✅ Pass if: both per-file cohort dropdowns update to match your new bulk choice — you should NOT have to manually re-set each one.
5. Click **"🔄 Clear form"** without uploading. ✅ Pass if: the file list clears and you can select fresh files (nothing was saved).
6. Re-select your 2 files, set a cohort, click **"📤 Upload transcripts to store"**. ✅ Pass if: a success message appears AND the uploader is empty again afterward, ready for a new batch without you doing anything.
7. Confirm the transcript list table now shows a **"Cohort"** column with your chosen cohort for those rows.
8. Repeat steps 2-7 for **Observer/Instructor Transcript Store** — confirm it works identically and independently (its own bulk selector, its own per-file override, its own Cohort column, its own Clear form button).
9. ✅ Pass if: all of the above hold for both transcript stores.

### E.2 — Cohort filter in LLM Analysis (3 min)

1. Go to **Teacher Dashboard → 🤖 LLM Analysis**.
2. In either **ITA Step 1** or **DTA**'s data-source step, confirm a new **"Filter by cohort (optional...)"** multiselect appears below the source checkboxes, listing your existing cohorts (including the test one from E.1).
3. Select your test cohort — confirm the source counts update to reflect only that cohort's data.
4. For ITA specifically: advance through the wizard steps and back — confirm the cohort selection is preserved (doesn't reset).
5. Leave the filter empty and confirm all data (unfiltered) loads as before — this is the "no cohort filter" default and should behave exactly like it did before this update.
6. ✅ Pass if: the filter narrows results correctly, ITA's selection survives step navigation, and leaving it empty still includes everything (no accidental default-narrowing).

### E.3 — Report Generation shows cohort scope (3 min)

1. Run a small ITA or DTA analysis with your test cohort filter applied (from E.2) — just enough to complete a run.
2. Go to **Teacher Dashboard → Report Generation → 🤖 LLM Analysis**.
3. Open the run selector — confirm your new run's label includes the cohort you filtered to (e.g. "... — TestCohortE"), and that older runs from before this update show **"All cohorts"**.
4. Generate the PDF — confirm the header section includes a **"Cohort scope:"** line matching what you selected.
5. ✅ Pass if: the run picker and the generated PDF both correctly show which cohort(s) the run covers.

### E.4 — Cohort deletion, new (5 min)

You noticed there was no way to remove a cohort you'd created by mistake — now there is, gated so it can never orphan a reference.

1. Go to **Admin Dashboard → User Management → Cohort Management**.
2. Confirm a **"Delete a cohort"** section appears below "Create Cohort", with a dropdown of your existing cohorts.
3. Pick a cohort you know is **still in use** (e.g. `test`/`test2`/whatever you tagged transcripts with earlier) — ✅ pass if a warning shows exactly how many users/transcripts reference it, and **no delete button appears**.
4. Create a fresh throwaway cohort (e.g. `TestCohortDelete`) via "Create Cohort" above — confirm the success message now appears correctly (this reuses the same fix as the upload-form flash-message bug).
5. Select it in the "Delete a cohort" dropdown — ✅ pass if it says "unused — safe to delete" and a delete button appears.
6. Click it — ✅ pass if a success message appears and the cohort is gone from every cohort dropdown in the app (Cohort Management, the upload forms, LLM Analysis's cohort filter).
7. ✅ Pass if: an in-use cohort is correctly blocked with an accurate count, and an unused one deletes cleanly.

### Cleanup

Delete your `test`/`test2`/`TestCohortE` test transcripts (Interview and Observer stores) using "Delete a transcript", and now also any leftover test cohorts via the new "Delete a cohort" control (E.4) once they're unused — this is all on the disposable `test` environment, so it's optional, but keeps things tidy for later testing.

---

## F. Competency Progression Index — multi-module, multi-instrument, cohort scope (10 min)

You asked for the CPI tab's Step 1 to combine one or more modules' content assessments plus AI Misconceptions/AICI gain, and to show cohort scope. Here's what changed and what to check.

### F.1 — Step 1 configuration (4 min)

1. Go to **Teacher Dashboard → 📉 Competency Progression → Step 1 — Configure**.
2. Confirm a **"Modules"** multiselect appears (not a single dropdown), defaulting to all your modules selected.
3. Confirm **3 checkboxes** appear: "Content MCQ" (checked by default), "AI Misconceptions gain (Post−Pre)", "AICI gain (Post−Pre)".
4. Confirm a **"Cohort scope:"** caption appears, showing whichever cohort(s) your sidebar's Cohort filter is currently set to (or "All cohorts" if none selected). Try changing the sidebar's Cohort filter and confirm this caption updates to match.
5. ✅ Pass if: all of the above render correctly and the cohort caption tracks the sidebar filter.

### F.2 — Single-module classic behavior still works (2 min)

This confirms nothing broke for your existing single-module workflow.

1. In the Modules multiselect, narrow it down to **exactly one module**, and leave only "Content MCQ" checked.
2. Confirm the original **CPI_quant method** radio (CTT / IRT Rasch / IRT 2PL / Both) and the module/instrument-key read-only fields reappear — this is the original behavior, unchanged.
3. Click **"Compute CPI_quant"** — ✅ pass if the results look exactly like they did before this update (same numbers, "CTT"-labeled metrics).

### F.3 — Multi-module / multi-instrument combining (3 min)

1. Select **2 or more modules**, and check **all 3 instrument-type boxes** (Content MCQ + both gain scores).
2. Confirm the CPI_quant method radio and IRT options **disappear** (replaced by a "Combining: ..." caption listing your selected modules and instrument types) — IRT modeling isn't offered for a pooled multi-instrument selection.
3. Click **"Compute CPI_quant"** — ✅ pass if it completes with no error, showing "Outcome"-labeled metrics (not "CTT").
4. Continue to Step 3, score a few reflections, then Step 4 — confirm CPI+ computes and the per-student table populates as before.
5. ✅ Pass if: switching between single-module and multi-module/multi-instrument selections both work, with no errors either way.

### F.4 — Report Generation shows CPI run scope (2 min)

1. Go to **Teacher Dashboard → Report Generation → v. Competency Progression**.
2. Confirm a **run picker** now appears above the report options (previously this always silently used your most recent run with no way to choose) — its labels show each run's module/instrument scope.
3. Generate the PDF — confirm the **Methodology** section describes the actual modules/instruments your selected run used, not a generic fixed description.
4. ✅ Pass if: you can pick a specific past run, and the report accurately describes what it covers.

---

## G. Registration confirmation + duplicate-account prevention (5 min)

You reported that impatient students would create multiple accounts while waiting for approval. The likely root cause was found and fixed: the "registration successful" message never reliably reached the screen, and the redirect to the Login page silently failed.

1. Log out (or open a private/incognito window) and go to the **Register** page.
2. Register a throwaway test account (e.g. `TestRegE2E1`).
3. ✅ Pass if: a clear green banner appears — *"Registration successful for 'TestRegE2E1'! Please wait for Admin approval..."* — AND the page has switched to the **Login** view (not still showing the registration form).
4. Reload the page — ✅ pass if the banner does **not** reappear (it's a one-time confirmation, not a persistent message).
5. Try registering the **same username with different capitalization** (e.g. `testrege2e1`) — ✅ pass if it's rejected as already taken (this used to silently succeed as a second, separate account).
6. Clean up: delete `TestRegE2E1` via Admin Dashboard → User Management once done testing (optional, disposable `test` environment).

---

## H. PDF report table overflow fix, re-test (2 min)

You found this in F.4's CPI report (module_id spilling into the Band column) and independently in the ITA report (Participant filename spilling into Code Name, Description running past the page margin). Root cause was shared by every report — fixed once in the common PDF-building code.

Your re-test of the ITA report then found a **second, separate issue**: the Description column looked cut off mid-sentence with no visible wrapping. That turned out to be unrelated to the wrapping fix — the ITA report's own code was hard-truncating each description to 80 characters before it ever reached the PDF builder, a leftover from before wrapping worked. Fixed by removing that truncation.

1. Regenerate the **CPI+ report** (Report Generation → v. Competency Progression) from a run with a multi-module scope — ✅ pass if the `module_id`/scope column wraps cleanly within its own column, no longer bleeding into Band.
2. Regenerate an **ITA report** (Report Generation → iv. LLM Analysis) with a transcript that has a long filename/participant ID — ✅ pass if Participant, Code Name, and Description all stay within their own columns and margins, wrapping onto multiple full lines (not cut off) instead of overflowing or truncating.
3. ✅ Pass if: both reports render cleanly with no column bleed-through and no truncated/cut-off text anywhere in the tables.

---

## I. ITA report — representative quotes cited per theme (2 min)

You pointed out the report wasn't reliably citing significant participant quotes per theme. Found the cause: Phase 6's prompt only asked the LLM to include quotes "where available" in free-flowing prose — a soft request it didn't reliably follow. Fixed by deterministically listing up to 3 attributed, verbatim quotes under each theme in the PDF, independent of what the narrative prose does.

1. Regenerate an **ITA report** (Report Generation → iv. LLM Analysis) from a completed run.
2. Under each **Theme N: ...** section, confirm a **"Representative quotes:"** block appears below the summary, listing up to 3 verbatim quotes, each attributed to a participant (e.g. `"quote text" — username`).
3. A theme with no codes carrying a quote should simply show no quotes block (not an error or empty section) — this is expected, not a bug.
4. ✅ Pass if: quotes are now reliably present (when the underlying data has them) and correctly attributed, for every theme.

---

## J. Data-quality checks (normality, straight-lining, missingness) + exclusion options (15 min)

Your dissertation methodology calls for normality checks (plus other data-quality screens relevant to survey fatigue) before any inferential test. Basic Statistics now surfaces these, and Inferential Statistics lets you optionally exclude flagged participants per criterion before running a test — each with a citation explaining why.

### J.1 — Basic Statistics: Data Quality & Distributional Checks (5 min)

1. Go to **Teacher Dashboard → 📊 Basic Statistics**.
2. Scroll past **Survey Construct Means** — confirm a new **"🔍 Data Quality & Distributional Checks"** section appears, with 3 expanders.
3. Open **"📐 Normality checks (Shapiro-Wilk)"** (expanded by default) — confirm a table listing every data source (all 7 modules' Content MCQ, AI Misconceptions Pre/Post, AICI Pre/Post, each Cognitive Engagement/SIMS construct), with columns N, Shapiro W, p-value, Skewness, Kurtosis, Verdict. Groups with n<20 should show "-- exploratory, n<20" in the verdict.
4. Open **"🚩 Straight-lining (careless-responding flags)"** — confirm a flagged-count metric and a table of flagged (student, survey) pairs, plus a citation (Meade & Craig, 2012) underneath.
5. Open **"📉 Participation trend across modules (missingness)"** — confirm a table and line chart showing % of cohort participating per module 1-7 (Module 7 will legitimately show near-0%, since it's disabled — not a bug).
6. Scroll up to **Assessment Scores → Student view** and **Survey Construct Means → By Student (distribution)** — confirm a small normality caption (W, p, skew, kurtosis, verdict) appears under each histogram.
7. In the sidebar, filter down to a **single student** — confirm the Data Quality section degrades gracefully (shows "too few students to test" or similar), not an error.
8. ✅ Pass if: all 3 expanders render correctly for every data source, single-student filtering doesn't crash, and the participation-trend chart shows a believable declining/flat pattern.

### J.2 — Basic Statistics PDF report includes data quality (3 min)

1. Go to **Report Generation → i. Basic Statistics**.
2. Confirm **"Data quality & normality checks"** is included in the section checklist (checked by default).
3. Generate the PDF — confirm it includes a normality table, straight-lining summary, and missingness trend, matching what you saw live in J.1.
4. ✅ Pass if: the PDF section renders cleanly with no column overflow or truncation (same fix as Item H).

### J.3 — Inferential Statistics: exclude flagged participants (7 min)

Before each test, you can now optionally exclude participants who fail a data-quality criterion — outliers, excessive missing data, and (for survey instruments only) straight-lining. Each option shows how many participants it would exclude and cites the source for why it matters.

1. Go to **📈 Inferential Statistics → Pre vs Post**. Above the "Computing…" spinner, confirm a **"🧪 Exclude flagged participants before running this test"** expander appears.
2. Open it — confirm checkboxes for "Exclude outliers" and "Exclude participants with excessive missing data" each show a flagged count and a citation underneath (Osborne & Overbay 2004 / Tabachnick & Fidell for outliers; Little & Rubin / Enders 2010 for missingness).
3. Check **"Exclude outliers"** — if the count is > 0, confirm a warning appears listing which participant(s) were excluded, and the test result above updates (fewer participants counted, numbers may shift slightly). If the count shows **0 flagged**, that's a legitimate result for that instrument pair — try a different instrument or move to step 4 to see a non-zero case.
4. Go to **Between Groups** — confirm the same expander appears, scoped to whichever instrument you pick in the dropdown above it.
5. Go to **Across Modules**, select **"MCQ content knowledge"** — confirm the expander shows **only 2 checkboxes** (outliers, missing data) — no straight-lining option, since MCQ correctness isn't a carelessness signal.
6. Still in **Across Modules**, switch to **"Survey construct"** — confirm the expander now shows **3 checkboxes**, including "Exclude straight-lining respondents" with the Meade & Craig / Curran citation.
7. ✅ Pass if: the expander appears in all 3 sections, straight-lining is correctly hidden for MCQ-only comparisons and shown for survey comparisons, and checking a box with a non-zero flagged count visibly changes the test result and shows an audit-trail warning of who was excluded.

### Cleanup

None needed — this section is read-only/diagnostic (no new data is written), and the exclusion checkboxes reset each time you leave the tab.

### J.4 — Re-test: Across Modules module picker + "Cognitive Engagement" label fix (5 min)

While reviewing J.3, you asked why **Across Modules — Repeated Measures (Friedman Test)** showed **n = 8 students** for every module combination. Root cause confirmed directly against the data: the Friedman test requires complete data across **every** module it's given, and Module 7 (disabled for everyone except the 8-person `amherstyouthandrec1D` cohort) was always silently included — collapsing every other student out of the comparison. Fixed by adding a module picker, defaulting to currently-active modules only (excludes Module 7 by default; can still be added back deliberately). Same fix applied to the Inferential Statistics PDF report, which had the identical silent collapse with no UI to catch it. Also fixed while in this code: the "Survey" selectbox in this section (and in IRT Analysis's Likert Survey GRM section) still said **"SCCCES"** instead of **"Cognitive Engagement"** — a leftover from Item A's original rename that this selectbox wasn't caught by at the time.

1. Go to **📈 Inferential Statistics → Across Modules**, select **"MCQ content knowledge"**. Confirm a new **"Modules to include"** multiselect appears, defaulting to **Modules 1-6** (Module 7 not pre-selected). Confirm the **N subjects** metric now shows a much larger number than before (expect ~95, not 8).
2. Manually add **Module 7** back into the multiselect — confirm N subjects drops back down to a small number (the Amherst-only complete-case result) — this confirms the picker genuinely controls the test, it's not just cosmetic.
3. Switch to **"Survey construct"** — confirm the **"Survey"** dropdown now reads **"Cognitive Engagement"**, not "SCCCES". Confirm the same **"Modules to include"** multiselect appears here too, defaulting to Modules 1-6, with N subjects now much higher than 8 (expect ~86 for Cognitive Engagement's Engagement with Task construct).
4. Go to **🔬 IRT Analysis → Likert Survey — Graded Response Model (GRM)** — confirm its **"Survey"** dropdown also now reads "Cognitive Engagement", not "SCCCES".
5. Go to **Report Generation → ii. Inferential Statistics**, generate the PDF — confirm it completes without error (this report's Across Modules section now uses the same active-modules-by-default scoping, with no UI needed there since it's a static report).
6. ✅ Pass if: both module pickers default sensibly and visibly change N when you add/remove modules, "Cognitive Engagement" appears everywhere "SCCCES" used to, and the PDF report still generates cleanly.

---

## K. Data-driven test selection: recommendations, 2-group t-test/Mann-Whitney U, RM-ANOVA + sphericity (20 min)

Your dissertation methodology calls for the appropriate inferential test to be determined by the data itself (group count + assumption checks), not predetermined. Every test in Inferential Statistics now computes a live assumption check on the exact data you've selected and shows an explicit recommendation — both results still display (nothing is hidden), the recommendation is advisory.

### K.1 — Pre vs Post: recommendation + always-on Wilcoxon (3 min)

1. Go to **📈 Inferential Statistics → Pre vs Post**, pick either instrument pair.
2. Confirm a **"✅ Recommended: ..."** green callout appears above the results, naming either "Paired t-test" or "Wilcoxon signed-rank", with a one-line rationale citing the normality verdict of the *difference scores*.
3. Confirm the **"Show Wilcoxon signed-rank test"** checkbox is gone — the Wilcoxon result now always appears as a caption below the main metrics (no manual toggle needed).
4. ✅ Pass if: the callout appears with a sensible recommendation, and Wilcoxon results show without needing to check anything.

### K.2 — Between Groups: 2-group vs 3+-group, Levene's test (7 min)

1. Still in Inferential Statistics, go to **Between Groups**. Pick a **"Group by"** option that yields exactly 2 groups (e.g. **gender**, if only Male/Female are present).
2. Confirm the metrics row now shows **"t statistic" / "Independent t-test p-value" / "Cohen's d" / "Mann-Whitney U p"** — not F/ANOVA/Kruskal-Wallis. A caption below shows the ANOVA/Kruskal-Wallis numbers too (for reference — mathematically equivalent for 2 groups), and a Welch's t-test line appears if variances are unequal.
3. Open the new **"📐 Assumption checks (normality + Levene's)"** expander — confirm a per-group normality table and a Levene's test verdict appear.
4. Confirm the recommendation callout names one of: Independent t-test, Mann-Whitney U, or Welch's t-test — matching what the normality/Levene's verdicts in step 3 would suggest.
5. Switch **"Group by"** to something with 3+ groups (e.g. **cohort_id**). Confirm the metrics row switches back to the original **F statistic / ANOVA p-value / η² / Kruskal-Wallis p** layout, and the recommendation now names ANOVA or Kruskal-Wallis.
6. ✅ Pass if: the metric layout correctly switches with group count, the assumption-checks expander shows real numbers, and the recommendation is consistent with them.

### K.3 — Across Modules: real RM-ANOVA + sphericity (5 min)

1. Go to **Across Modules**, select **"MCQ content knowledge"**.
2. Below the existing Friedman metrics, confirm a second row appears: **"RM-ANOVA F" / "p (uncorrected)" / "p (Greenhouse-Geisser)" / "Generalized η²"**, plus a caption reporting Mauchly's test (W, p) and whether sphericity holds.
3. Confirm the recommendation callout names either "RM-ANOVA" or "Friedman test", with a rationale — and if sphericity is violated, the rationale should say to prefer the Greenhouse-Geisser corrected p-value.
4. Switch to **"Survey construct"** — confirm the same RM-ANOVA row and sphericity caption appear there too.
5. Narrow the module picker (from Task J) down to **exactly 2 modules** — confirm you get a clear message directing you to use "Pre vs Post" instead, not a cryptic error (2 time points is a paired comparison, not repeated-measures).
6. ✅ Pass if: RM-ANOVA metrics appear for both MCQ and survey paths, sphericity is reported, and narrowing to 2 modules degrades gracefully.

### K.4 — Inferential Statistics PDF report (3 min)

1. Go to **Report Generation → ii. Inferential Statistics**, generate the PDF.
2. Confirm it completes without error — every section (Pre vs Post, Between Groups, Across Modules) now includes a "Recommended: ..." line, Between Groups also shows the Levene's verdict, and the survey-constructs summary table has a new "Recommended" column.
3. ✅ Pass if: the PDF generates cleanly and the new recommendation text/column appear in each relevant section.

### Cleanup

None needed — read-only/diagnostic, no new data is written.

---

## L. Correlations tab — engagement/motivation vs. assessment performance (30 min)

A new **"🔗 Correlations"** tab now sits right after Inferential Statistics. It answers whether cognitive engagement (SCCCES) or motivation (SIMS) relates to assessment performance — both within students over time, and between students overall — following your confirmed 4-phase methodology.

### L.1 — Reliability & Redundancy (7 min)

1. Go to **🔗 Correlations → Reliability & Redundancy**.
2. Confirm a **per-sub-construct reliability table** appears for Cognitive Engagement (SCCCES) by default — 10 rows, one per sub-construct. Confirm the 1-item sub-constructs (Engagement with task, Experience of flow) show **"single_item_no_reliability"**, not a blank or a 0.0. Confirm the 2-item ones (Effort and persistence, Attention, Culture, Plausibility, Credibility, Comprehensibility) show **Spearman-Brown** values, and the 3-item ones (Coherency, Personal relevance) show **Cronbach's α**.
3. Switch the **"Survey"** dropdown to Interest & Motivation (SIMS) — confirm 4 rows (Intrinsic, Identified, External, Amotivation), each 3-4 items, using Cronbach's α or Spearman-Brown as appropriate.
4. Confirm an **inter-sub-construct correlation matrix** appears below — a 10×10 (SCCCES) or 4×4 (SIMS) table. Open the caption and confirm it references Heddy et al. (2018)'s own factor-analysis finding that the 4 "message appraisal" sub-constructs collapse into one factor.
5. ✅ Pass if: both tables render for both surveys, single-item constructs are clearly labeled "not estimable" rather than blank, and the citations are visible.

### L.2 — Composite Builder (5 min)

1. Switch to **Composite Builder**. Confirm 4 multiselect boxes appear — Engagement, Message appraisal, Personal relevance, Culture — each pre-populated with the theory-driven default grouping (e.g. Engagement defaults to task+effort+flow+attention).
2. Try removing one sub-construct from a composite and click **"💾 Save composite definitions"** — confirm a green success message appears.
3. Confirm the **RAI** explanation appears below, correctly citing **Ryan & Connell (1989)** — not Guay, Vallerand & Blanchard (2000) (the SIMS source paper, which doesn't itself present the RAI formula). Confirm a count like "X of Y (student, module) rows have all 4 SIMS sub-constructs present."
4. ✅ Pass if: the composite editor works, saving shows confirmation, and the RAI citation is correct.

### L.3 — Mixed-Effects Model (10 min)

1. Switch to **Mixed-Effects Model**. Select 1-2 predictors (e.g. Engagement, RAI).
2. Open the **"🧪 Exclude flagged participants"** expander (same as Inferential Statistics) if you want to try excluding outliers/missing-data participants first — optional.
3. Click **"▶ Run Mixed-Effects Model"**. Confirm it completes within a few seconds and:
   - A **standing yellow caveat** about asymptotic (not small-sample-corrected) standard errors appears — this should always show, every run.
   - A **model comparison table** appears (M0_null, M1_module, M2_within, M3_full, and possibly M3b_random_slope), each with AIC/BIC/Converged.
   - A green **"✅ Best model by BIC"** message names one of the blocks.
   - A **coefficients table** appears for the best model, showing `_within` and `_between` terms separately for each predictor.
   - Expanding **"📊 Likelihood ratio tests"** and **"📐 Variance Inflation Factors"** shows populated tables.
4. Open **"ℹ️ What do these numbers mean?"** and confirm it now includes, in order: a block-by-block explanation of the model comparison table (M0_null → M3b_random_slope); how "Best model by BIC" is selected, citing **Schwarz (1978)**; how "Best model by AIC" is selected, citing **Akaike (1974)**; a general guide to reading a coefficients table (including what the `_reml` suffix means); an explanation of how to interpret a random-slope model's coefficients (e.g. `M3b_random_slope_reml`), citing **Barr, Levy, Scheepers & Tily (2013)**; the existing LRT explanation; and the existing within/between-person and VIF explanations.
5. ✅ Pass if: the model runs without error, the small-sample caveat is always visible, within/between coefficients are clearly distinguished, and the expanded help text correctly explains every table on the page with its citation.

### L.4 — Repeated-Measures Correlations (8 min)

1. Switch to **Repeated-Measures Correlations**. Select the same 1-2 predictors.
2. Click **"▶ Run Repeated-Measures Correlations"**. Confirm a results table appears with columns for r, **Effect size**, N students, uncorrected p, FDR-corrected p, and a "Significant (FDR)" Yes/No column.
3. Confirm the **Effect size** column shows a Cohen (1988) label (Negligible / Small / Medium / Large) matching the magnitude of each row's r value.
4. If any predictor is FDR-significant, confirm a green **"✅ Strongest FDR-significant predictor"** callout appears.
5. Open **"ℹ️ What do these numbers mean?"** and confirm it references Bakdash & Marusich (2017) for rmcorr, Cohen (1988) for the effect-size labels (including the caveat that these are rough, domain-independent benchmarks), and Benjamini & Hochberg (1995) for FDR.
6. ✅ Pass if: results render with effect-size labels, both uncorrected and FDR-corrected p-values shown side by side (nothing is silently hidden).

### L.5 — Correlations Report (Report Generation tab) (8 min)

1. Switch to the **"📄 Report Generation"** tab (top-level tab, not the Correlations tab).
2. Click the **"vi. Correlations"** option in the Report section radio (between "v. Competency Progression" and "vii. Full Programme Report" — note Full Programme Report and Instruments & References have shifted to vii/viii to make room).
3. Confirm all 4 section checkboxes are pre-selected: Reliability & Redundancy, Composite Correlation Matrix, Mixed-Effects Model, Repeated-Measures Correlations.
4. Click **"📄 Generate Correlations PDF"**. Confirm it completes (may take up to ~30 seconds) with no error, and a **"⬇️ Download Correlations Report (PDF)"** button appears with a green "PDF ready" message.
5. Download and open the PDF. Confirm it includes: a reliability table for both SCCCES and SIMS; an inter-sub-construct correlation matrix for both surveys; a mixed-effects model block-comparison table plus coefficients tables for the BIC-best and AIC-best blocks (using the full default predictor set — all 4 composites + RAI + Amotivation, not just 1-2); a repeated-measures correlations table with effect-size labels.
6. ✅ Pass if: the PDF generates without error and contains real numbers (not blank/placeholder sections) for all 4 phases, using the full predictor set rather than requiring you to have run the live Correlations tab first.

### Cleanup

None needed — read-only/diagnostic. The saved composite definitions (L.2) persist only for your current session.

---

## If anything fails

Note which item failed and what you saw, then let Claude know — don't try to fix anything yourself. This is all on the disposable `test` environment; nothing here touches the real pilot data on `production`.

## Once everything passes

Let Claude know, and production deploy will follow the same sequence already used for every prior batch: fresh production backup → merge `test` → `main` → push → verify live.
