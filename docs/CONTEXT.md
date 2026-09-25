# Project context: facts established so far

## Task (from reference/PROBLEM_STATEMENT.md)
* 3 sources: S1 is a deduplicated reference; S2 and S3 are noisy. For each test S1, output matched S2/S3 ids (zero, one or many).
* Output: `matching_results.tsv` (scored) and `candidate_pairs.tsv` (the exact candidate set scored by the model; matches ⊆ candidates),
  tab-separated, one row per test S1, empty list allowed, no duplicates, only S2-/S3- ids.
* Metric: F0.5 computed **per S1 entity, then averaged over all S1** (singletons included). An empty prediction on a singleton scores 1.0; any prediction on a singleton scores 0.
* Rules: no external lookup, APIs, geocoding or external data. The final model must be MIT/Apache licensed and ≤8B parameters. Country is an open set, and **France appears only in test**.
* Final package: output/ (2 tsv), code/ (runnable, README, requirements), and the methodology doc (`reference/Documentation_template.md`).

## Data facts (measured)
| | train | test |
|---|---|---|
| S1 | 2,206,821 (US 1.32M, India 0.88M) | 1,732,544 (India 810k, US 663k, **France 259k**) |
| S2 / S3 | 5,034,616 / 5,285,603 | 4,887,273 / 5,082,316 |
* GT: 7,638,365 links. **Singletons 5.58%.** Matches per S1: 0:123k, 1:119k, 2:375k, 3:531k, 4:484k, 5:322k, 6:165k, 7:64k, 8+:24k (max 11).
  S2 per S1 max 5, S3 per S1 max 6. So the problem is **one-to-many**; S2 and S3 are not deduplicated.
* **Every S2/S3 id belongs to at most one S1** (0 of 7.6M ids shared), which gives the exclusivity rule. **0 cross-country links**, so blocking is within country.
* ~26% of pool records are unmatched distractors. 80.5% of S1 have both S2 and S3 matches.
* Noise: S2 names 28% ALL-CAPS; ~9% of S2 names and ~5% of S3 names in Indic scripts (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati);
  junk prefixes (`>>`, `***`, `#`, `@`) ~2%; domains/handles ~3% (`victorylaboratories.com`); empty address ~3.3% (test ~2.7%);
  `NULL`/`N/A` tokens in addresses ~2.5%; reordered address components; typos; legal-suffix churn; `t/a` / DBA names; opaque trade names
  (only the address matches); state names vs codes; native-script state names.
* S1 names: only ~70% unique strings (common names: "Summit Inc"). Commonness features matter for precision.
* TSV quoting: standard CSV quoting with doubled `""`. Read with `pd.read_csv(sep='\t', dtype=str, keep_default_na=False)`.
* EDA report claims (reference/DATA_ANALYSIS_REPORT.md): postal codes agree in 98% of true pairs when both are present; name-token ∪ address-token blocking reaches ~99.97% recall;
  24% of true pairs have reordered name tokens; pairs with name Jaccard < 0.2 and address Jaccard < 0.2 hold < 0.5% of true pairs.

## Artifacts
* `notebook/business_entity_resolution.ipynb`: a single Kaggle notebook (P100), EDA → blocking → models → stage 2 → rules → test submission.
  It has not been run yet. The lab code in `erlab/` is ported from it and smoke-tested locally end to end on a mini dataset (all 20 experiments OK).
* Local smoke numbers are meaningless (the mini pool is easy). Real numbers come from the remote runs.

## Remote machine
* 64-core CPU, 128 GB RAM, RTX 40-series GPU. Access is SSH + VS Code + Copilot agent. `git pull` works but there is no push or clipboard.
  A few files can be downloaded. Results come back via `results/PASTE_BACK.md`.
