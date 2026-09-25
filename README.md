# ER-lab: Business Entity Resolution experiment lab

This is the experiment framework for the **ML Challenge 2026 Business Entity Resolution** task: for each Source-1 business, find its matching Source-2/3 records, scored by macro F0.5 per S1.
It is built for a remote GPU box you can only `git pull` to. Experiments are written here, run there, and their compact reports are pasted back.

```
er_lab/
├── run.py                   ← the only entry point
├── erlab/                   ← package (all stages cached)
│   ├── config.py            defaults + overrides (--set a.b=c)
│   ├── data.py              discovery, TSV loading, pool, ground truth, folds, mini-dataset maker
│   ├── text.py              normalisation engine (Unicode transliteration, names, addresses, learned maps)
│   ├── vec.py               normalised frames, commonness features, hashed sparse matrices, IDF
│   ├── blocking.py          passes A,B,E,H (keys) C,D,F,I (TF-IDF/BM25 top-K) G,S (dense GPU) + union
│   ├── encoders.py          hash-encoder (trained from scratch), sentence-transformers (+contrastive fine-tune)
│   ├── features.py          ~140 pair features, within-S1 context, safe pruning, stage-2 cross-source features
│   ├── models.py            LightGBM (binary/lambdarank), XGBoost, CatBoost, LogReg, MLP, hill-climb blending
│   ├── crossenc.py          Ditto-style cross-encoder reranker (uncertain band only)
│   ├── evaluate.py          official metric, rule search (staged thresholds), expected-F0.5 set selection
│   ├── pipeline.py          cached worlds / passes / features
│   ├── experiments.py       runners (env, blocking, model, ablation, blend, stage2, hpo, decision, submit)
│   ├── registry.py          ← THE EXPERIMENT LIST (waves 0-4)
│   ├── submit.py            final training → test inference → files → own + official validation
│   └── report.py            results/PASTE_BACK.md, LEADERBOARD.csv, runs/<EXP>.md, summary.json
├── docs/                    RESEARCH_SYNTHESIS.md, AGENT_INSTRUCTIONS.md, HANDOFF_PROMPT.md, CONTEXT.md
├── reference/               problem statement, official validator, EDA report, documentation template
└── notebook/                self-contained Kaggle notebook (P100) + its source parts + builder
```

## Setup (remote)
```bash
git pull                                   # branch er-lab
cd er_lab
pip install -r requirements.txt            # torch: install the CUDA build first if missing
export ER_DATA=/path/to/dataset            # folder containing train/ and test/ (searched recursively)
```
On Windows PowerShell use `$env:ER_DATA="D:\path\to\dataset"`. Caches go to `./cache` (set `ER_CACHE` to a fast disk with 100+ GB free).

## The loop
```bash
python run.py list                                     # all experiments
python run.py exp ENV                                  # 1 min: versions, GPU, RAM, data found?
python run.py make-mini --out ./mini_data              # once: small faithful dataset
python run.py wave 0 1 2 3 --smoke --data ./mini_data --keep-going   # 20-30 min: every code path on the mini data
python run.py wave 1                                   # real run on the full data (blocking)
python run.py wave 2                                   # models / blend / ablation / decision rules
python run.py wave 3                                   # stage-2, HPO, embeddings, cross-encoder
python run.py exp S01                                  # final submission → results/submission/*.tsv
```
After every command: **paste `results/PASTE_BACK.md` into the chat.** It stays short. Full reports are in `results/runs/`, and `results/LEADERBOARD.csv` is the running history. Both are small, so download them if asked.

Overrides without code changes: `python run.py exp M01 --set world.n_s1_sample=500000 model.rounds=8000`.

## Design principles
* The validation mirrors the competition. S1 entities are split fit/tune/hold by hash. Blocking never sees labels. Everything that can overfit (early stopping, thresholds) uses tune, and hold is only reported.
* All blocking runs against the full pool (realistic distractor density). Every run reports the oracle F0.5, the ceiling given the candidates.
* Decision rules are optimised for the actual metric: staged thresholds or the expected-F0.5 set selector, including "no match".
* Caching means a new model experiment reuses the world, passes and features (minutes, not hours).
* Rules: only competition data is used. Pretrained models are MIT/Apache and ≤8B parameters (`intfloat/multilingual-e5-small`, MIT). No lookups, geocoding or external data.
