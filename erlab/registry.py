"""Experiment registry. Add new experiments here (Claude writes them, you git pull and run).

Waves (run in order; each wave's PASTE_BACK.md is sent back to the chat):
  0  ENV, SMOKE                         — environment + end-to-end on a mini dataset (minutes)
  1  B01..B05, P01, P02                 — blocking architecture on the FULL train pool; P = stage-1 learned pruning
  2  M01..M08, M09, M10, D01            — model families, blend, ablations, decision rules
  3  M11, M12, N01, N02                 — stage-2 cross-source reranker, HPO, sentence-transformers, cross-encoder
  4  S01 / S02                          — final training + test inference + validated submission files
"""

SMOKE_CFG = {'smoke': True, 'world': {'n_s1_sample': 4000, 'pool': 'full', 'map_min_count': 5},
             'retr': {'df_cap_min': 50, 'chunk': 500},
             'encoder': {'epochs': 2, 'batch': 256},
             'model': {'rounds': 200, 'early_stop': 30},
             'block': {'topk_max': 20, 'k_grid': [1, 3, 5, 10, 20]},
             'st': {'batch': 256, 'ft_batch': 64},
             'crossenc': {'max_train_pairs': 20_000, 'batch': 64}}

EXPERIMENTS = {
    # ------------------------------------------------------------------ wave 0
    'ENV': dict(wave=0, kind='env', desc='environment + data discovery'),
    'SMOKE': dict(wave=0, kind='submit', desc='end-to-end smoke test on the mini dataset (run with --data ./mini_data)',
                  cfg=SMOKE_CFG, args=dict(models=[dict(kind='lgb', params={})])),
    # ------------------------------------------------------------------ wave 1: blocking
    'B01': dict(wave=1, kind='blocking', desc='baseline multi-pass blocking A,B,C,D,E,F,H,G (TF-IDF + keys + hash encoder)'),
    'B02': dict(wave=1, kind='blocking', desc='BM25 weighting for C/D/F + char-3gram pass I',
                cfg={'retr': {'bm25': True}, 'block': {'passes': ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'G', 'I']}}),
    'B03': dict(wave=1, kind='blocking', desc='stricter df cap (0.0003) — cheaper retrieval, recall cost?',
                cfg={'retr': {'df_cap_frac': 0.0003}}),
    'B04': dict(wave=1, kind='blocking', desc='+ pretrained multilingual-e5-small dense pass S (MIT)',
                cfg={'st': {'enabled': True}}),
    'B05': dict(wave=1, kind='blocking', desc='+ contrastively fine-tuned e5-small dense pass S (SC-Block style)',
                cfg={'st': {'enabled': True, 'finetune': True}}),
    # stage-1 learned blocking: shrink the candidate set per S1 (the final ranking rewards fewer candidates)
    'P01': dict(wave=1, kind='prune', desc='stage-1 learned blocking (cheap signals) on B04 retrieval: matcher F0.5 vs candidates/S1',
                cfg={'st': {'enabled': True}, 'prune1': {'enabled': True}}, args=dict(baseline=True)),
    'P02': dict(wave=1, kind='prune', desc='P01 on ALL train S1 + within-pool-record ranks + top-m S1 per pool record (unique assignment)',
                cfg={'world': {'n_s1_sample': 10 ** 9}, 'st': {'enabled': True}, 'prune1': {'enabled': True, 'pool_ctx': True}}),
    # ------------------------------------------------------------------ wave 2: models on B01 blocking
    'M01': dict(wave=2, kind='model', desc='LightGBM baseline (127 leaves)', args=dict(show_imp=True)),
    'M02': dict(wave=2, kind='model', desc='LightGBM shallow/regularised (synthetic-noise law: 31 leaves, ff 0.5)',
                cfg={'model': {'params': dict(num_leaves=31, min_data_in_leaf=1000, feature_fraction=0.5, lambda_l2=10.0)}}),
    'M03': dict(wave=2, kind='model', desc='LightGBM deep (511 leaves)',
                cfg={'model': {'params': dict(num_leaves=511, min_data_in_leaf=100, learning_rate=0.03)}}),
    'M04': dict(wave=2, kind='model', desc='LightGBM lambdarank (listwise within S1)', cfg={'model': {'kind': 'lgb_rank'}}),
    'M05': dict(wave=2, kind='model', desc='XGBoost (GPU)', cfg={'model': {'kind': 'xgb'}}),
    'M06': dict(wave=2, kind='model', desc='CatBoost (GPU)', cfg={'model': {'kind': 'cat'}}),
    'M07': dict(wave=2, kind='model', desc='Logistic regression (linear diversity)', cfg={'model': {'kind': 'lr'}}),
    'M08': dict(wave=2, kind='model', desc='MLP on quantile-normalised features (GPU)', cfg={'model': {'kind': 'mlp'}}),
    'M09': dict(wave=2, kind='blend', desc='hill-climbing blend of M01..M08 (prob + rank)',
                args=dict(members=['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08'])),
    'M10': dict(wave=2, kind='ablation', desc='feature-group ablation (quick LightGBM on 60k fit S1)'),
    'D01': dict(wave=2, kind='decision', desc='decision rules deep dive (global / staged / expected-F0.5) on M01', args=dict(pred='M01')),
    # ------------------------------------------------------------------ wave 3: advanced
    'M11': dict(wave=3, kind='stage2', desc='stage-2 cross-source confirmation reranker (OOF stacking)'),
    'M12': dict(wave=3, kind='hpo', desc='Optuna TPE on 5 LightGBM params (objective = tune F0.5)', args=dict(trials=40)),
    'N01': dict(wave=3, kind='model', desc='LightGBM + pretrained e5-small embedding features/pass',
                cfg={'st': {'enabled': True}}),
    'N02': dict(wave=3, kind='stage2', desc='stage 2 + Ditto-style cross-encoder on the uncertain band',
                cfg={'crossenc': {'enabled': True}}, args=dict(crossenc=True)),
    'N03': dict(wave=3, kind='stage2', desc='stage 2 + cross-encoder on ALL stage-1 survivors (affordable: ~5 cands/S1)',
                cfg={'crossenc': {'enabled': True, 'band': [0.0, 1.0], 'top_n': 50}}, args=dict(crossenc=True)),
    'N04': dict(wave=3, kind='stage2', desc='stage 2 + LARGE cross-encoder multilingual-e5-large (560M, MIT) on the uncertain band',
                cfg={'crossenc': {'enabled': True, 'model': 'intfloat/multilingual-e5-large', 'max_len': 64, 'batch': 32,
                                  'lr': 1.5e-5, 'max_train_pairs': 400_000, 'train_minutes': 75, 'pred_batch': 256,
                                  'band': [0.02, 0.98], 'top_n': 8, 'max_pred_pairs': 1_500_000}},
                args=dict(crossenc=True)),
    # ------------------------------------------------------------------ wave 4: submission
    'S01': dict(wave=4, kind='submit', desc='final: LightGBM baseline + best rule → test submission',
                args=dict(models=[dict(kind='lgb', params={})])),
    'S03': dict(wave=4, kind='submit', desc='final: stage-1 learned blocking (small candidate_pairs.tsv) + LightGBM matcher → test submission',
                cfg={'st': {'enabled': True}, 'prune1': {'enabled': True, 'pool_ctx': True}, 'world': {'n_s1_sample': 10 ** 9}},
                args=dict(models=[dict(kind='lgb', params={})])),
    'S02': dict(wave=4, kind='submit', desc='final: stage-2 cross-source reranker → test submission',
                args=dict(models=[dict(kind='lgb', params={})], stage2=True)),
}


# ------------------------------------------------------------------ approach pipelines (each: holdout → refit → test)
# Every approach writes results/submissions/<ID>/{holdout.json, matching_results.tsv, candidate_pairs.tsv, submission_meta.json}
# and appends one row to results/submissions/SUMMARY.csv. All share: full train S1 set, stage 0 (keys + TF-IDF + hash
# encoder + e5-small), stage-1 learned blocking with per-pool-record context, the leak-free 'enc' fold.
FULL = {'world': {'n_s1_sample': 10 ** 9}, 'st': {'enabled': True}, 'prune1': {'enabled': True, 'pool_ctx': True}}
CE_SMALL = {'crossenc': {'enabled': True, 'model': 'intfloat/multilingual-e5-small', 'max_len': 96, 'batch': 256, 'lr': 3e-5,
                         'max_train_pairs': 2_000_000, 'band': [0.005, 0.995], 'top_n': 12, 'max_pred_pairs': 4_000_000,
                         'train_minutes': 60, 'pred_batch': 1024}}
CE_LARGE = {'crossenc': {'enabled': True, 'model': 'intfloat/multilingual-e5-large', 'max_len': 64, 'batch': 32, 'lr': 1.5e-5,
                         'max_train_pairs': 400_000, 'band': [0.02, 0.98], 'top_n': 8, 'max_pred_pairs': 1_500_000,
                         'train_minutes': 75, 'pred_batch': 256}}


def _deep(*ds):
    from .config import deep_merge
    out = {}
    for d in ds:
        out = deep_merge(out, d)
    return out


EXPERIMENTS.update({
    'SUB1': dict(wave=5, kind='submit', desc='A1: LightGBM matcher on stage-1 candidates',
                 cfg=FULL, args=dict(models=[dict(kind='lgb', params={})])),
    'SUB2': dict(wave=5, kind='submit', desc='A2: GBDT trio LightGBM + XGBoost + CatBoost (averaged probabilities)',
                 cfg=FULL, args=dict(models=[dict(kind='lgb', params={}), dict(kind='xgb', params={}), dict(kind='cat', params={})])),
    'SUB3': dict(wave=5, kind='submit', desc='A3: LightGBM + stage-2 cross-source reranker (OOF stacking)',
                 cfg=FULL, args=dict(models=[dict(kind='lgb', params={})], stage2=True)),
    'SUB4': dict(wave=5, kind='submit', desc='A4: A3 + multilingual-e5-small cross-encoder (trained on the enc fold) stacked',
                 cfg=_deep(FULL, CE_SMALL), args=dict(models=[dict(kind='lgb', params={})], stage2=True, crossenc=True)),
    'SUB5': dict(wave=5, kind='submit', desc='A5: A3 + LARGE multilingual-e5-large (560M, MIT) cross-encoder stacked',
                 cfg=_deep(FULL, CE_LARGE), args=dict(models=[dict(kind='lgb', params={})], stage2=True, crossenc=True)),
    'SUB6': dict(wave=5, kind='submit', desc='A6: LightGBM with Optuna-tuned params (from M12)',
                 cfg=FULL, args=dict(models=[dict(kind='lgb', params={}, params_file='M12_best_params.json')])),
    'SUMMARY': dict(wave=5, kind='summary', desc='approach leaderboard (holdout F0.5) from results/submissions/SUMMARY.csv'),
})
EXPERIMENTS['M12']['args'] = dict(trials=30)

# `python run.py overnight` (10-12 h): approaches in priority order so that the most useful results and valid
# submissions exist even if the night is cut short. One process, one cached context: each approach only pays for its
# own models; the test candidates/features are built once (first approach) and reused.
OVERNIGHT = dict(
    ids=['SUB1',        # A1 LightGBM                                   → first holdout score + submission
         'SUB3',        # A3 + stage-2 cross-source reranker
         'SUB4',        # A4 + e5-small cross-encoder
         'SUB2',        # A2 GBDT trio
         'M12', 'SUB6',  # Optuna (30 trials) → A6 tuned LightGBM
         'SUMMARY',
         'P02',         # analysis: matcher F0.5 vs candidates/S1 frontier
         'SUB5',        # A5 + LARGE e5-large cross-encoder (slowest, last)
         'SUMMARY'],
    sets=['world.n_s1_sample=1000000000', 'st.enabled=True', 'prune1.enabled=True', 'prune1.pool_ctx=True'],   # = FULL
)
