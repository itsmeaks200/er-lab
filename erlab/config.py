"""Default configuration + deep-merge of experiment overrides.

Every experiment in experiments.py is just a dict of overrides on top of DEFAULTS. Cache keys are hashes of the
config sub-trees that influence each stage, so changing e.g. the model does not rebuild candidates.
"""
import copy
import os

DEFAULTS = {
    'paths': {
        'data_dir': os.environ.get('ER_DATA', './dataset'),       # folder containing train/ and test/ (searched recursively)
        'cache_dir': os.environ.get('ER_CACHE', './cache'),       # large intermediate files (never committed)
        'results_dir': os.environ.get('ER_RESULTS', './results'), # small reports (paste back / download)
    },
    'seed': 42,
    'world': {
        'n_s1_sample': 300_000,       # train S1 entities in the experiment world (fit/tune/hold by entity hash)
        'fold_pcts': [60, 20, 20],
        'pool': 'full',               # 'full' = every S2/S3 record (realistic) | 'mini' = matches of sampled S1 + distractors
        'mini_distractors': 200_000,
        'map_pairs': 300_000,         # fit-fold positive pairs used to learn synonym maps
        'map_min_count': 40,
        'map_min_prob': 0.5,
    },
    'vec': {'hash_bits': 22, 'char_bits': 20},
    'retr': {
        'df_cap_frac': 0.0008, 'df_cap_min': 2000, 'char_df_cap_frac': 0.003,
        'skel_weight': 0.5, 'num_weight': 1.0, 'bigram_weight': 1.0, 'alpha_name': 0.6,
        'bm25': False, 'chunk': 2000,
    },
    'block': {
        # A exact squash name | B skeleton prefix | C tfidf name | D tfidf addr | E house#+street | F tfidf name+addr
        # G hash-encoder dense | H postal+name | I char-3gram tfidf | S sentence-transformer dense
        'passes': ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'G'],
        'topk_max': 40,
        'k': {},                      # explicit K per ranked pass; missing → chosen from recall@K curve
        'k_keep': 0.995,
        'k_grid': [1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40],
        'max_block': 300,
        'greedy_min_gain': 0.0005,
        'select': 'greedy',           # 'greedy' (fit-fold recall) | 'all'
    },
    'encoder': {'enabled': True, 'bucket_bits': 18, 'dim': 64, 'epochs': 10, 'batch': 2048, 'tau': 0.05, 'lr': 0.01},
    'st': {                           # pretrained sentence-transformer (MIT/Apache licences only)
        'enabled': False, 'model': 'intfloat/multilingual-e5-small', 'max_len': 64, 'batch': 1024,
        'finetune': False, 'ft_epochs': 1, 'ft_batch': 256, 'ft_lr': 3e-5, 'prefix': 'query: ',
    },
    'feats': {'chunk': 1_000_000, 'prune': True, 'prune_max_loss': 0.001, 'version': 3},
    'model': {
        'kind': 'lgb',                # lgb | lgb_rank | xgb | cat | lr | mlp
        'params': {},                 # overrides of the per-kind defaults (models.py)
        'rounds': 5000, 'early_stop': 200, 'max_train_rows': 40_000_000,
        'feature_drop': [],           # prefixes of feature groups to drop (ablations)
    },
    'decide': {'mode': 'staged', 'calibrate': 'isotonic'},   # staged thresholds | expf (expected-F0.5 set selection)
    'stage2': {'enabled': False, 'keep_feats': 15},
    'crossenc': {
        'enabled': False, 'model': 'intfloat/multilingual-e5-small', 'max_len': 96, 'epochs': 1,
        'batch': 256, 'lr': 3e-5, 'neg_per_pos': 3, 'max_train_pairs': 3_000_000, 'band': [0.01, 0.995], 'top_n': 12,
    },
}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def make_cfg(overrides=None, cli=None):
    cfg = deep_merge(DEFAULTS, overrides or {})
    for kv in (cli or []):                      # --set block.topk_max=30 world.pool=mini
        key, val = kv.split('=', 1)
        node = cfg
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            import ast
            node[parts[-1]] = ast.literal_eval(val)
        except Exception:
            node[parts[-1]] = val
    return cfg
