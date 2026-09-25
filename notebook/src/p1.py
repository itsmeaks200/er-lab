# %% [markdown]
# # Business Entity Resolution — end-to-end competition notebook
#
# **Goal.** For every Source‑1 (reference) entity, return the Source‑2/Source‑3 records that refer to the same business,
# or nothing. The metric is **macro F0.5 at the S1-entity level**, so precision counts more than recall, and
# a correct "no match" earns full credit.
#
# The notebook answers four questions in order. Every analysis cell ends in a concrete pipeline decision.
#
# 1. **What does the data say about the problem?** Integrity, source noise, ground-truth structure, singletons (§1–§5).
# 2. **Which representations and candidate generator give the most recall for a manageable volume?** (§6–§7, §10–§11)
# 3. **Which features separate true matches from *hard* negatives?** (§8–§9, §14–§15)
# 4. **Which decision rule maximises macro F0.5?** (§12–§13, §16–§18)
#
# Then the frozen pipeline runs on the test set and writes `matching_results.tsv` and `candidate_pairs.tsv` (§19–§23).
#
# ```text
# RAW TSV → normalisation (translit, legal forms, learned token maps) → multi-pass blocking (keys + TF-IDF + GPU encoder)
#        → pair features (fuzzy, TF-IDF, numeric, rarity, group context) → GBDT scoring → tuned per-source / anchor
#        thresholds → exclusivity (one S1 per S2/S3 record) → matches ⊆ candidates → validated submission
# ```
#
# **Rules respected:** only competition files are used. There is no internet, no external lookup and no geocoding.
# Countries are never hard-coded (France is only in test). Matching is one-to-many, and singletons are never forced to match.
#
# Every tunable knob is in `CFG` (next cell). The defaults are sized for a Kaggle P100 session (≈29 GB RAM, 4 CPUs, 16 GB GPU).
#
# ### How to run on Kaggle
# 1. **Add Data** → attach the competition dataset (any folder name; the notebook finds the 7 TSVs under `/kaggle/input`).
# 2. **Settings** → Accelerator **GPU P100**. Internet can stay **off**, because nothing is downloaded.
# 3. **Run All** (or *Save Version → Save & Run All*). Expect a few hours; per-step timings are printed and summarised in §22.
# 4. Outputs: `/kaggle/working/matching_results.tsv` and `/kaggle/working/candidate_pairs.tsv` (hard-linked into `/kaggle/working/output/`), plus
#    `artifacts/` (models, encoder, experiment log, findings) and `Documentation_template_filled.md` (a pre-filled methodology write-up).
#
# If memory or time is tight, lower `N_S1_SAMPLE` (the experiment sample), `TOPK_MAX` or `FEAT_CHUNK`, or set `RUN_CATBOOST` / `RUN_STAGE2` to False.
#
# ### Where the knowledge-base / EDA-report insights are used
# | insight | where |
# |---|---|
# | 0 cross-country links → partition by country | all blocking passes (§10), no country feature |
# | each S2/S3 record belongs to at most one S1 | exclusivity option in the decision rule (§17) |
# | inverted-index stop-word cap on `llc`, `road`, `delhi`… | document-frequency cap in TF-IDF retrieval (§7b/§10) |
# | name ∪ address token blocking reaches ~100% recall | passes C/D/F + greedy union (§10) |
# | postal codes agree in ~98% of true pairs | postal match/conflict features, pass H (§7, §8) |
# | "safe pruning zone" (name and address both dissimilar) | validated pruning (§12) |
# | 24% of true pairs have reordered tokens | token-sort/set and set-equality features (§8) |
# | within-S1 relative scores | rank / gap / z-score context features (§8) |
# | 80% of S1 have both S2 and S3 matches (cross-source confirmation) | stage-2 reranker (§15b) |
# | entity-grouped validation, never pair-level | fit/tune/hold by S1 entity (§12) |
# | official validator | embedded and executed (§21) |
# | licence: MIT/Apache ≤ 8B parameters | only our own small encoder is trained, with no pretrained weights (§10) |

# %%
# =====================================================================================
# 0. SETUP — imports, configuration, utilities
# =====================================================================================
import os, sys, gc, re, csv, json, math, time, random, pickle, unicodedata, warnings, subprocess
from collections import Counter
import multiprocessing as mp

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_colwidth', 80)
plt.rcParams['figure.dpi'] = 100

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

try:
    display
except NameError:          # running outside Jupyter
    display = print


def _opt_import(name):
    try:
        return __import__(name)
    except Exception as e:  # noqa
        print(f'  [optional] {name} not available: {type(e).__name__}: {e}')
        return None


psutil = _opt_import('psutil')
lgb = _opt_import('lightgbm')
xgb = _opt_import('xgboost')
catboost = _opt_import('catboost')
torch = _opt_import('torch')

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein as RF_Lev, JaroWinkler as RF_JW
    try:
        from rapidfuzz.process import cpdist          # vectorised pairwise scorer (rapidfuzz >= 3.6)
    except ImportError:
        cpdist = None
    HAS_RF = True
except ImportError:
    HAS_RF, cpdist = False, None

try:
    import numba
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

try:
    import pyarrow  # noqa
    STR = 'string[pyarrow]'   # ~3x less RAM than python objects for 25M strings
except ImportError:
    STR = object

USE_GPU = bool(torch is not None and torch.cuda.is_available())
if torch is not None:
    torch.manual_seed(SEED)
    if USE_GPU:
        torch.cuda.manual_seed_all(SEED)
if not HAS_RF:
    class _NoRF:                      # fuzzy scorers become NaN features (model still trains on the rest)
        def __getattr__(self, k):
            return None
    fuzz = RF_Lev = RF_JW = _NoRF()
N_JOBS = max(1, min(os.cpu_count() or 1, 8))

CFG = dict(
    # ---------- IO ----------
    LOCAL_ROOTS=['./dataset', '../dataset', '.'],            # used only when /kaggle/input does not exist
    WORK_DIR='/kaggle/working' if os.path.isdir('/kaggle/working') else './working',
    # ---------- experiment scale ----------
    N_S1_SAMPLE=150_000,          # train S1 entities used for blocking/model experiments (entity-level sample)
    FOLD_PCTS=(60, 20, 20),       # fit / tune / hold (by S1 entity hash)
    EDA_SAMPLE=300_000,
    POS_SAMPLE=150_000,           # positive pairs for representation experiments
    QUICK_S1=60_000,              # fit S1 used for ablation models (E2/E3/E7/LOCO)
    # ---------- hashing / retrieval ----------
    HASH_BITS=22, CHAR_BITS=20,
    DF_CAP_FRAC=0.0008, DF_CAP_MIN=2000,   # tokens more frequent than this are ignored by the retrieval passes
    RARE_DF=50,
    SKEL_WEIGHT=0.5, NUM_WEIGHT=1.0, BIGRAM_WEIGHT=1.0, ALPHA_NAME=0.6,
    TOPK_MAX=30, K_GRID=[1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30], K_RECALL_KEEP=0.995,
    RETR_CHUNK=1000, MAX_BLOCK_POOL=300, GREEDY_MIN_GAIN=0.0005,
    # ---------- features / models ----------
    FEAT_CHUNK=1_000_000,
    LGB_ROUNDS=2500, LGB_LR=0.06, EARLY_STOP=100, QUICK_ROUNDS=600,
    MAX_TRAIN_ROWS=12_000_000,
    PRUNE_MAX_RECALL_LOSS=0.001,  # 'safe pruning zone' is applied only if it loses <= 0.1% of true pairs
    RUN_STAGE2=True, STAGE2_MIN_GAIN=0.0005, STAGE2_KEEP_FEATS=12,
    RUN_XGB=True, RUN_CATBOOST=True, RUN_LR=True,
    # ---------- GPU encoder (learned char-n-gram embeddings; trained ONLY on competition train data) ----------
    RUN_ENCODER=True, ENC_BUCKET_BITS=18, ENC_DIM=64, ENC_EPOCHS=10, ENC_BATCH=1024, ENC_TAU=0.05, ENC_LR=0.01,
    DENSE_CHUNK=256, EMB_MIN_GAIN=0.001,
)
os.makedirs(CFG['WORK_DIR'], exist_ok=True)
ART_DIR = os.path.join(CFG['WORK_DIR'], 'artifacts')
os.makedirs(ART_DIR, exist_ok=True)

T0 = time.time()
TIMINGS, FINDINGS, EXPERIMENTS = [], [], []


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9 if psutil else float('nan')


def log(*msg):
    print(f'[{time.time() - T0:7.0f}s | RSS {rss_gb():5.1f} GB]', *msg, flush=True)


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        log(f'>> {self.name}')
        return self

    def __exit__(self, *a):
        dt = time.time() - self.t
        TIMINGS.append(dict(step=self.name, seconds=round(dt, 1), rss_gb=round(rss_gb(), 2)))
        log(f'<< {self.name}: {dt:.1f}s')


def finding(title, evidence, why, action):
    """Every important analytical result is logged as Finding → Evidence → Why → Action."""
    FINDINGS.append(dict(finding=title, evidence=evidence, why=why, action=action))
    print(f'\n■ FINDING : {title}\n  Evidence: {evidence}\n  Why     : {why}\n  Action  : {action}')


def add_experiment(exp, change, m, decision):
    EXPERIMENTS.append(dict(Experiment=exp, Change=change, F05=round(m['F05'], 4),
                            Precision=round(m['P_micro'], 4), Recall=round(m['R_micro'], 4), Decision=decision))


# ---------------- light-weight parallel map (fork on Linux/Kaggle; serial elsewhere) ----------------
def _apply_chunk(args):
    func, items = args
    return [func(x) for x in items]


def run_parallel(fn, args_list, n_jobs=N_JOBS):
    if n_jobs <= 1 or len(args_list) < 2:
        return [fn(a) for a in args_list]
    try:
        ctx = mp.get_context('fork')
    except ValueError:
        return [fn(a) for a in args_list]
    with ctx.Pool(min(n_jobs, len(args_list))) as pool:
        return pool.map(fn, args_list, chunksize=1)


def pmap(func, items, chunk=100_000):
    items = list(items)
    if len(items) < 2 * chunk:
        return [func(x) for x in items]
    parts = run_parallel(_apply_chunk, [(func, items[i:i + chunk]) for i in range(0, len(items), chunk)])
    return [y for part in parts for y in part]


def umap(ser, func, as_str=True):
    """Apply a python function to the UNIQUE values of a Series (parallel), broadcast back."""
    codes, uniq = pd.factorize(ser, sort=False)
    res = pmap(func, uniq.tolist())
    arr = np.empty(len(res), dtype=object)
    arr[:] = res
    out = arr[codes]
    return pd.Series(out, index=ser.index, dtype=STR if as_str else None)


print(f'python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__} | cpus={os.cpu_count()} '
      f'| GPU={USE_GPU} | rapidfuzz={HAS_RF} (cpdist={cpdist is not None}) | numba={HAS_NUMBA} '
      f'| lightgbm={lgb is not None} | xgboost={xgb is not None} | catboost={catboost is not None} | str dtype={STR}')
if USE_GPU:
    print('GPU:', torch.cuda.get_device_name(0))
if not HAS_RF:
    print('WARNING: rapidfuzz missing -> fuzzy features fall back to slow difflib / NaN.')

# %% [markdown]
# ## 1. Dataset discovery
# The notebook searches `/kaggle/input/` recursively for the seven required TSV files, so the dataset folder name does not matter.
# It also looks for a submission validator script. Files are read as **tab-separated**, with `keep_default_na=False` so literal strings like `"null"`/`"NA"` stay visible for the integrity checks.

# %%
REQUIRED = {
    'train_s1': 'train_source1.tsv', 'train_s2': 'train_source2.tsv', 'train_s3': 'train_source3.tsv',
    'train_gt': 'train_ground_truth.tsv',
    'test_s1': 'test_source1.tsv', 'test_s2': 'test_source2.tsv', 'test_s3': 'test_source3.tsv',
}
SOURCE_COLS = ['entity_id', 'business_name', 'business_address', 'country']
GT_COLS = ['source1_entity_id', 'matched_entity_ids']


def discover_files():
    if os.path.isdir('/kaggle/input'):
        roots = ['/kaggle/input']
        print('/kaggle/input contains:', sorted(os.listdir('/kaggle/input')))
    else:
        roots = [r for r in CFG['LOCAL_ROOTS'] if os.path.isdir(r)][:1]
        print('No /kaggle/input — falling back to local roots', roots)
    wanted = {v.lower(): k for k, v in REQUIRED.items()}
    found, validators, tsvs = {}, [], []
    for root in roots:
        for dp, dns, fns in os.walk(root):
            dns[:] = sorted(d for d in dns if not d.startswith('.') and d not in ('working', '__pycache__'))
            for fn in sorted(fns):
                p, low = os.path.join(dp, fn), fn.lower()
                if low.endswith('.tsv'):
                    tsvs.append(p)
                if low in wanted:
                    found.setdefault(wanted[low], []).append(p)
                if low.endswith('.py') and ('valid' in low or 'check' in low or 'eval' in low):
                    validators.append(p)
    print(f'\nTSV files found ({len(tsvs)}):')
    for p in tsvs:
        print(f'   {p}  ({os.path.getsize(p) / 1e6:,.1f} MB)')
    missing = [REQUIRED[k] for k in REQUIRED if k not in found]
    if missing:
        raise FileNotFoundError(f'Missing required competition files: {missing}. Searched roots {roots}. '
                                f'Attach the competition dataset to the notebook.')
    for k, v in found.items():
        if len(v) > 1:
            print(f'   WARNING: {REQUIRED[k]} found {len(v)} times, using {v[0]}')
    return {k: v[0] for k, v in found.items()}, validators


def load_tsv(path, usecols=None):
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, na_filter=False,
                     usecols=usecols, encoding='utf-8')
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(STR)
    return df


def describe_frame(name, df):
    print(f'\n--- {name}: shape={df.shape}  memory={df.memory_usage(deep=True).sum() / 1e6:,.0f} MB')
    print('    dtypes:', {c: str(t) for c, t in df.dtypes.items()})
    display(df.head(3))


with Timer('discover + load all files'):
    PATHS, VALIDATORS = discover_files()
    print('\nValidator-like scripts:', VALIDATORS or 'none found')
    RAW = {k: load_tsv(p) for k, p in PATHS.items()}

for k in REQUIRED:
    need = GT_COLS if k == 'train_gt' else SOURCE_COLS
    miss = [c for c in need if c not in RAW[k].columns]
    if miss:
        raise ValueError(f'{PATHS[k]} is missing columns {miss}; found {list(RAW[k].columns)}')
    describe_frame(k, RAW[k])
print('\nAll 7 required files present with expected columns.')

# %% [markdown]
# ## 2. Data integrity
# These checks look for duplicate IDs, duplicate records, empty or whitespace-only fields, placeholders (`null`, `N/A`, …), malformed IDs and wrong
# source prefixes. They also flag countries unseen in training and every ground-truth inconsistency: duplicates, invalid references, and IDs mapped to several S1 entities.

# %%
PLACEHOLDERS = {'null', 'none', 'nan', 'n/a', 'na', 'nil', 'unknown', '-', '--', '0', '.', 'not available', 'tbd'}
TRAIN_COUNTRIES = set(RAW['train_s1'].country.str.strip().unique().tolist())


def integrity_report(name, df, prefix):
    r = dict(file=name, rows=len(df))
    ids = df.entity_id
    r['dup_ids'] = int(ids.duplicated().sum())
    r['dup_full_rows'] = int(df.duplicated().sum())
    r['dup_name_addr_country'] = int(df.duplicated(['business_name', 'business_address', 'country']).sum())
    for col, short in [('business_name', 'name'), ('business_address', 'addr'), ('country', 'ctry')]:
        s = df[col]
        st = s.str.strip()
        r[f'{short}_empty'] = int((s == '').sum())
        r[f'{short}_ws_only'] = int(((st == '') & (s != '')).sum())
        r[f'{short}_placeholder'] = int(st.str.lower().isin(PLACEHOLDERS).sum())
    r['addr_contains_null_token'] = int(df.business_address.str.lower().str.contains(r'\b(?:null|none|nan|n/a)\b').sum())
    r['malformed_ids'] = int((~ids.str.match(r'^S[123]-\d+$')).sum())
    r['wrong_prefix'] = int((~ids.str.startswith(prefix + '-')).sum())
    cs = df.country.str.strip().value_counts()
    r['countries'] = {str(k): int(v) for k, v in cs.items()}
    r['unseen_countries'] = sorted(set(cs.index.tolist()) - TRAIN_COUNTRIES)
    return r


def gt_integrity(gt, s1_ids, s2_ids, s3_ids):
    r = dict(rows=len(gt))
    r['dup_s1_rows'] = int(gt.source1_entity_id.duplicated().sum())
    r['gt_s1_not_in_source1'] = int((~gt.source1_entity_id.isin(s1_ids)).sum())
    r['source1_missing_from_gt'] = int((~s1_ids.isin(gt.source1_entity_id)).sum())
    ex = gt.matched_entity_ids.astype(object).str.split(',').explode()
    raw_nonempty_rows = gt.matched_entity_ids.str.strip() != ''
    r['empty_rows(singletons)'] = int((~raw_nonempty_rows).sum())
    ex = ex[ex.notna()]
    tok = ex.str.strip()
    nonempty_row = raw_nonempty_rows.reindex(tok.index).to_numpy(bool)
    r['empty_tokens_in_nonempty_rows'] = int(((tok == '') & nonempty_row).sum())
    r['ids_with_whitespace'] = int(((ex != tok) & (tok != '')).sum())
    tok = tok[tok != '']
    r['bad_prefix_or_format'] = int((~tok.str.match(r'^S[23]-\d+$')).sum())
    r['invalid_refs'] = int((~(tok.isin(s2_ids) | tok.isin(s3_ids))).sum())
    pairs = pd.DataFrame({'row': tok.index.to_numpy(), 'id': tok.to_numpy()})
    r['dup_within_row'] = int(pairs.duplicated().sum())
    pairs = pairs.drop_duplicates()
    r['ids_mapped_to_multiple_s1'] = int(pairs.id.duplicated().sum())
    r['total_matched_ids'] = int(len(pairs))
    return r


with Timer('integrity checks'):
    rep = [integrity_report(k, RAW[k], 'S' + k[-1]) for k in ['train_s1', 'train_s2', 'train_s3', 'test_s1', 'test_s2', 'test_s3']]
    integ = pd.DataFrame(rep).set_index('file')
    display(integ.drop(columns=['countries']))
    display(pd.DataFrame({k: v['countries'] for k, v in zip(integ.index, rep)}).fillna(0).astype(int))
    GT_INTEG = gt_integrity(RAW['train_gt'], RAW['train_s1'].entity_id, RAW['train_s2'].entity_id, RAW['train_s3'].entity_id)
    display(pd.Series(GT_INTEG, name='ground truth').to_frame())

new_c = sorted(set(c for r in rep for c in r['unseen_countries']))
bad = integ[['dup_ids', 'malformed_ids', 'wrong_prefix']].sum().sum() + GT_INTEG['invalid_refs'] + GT_INTEG['dup_s1_rows']
finding('ID / reference integrity',
        f"dup ids + malformed + wrong prefix + invalid GT refs + dup GT rows = {int(bad)}; "
        f"GT ids mapped to >1 S1 = {GT_INTEG['ids_mapped_to_multiple_s1']}",
        'Broken keys would silently corrupt labels, candidate recall and the submission.',
        'IDs are used as-is (string keys). GT is exploded once into a pair table. '
        + ('Each S2/S3 id belongs to at most one S1 → exclusivity constraint is available to the decision rule.'
           if GT_INTEG['ids_mapped_to_multiple_s1'] == 0 else 'Some ids map to several S1 → exclusivity NOT enforced.'))
finding('Countries unseen in training', f'{new_c} appear only in test',
        'A model with country-specific features/thresholds would be undefined on these rows.',
        'No country feature, no hard-coded country list; blocking is done *within* the (data-driven) country value; '
        'transliteration & legal-form handling are language-agnostic.')
finding('Missing / placeholder addresses',
        f"empty addr: " + ', '.join(f"{i}={integ.loc[i, 'addr_empty'] / integ.loc[i, 'rows']:.2%}" for i in integ.index)
        + f"; addr containing null/N/A tokens: " + ', '.join(f"{i}={integ.loc[i, 'addr_contains_null_token'] / integ.loc[i, 'rows']:.2%}" for i in integ.index),
        'Address similarity is undefined for these records; placeholder tokens create fake agreement.',
        'Placeholder tokens are removed during normalisation; missingness flags are model features; name-only retrieval passes exist.')
