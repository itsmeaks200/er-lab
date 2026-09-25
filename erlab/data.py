"""Dataset discovery, TSV loading, pool construction, ground truth, entity-level folds, mini-dataset maker."""
import os

import numpy as np
import pandas as pd

from .utils import STR, log

REQUIRED = {
    'train_s1': 'train_source1.tsv', 'train_s2': 'train_source2.tsv', 'train_s3': 'train_source3.tsv',
    'train_gt': 'train_ground_truth.tsv',
    'test_s1': 'test_source1.tsv', 'test_s2': 'test_source2.tsv', 'test_s3': 'test_source3.tsv',
}


def discover(data_dir):
    wanted = {v: k for k, v in REQUIRED.items()}
    found = {}
    for dp, dns, fns in os.walk(data_dir):
        dns[:] = sorted(d for d in dns if not d.startswith('.'))
        for fn in sorted(fns):
            if fn.lower() in wanted and wanted[fn.lower()] not in found:
                found[wanted[fn.lower()]] = os.path.join(dp, fn)
    missing = [v for k, v in REQUIRED.items() if k not in found]
    if missing:
        raise FileNotFoundError(f'missing {missing} under {os.path.abspath(data_dir)} (set --data or ER_DATA)')
    return found


def load_tsv(path, nrows=None):
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, na_filter=False, encoding='utf-8', nrows=nrows)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(STR)
    return df


def build_pool(s2, s3):
    s2 = s2.copy(); s2['src'] = np.int8(2)
    s3 = s3.copy(); s3['src'] = np.int8(3)
    P = pd.concat([s2, s3], ignore_index=True)
    P['country_key'] = P.country.str.strip().str.lower()
    return P.sort_values('country_key', kind='stable').reset_index(drop=True)


def ground_truth_pairs(gt):
    ex = gt[['source1_entity_id', 'matched_entity_ids']].astype(object).copy()
    ex['o_id'] = ex.matched_entity_ids.str.split(',')
    ex = ex.explode('o_id')
    ex['o_id'] = ex.o_id.str.strip()
    ex = ex[ex.o_id.notna() & (ex.o_id != '')]
    return pd.DataFrame({'s1_id': ex.source1_entity_id.to_numpy(), 'o_id': ex.o_id.to_numpy()}).drop_duplicates().reset_index(drop=True)


def entity_folds(ids, pcts):
    h = pd.util.hash_pandas_object(pd.Series(ids, dtype=object), index=False).to_numpy() % 100
    return np.where(h < pcts[0], 'fit', np.where(h < pcts[0] + pcts[1], 'tune', 'hold'))


def country_slices(Pdf):
    ck = Pdf.country_key.astype(object).to_numpy()
    if len(ck) == 0:
        return {}
    b = np.flatnonzero(ck[1:] != ck[:-1]) + 1
    starts, ends = np.r_[0, b], np.r_[b, len(ck)]
    return {ck[s]: (int(s), int(e)) for s, e in zip(starts, ends)}


def make_mini_dataset(data_dir, out_dir, n_train_s1=20_000, n_test_s1=10_000, distractor_frac=0.02, seed=42):
    """Small but structurally faithful copy of the dataset (for smoke tests): sampled S1 + all their matches +
    a random share of other pool records. Test: sampled test S1 + random share of the test pool."""
    paths = discover(data_dir)
    rng = np.random.RandomState(seed)
    os.makedirs(os.path.join(out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'test'), exist_ok=True)
    s1 = pd.read_csv(paths['train_s1'], sep='\t', dtype=str, keep_default_na=False)
    gt = pd.read_csv(paths['train_gt'], sep='\t', dtype=str, keep_default_na=False)
    keep = set(rng.choice(s1.entity_id.to_numpy(), min(n_train_s1, len(s1)), replace=False))
    s1 = s1[s1.entity_id.isin(keep)]
    gt = gt[gt.source1_entity_id.isin(keep)]
    matched = set(i for v in gt.matched_entity_ids for i in v.split(',') if i)
    s1.to_csv(os.path.join(out_dir, 'train', 'train_source1.tsv'), sep='\t', index=False)
    gt.to_csv(os.path.join(out_dir, 'train', 'train_ground_truth.tsv'), sep='\t', index=False)
    for k in ('train_s2', 'train_s3'):
        d = pd.read_csv(paths[k], sep='\t', dtype=str, keep_default_na=False)
        m = d.entity_id.isin(matched) | (rng.rand(len(d)) < distractor_frac)
        d[m].to_csv(os.path.join(out_dir, 'train', REQUIRED[k]), sep='\t', index=False)
    t1 = pd.read_csv(paths['test_s1'], sep='\t', dtype=str, keep_default_na=False)
    t1 = t1.sample(min(n_test_s1, len(t1)), random_state=seed)
    t1.to_csv(os.path.join(out_dir, 'test', 'test_source1.tsv'), sep='\t', index=False)
    frac = min(1.0, 3.5 * len(t1) / 1_732_544 * 1.3)
    for k in ('test_s2', 'test_s3'):
        d = pd.read_csv(paths[k], sep='\t', dtype=str, keep_default_na=False)
        d[rng.rand(len(d)) < frac].to_csv(os.path.join(out_dir, 'test', REQUIRED[k]), sep='\t', index=False)
    log(f'mini dataset written to {out_dir}')
