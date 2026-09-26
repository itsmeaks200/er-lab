"""Stage 1 = learned blocking ("pruner"). Sits between label-free retrieval (stage 0 = union of blocking passes) and the
matcher. It sees only cheap, vectorised signals: retrieval scores/ranks, sparse token/char cosines, dense embedding
cosines, house-number/postal agreement, record attributes and within-S1 (optionally within-pool-record) ranks.
No string alignment (rapidfuzz), so it costs O(nnz) per pair. A small LightGBM scores every stage-0 pair, and a cut
(probability threshold t, top-n per S1, optional top-m S1 per pool record) keeps the smallest candidate set whose
oracle-F0.5 loss on the tune fold stays within a budget. Its survivors are exactly what the matcher scores, i.e. the
content of candidate_pairs.tsv."""
import os

import numpy as np
import pandas as pd

from . import features as FT
from . import evaluate as EV
from .blocking import PASS_BITS, KEY_PASSES
from .models import Model
from .utils import log, Timer, Progress

REC = ['a_missing', 'n_ntok', 'a_ntok', 'nf_s1', 'nf_pool', 'f_nonascii', 'f_domain', 'f_junk']
CTX = ['n_tok_cos', 'a_tok_cos', 'n_char_cos', 'comb', 'embG_cos', 'embS_cos']
PARAMS = dict(num_leaves=63, learning_rate=0.08, min_data_in_leaf=200, feature_fraction=0.8, lambda_l2=1.0)


def _seg_rank_gap_py(order, starts, v, rank, gap):
    """Per group segment (rows order[starts[s]:starts[s+1]], in position order): stable rank by descending v and gap
    to the segment max. Parallel over segments."""
    for s in prange(len(starts) - 1):
        a, b = starts[s], starts[s + 1]
        n = b - a
        vals = np.empty(n, np.float32)
        for u in range(n):
            vals[u] = v[order[a + u]]
        o = np.argsort(-vals, kind='mergesort')
        mx = vals[o[0]]
        for r in range(n):
            rank[order[a + o[r]]] = r + 1
        for u in range(n):
            gap[order[a + u]] = mx - vals[u]


try:
    import numba
    from numba import prange
    _seg_rank_gap = numba.njit(parallel=True, cache=True)(_seg_rank_gap_py)
except Exception:
    prange = range
    _seg_rank_gap = None


def group_rank_gap(g, v):
    """Within groups g: rank of v (1 = best, ties broken by position) and gap to the group max."""
    v = np.nan_to_num(np.asarray(v, np.float32), nan=-1.0)
    n = len(v)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    if _seg_rank_gap is not None:              # parallel path: one stable sort by group, then per-segment sorts
        g = np.asarray(g)
        order = np.arange(n) if np.all(g[1:] >= g[:-1]) else np.argsort(g, kind='stable')
        gs = g[order]
        starts = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, n].astype(np.int64)
        rank, gap = np.empty(n, np.float32), np.empty(n, np.float32)
        _seg_rank_gap(order.astype(np.int64), starts, v, rank, gap)
        return rank, gap
    order = np.lexsort((-v, g))
    gs = g[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    rank = np.empty(n, np.float32)
    rank[order] = np.arange(n) - start + 1
    mx = np.empty(n, np.float32)
    mx[order] = v[order][start]
    return rank, mx - v


def _chunk_feats(q, p, W, emb):
    Qm, Pm, R = W['Qm'], W['Pm'], W['R']
    f = {}
    FT._tok_block(f, 'n_tok', Qm['ntok'], Pm['ntok'], q, p, R, 'ntok', rare=True)
    FT._tok_block(f, 'a_tok', Qm['atok'], Pm['atok'], q, p, R, 'atok', rare=True)
    sh, na, nb = FT._tok_block(f, 'a_num', Qm['anum'], Pm['anum'], q, p, R, 'anum', sizes=True)
    f['a_num_conflict'] = ((na > 0) & (nb > 0) & (sh == 0)).astype(np.float32)
    f['n_char_cos'] = FT.rowdot(Qm['nchar'], Pm['nchar'], q, p)
    a_, b_ = Qm['apost'][q], Pm['apost'][p]
    pa_, pb_, psh = np.diff(a_.indptr), np.diff(b_.indptr), np.diff(a_.multiply(b_).tocsr().indptr)
    f['a_post_eq'] = (psh > 0).astype(np.float32)
    f['a_post_conflict'] = ((pa_ > 0) & (pb_ > 0) & (psh == 0)).astype(np.float32)
    for name in sorted(emb or {}):
        Qe, Pe = emb[name]
        f[name + '_cos'] = (Qe[q].astype(np.float32) * Pe[p].astype(np.float32)).sum(1)
    return f


def cheap_features(cand, W, emb, passes, alpha=0.6, pool_ctx=False, chunk=4_000_000):
    """Identical code path for the train and test worlds. cand: stage-0 union (qi, pi, bits, s_X/r_X).
    Returns (X float32 [n_pairs, n_feats], column names), written in place to keep peak memory at ~1x."""
    qi, pi = cand.qi.to_numpy(), cand.pi.to_numpy()
    n = len(cand)
    X, cols, local = None, None, None
    pg = Progress('stage-1 cheap features', n, 'pairs')
    for s in range(0, max(n, 1), chunk):
        f = _chunk_feats(qi[s:s + chunk], pi[s:s + chunk], W, emb)
        if X is None:
            local = list(f)
            glob = ['comb'] + ['blk_' + p for p in passes]
            glob += [c for p in passes if 's_' + p in cand for c in ('s_' + p, 'r_' + p)]
            glob += [side + c for c in REC for side in ('q_', 'p_')] + ['src_s3', 'q_ncand'] + (['p_ncand'] if pool_ctx else [])
            for c in CTX:
                if c in local or c == 'comb':
                    glob += [c + '_qrank', c + '_qgap'] + ([c + '_prank', c + '_pgap'] if pool_ctx else [])
            cols = local + glob
            X = np.empty((n, len(cols)), np.float32)
        for j, c in enumerate(local):
            X[s:s + chunk, j] = f[c]
        del f
        pg.update(min(s + chunk, n))
    log('   stage-1 cheap features: record attributes, pass scores, within-S1 / within-pool ranks')
    ix = {c: j for j, c in enumerate(cols)}
    X[:, ix['comb']] = alpha * np.nan_to_num(X[:, ix['n_tok_cos']]) + (1 - alpha) * np.nan_to_num(X[:, ix['a_tok_cos']])
    bits = cand.bits.to_numpy()
    for p in passes:
        X[:, ix['blk_' + p]] = (bits & PASS_BITS[p]) > 0
        if 's_' + p in cand:
            X[:, ix['s_' + p]] = cand['s_' + p].to_numpy(np.float32)
            X[:, ix['r_' + p]] = cand['r_' + p].to_numpy(np.float32)
    for c in REC:
        X[:, ix['q_' + c]] = W['Q'][c].to_numpy()[qi]
        X[:, ix['p_' + c]] = W['P'][c].to_numpy()[pi]
    X[:, ix['src_s3']] = W['P'].src.to_numpy()[pi] == 3
    X[:, ix['q_ncand']] = np.bincount(qi, minlength=W['NQ'])[qi]
    if pool_ctx:
        X[:, ix['p_ncand']] = np.bincount(pi, minlength=W['NP'])[pi]
    for c in CTX:
        if c + '_qrank' not in ix:
            continue
        v = X[:, ix[c]]
        X[:, ix[c + '_qrank']], X[:, ix[c + '_qgap']] = group_rank_gap(qi, v)
        if pool_ctx:
            X[:, ix[c + '_prank']], X[:, ix[c + '_pgap']] = group_rank_gap(pi, v)
    return X, cols


def predict_rows(m, X, rows, chunk=5_000_000):
    out = np.empty(len(rows), np.float32)
    for s in range(0, len(rows), chunk):
        out[s:s + chunk] = m.predict(X[rows[s:s + chunk]])
    return out


def _cap_by_s1(qi, rows, max_rows, seed):
    if len(rows) <= max_rows:
        return rows
    qs = np.unique(qi[rows])
    keep = np.random.RandomState(seed).choice(qs, int(len(qs) * max_rows / len(rows)), replace=False)
    return rows[np.isin(qi[rows], keep)]


def fit_pruner(X, y, qi, rows, q_hash, pc, seed=42):
    """Full-fit model (early stopping on tune) scores tune/hold/test; fit rows get 2-fold out-of-fold scores, so the
    matcher is trained on survivors selected the same way as at inference."""
    fit, tune = rows['fit'], rows['tune']
    params = dict(PARAMS, **pc.get('params', {}))
    tr = _cap_by_s1(qi, fit, pc['max_train_rows'], seed)
    va = _cap_by_s1(qi, tune, max(pc['max_train_rows'] // 4, 1), seed)
    with Timer(f'stage-1 pruner fit ({X.shape[1]} cheap feats, {len(tr):,} rows)'):
        full = Model('lgb', params, pc['rounds'], pc['early_stop'], seed).fit(X[tr], y[tr], None, X[va], y[va], None)
    p = np.zeros(len(y), np.float32)
    rest = np.setdiff1d(np.arange(len(y)), fit)
    p[rest] = predict_rows(full, X, rest)
    rounds = max(50, int(full.best_iter * 1.05))
    h = q_hash[qi]
    for k in range(2):
        log(f'   stage-1 pruner: out-of-fold model {k + 1}/2 ({rounds} rounds)')
        trk, prk = tr[h[tr] != k], fit[h[fit] == k]
        mk = Model('lgb', params, rounds, 10 ** 9, seed).fit_fixed(X[trk], y[trk])
        p[prk] = predict_rows(mk, X, prk)
    return p, full


def ranks_of(p, qi, pi, pool_rank=False):
    qr, _ = group_rank_gap(qi, p)
    pr = group_rank_gap(pi, p)[0] if pool_rank else None
    return qr, pr


def cut_mask(p, qr, pr, prm):
    ok = p >= prm['t']
    if prm.get('n_max'):
        ok &= qr <= prm['n_max']
    if prm.get('m_pool') and pr is not None:
        ok &= pr <= prm['m_pool']
    return ok


def frontier(p, y, qi, qr, pr, rows, q_eval, n_true, pc):
    """Grid of cuts evaluated on one fold: candidates per S1, pair recall, oracle F0.5."""
    p, y, q, qr_ = p[rows], y[rows], qi[rows], qr[rows]
    pr_ = pr[rows] if pr is not None else None
    n_pos = max(int(n_true[q_eval].sum()), 1)
    out = []
    for m in (pc['m_grid'] if pr is not None else [0]):
        for n in pc['n_grid']:
            for t in pc['t_grid']:
                prm = dict(t=t, n_max=n, m_pool=m)
                k = cut_mask(p, qr_, pr_, prm)
                ora = EV.s1_metrics(q[k & (y == 1)], np.ones(int((k & (y == 1)).sum())), n_true, q_eval)['F05']
                per_q = np.bincount(q[k], minlength=len(n_true))[q_eval]
                out.append(dict(t=t, n_max=n, m_pool=m, cand_per_S1=per_q.mean(), p50=float(np.median(per_q)),
                                p95=float(np.percentile(per_q, 95)), max=int(per_q.max()) if len(per_q) else 0,
                                pair_recall=y[k].sum() / n_pos, oracle_F05=ora))
    return pd.DataFrame(out)


def choose_cut(fr, oracle_before, budget):
    ok = fr[oracle_before - fr.oracle_F05 <= budget + 1e-12]
    if not len(ok):
        ok = fr.sort_values('oracle_F05', ascending=False).head(1)
    best = ok.sort_values(['cand_per_S1', 'oracle_F05'], ascending=[True, False]).iloc[0]
    return dict(t=float(best.t), n_max=int(best.n_max), m_pool=int(best.m_pool))


def pareto(fr):
    """Rows not dominated in (fewer candidates, higher oracle F0.5)."""
    fr = fr.sort_values(['cand_per_S1', 'oracle_F05'], ascending=[True, False])
    best, keep = -1.0, []
    for i, r in fr.iterrows():
        if r.oracle_F05 > best + 1e-9:
            keep.append(i)
            best = r.oracle_F05
    return fr.loc[keep]


def save_pruner(m, path):
    m.obj.save_model(path, num_iteration=m.best_iter)


def load_pruner(path):
    import lightgbm as lgb
    m = Model('lgb', PARAMS)
    m.obj = lgb.Booster(model_file=path)
    m.best_iter = m.obj.current_iteration()
    return m
