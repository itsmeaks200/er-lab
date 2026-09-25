"""Pair features for candidate pairs (identical code path for validation and test) + within-S1 context features
+ stage-2 cross-source confirmation features."""
import numpy as np
import pandas as pd

from . import text as T
from .blocking import PASS_BITS
from .vec import REC_COLS
from .utils import obj, log

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein as RF_Lev, JaroWinkler as RF_JW
    try:
        from rapidfuzz.process import cpdist
    except ImportError:
        cpdist = None
    HAS_RF = True
except ImportError:
    HAS_RF, cpdist = False, None

    class _NoRF:
        def __getattr__(self, k):
            return None
    fuzz = RF_Lev = RF_JW = _NoRF()


def rf(scorer, a, b, scale):
    if scorer is None:
        return np.full(len(a), np.nan, np.float32)
    if cpdist is not None:
        return (cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / scale).astype(np.float32)
    return (np.fromiter((scorer(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a)) / scale).astype(np.float32)


def rowdot(A, B, ia, ib):
    return np.asarray(A[ia].multiply(B[ib]).sum(axis=1), dtype=np.float32).ravel()


def _tok_block(f, pre, A, B, qi, pi, R, space, rare=False, sizes=False, seteq=False):
    a, b = A[qi], B[pi]
    inter = a.multiply(b).tocsr()
    na = np.diff(a.indptr).astype(np.float32); nb = np.diff(b.indptr).astype(np.float32)
    sh = np.diff(inter.indptr).astype(np.float32)
    idf, idf2 = R['IDF'][space], R['IDF2'][space]
    ia, ib, ish = a @ idf, b @ idf, inter @ idf
    qa, qb, qsh = a @ idf2, b @ idf2, inter @ idf2
    empty = (na == 0) | (nb == 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        f[pre + '_shared'] = sh
        f[pre + '_jacc'] = np.where(empty, np.nan, sh / np.maximum(na + nb - sh, 1)).astype(np.float32)
        f[pre + '_overlap'] = np.where(empty, np.nan, sh / np.maximum(np.minimum(na, nb), 1)).astype(np.float32)
        f[pre + '_wjacc'] = np.where(empty, np.nan, ish / np.maximum(ia + ib - ish, 1e-6)).astype(np.float32)
        f[pre + '_cos'] = np.where(empty, np.nan, qsh / np.sqrt(np.maximum(qa * qb, 1e-12))).astype(np.float32)
    if rare:
        f[pre + '_rare_shared'] = (inter @ R['RARE'][space]).astype(np.float32)
    if sizes:
        f[pre + '_n1'], f[pre + '_n2'] = na, nb
    if seteq:
        f[pre + '_set_eq'] = ((sh == na) & (sh == nb) & (na > 0)).astype(np.float32)
    return sh, na, nb


def pair_features(qi, pi, W, emb=None, alpha=0.6):
    """W: world dict with Q, P (frames), Qm, Pm (matrices), R (idf). emb: dict name -> (Qe, Pe) dense arrays."""
    Q_, P_, Qm_, Pm_, R = W['Q'], W['P'], W['Qm'], W['Pm'], W['R']
    f = {}
    g = lambda df, c, idx: df[c].take(idx).tolist()
    qnr, pnr = g(Q_, 'business_name', qi), g(P_, 'business_name', pi)
    qar, par = g(Q_, 'business_address', qi), g(P_, 'business_address', pi)
    qb, pb = g(Q_, 'n_basic', qi), g(P_, 'n_basic', pi)
    qc, pc = g(Q_, 'n_core', qi), g(P_, 'n_core', pi)
    qs, ps = g(Q_, 'n_squash', qi), g(P_, 'n_squash', pi)
    qk, pk = g(Q_, 'n_skel', qi), g(P_, 'n_skel', pi)
    qalt, palt = g(Q_, 'n_alt', qi), g(P_, 'n_alt', pi)
    qab, pab = g(Q_, 'a_basic', qi), g(P_, 'a_basic', pi)
    qac, pac = g(Q_, 'a_canon', qi), g(P_, 'a_canon', pi)
    a_empty = (obj(qac) == '') | (obj(pac) == '')

    def eq(a, b, mask=None):
        v = (obj(a) == obj(b)).astype(np.float32)
        if mask is not None:
            v[mask] = np.nan
        return v
    f['n_raw_eq'], f['n_basic_eq'], f['n_core_eq'] = eq(qnr, pnr), eq(qb, pb), eq(qc, pc)
    f['n_squash_eq'], f['n_skel_eq'] = eq(qs, ps), eq(qk, pk)
    f['a_raw_eq'], f['a_basic_eq'], f['a_canon_eq'] = eq(qar, par, a_empty), eq(qab, pab, a_empty), eq(qac, pac, a_empty)
    f['n_ratio'] = rf(fuzz.ratio, qb, pb, 100)
    f['n_lev'] = rf(RF_Lev.normalized_similarity, qc, pc, 1)
    f['n_jw'] = rf(RF_JW.normalized_similarity, qc, pc, 1)
    f['n_tsort'] = rf(fuzz.token_sort_ratio, qc, pc, 100)
    f['n_tset'] = rf(fuzz.token_set_ratio, qc, pc, 100)
    f['n_partial'] = rf(fuzz.partial_ratio, qs, ps, 100)
    f['n_sq_ratio'] = rf(fuzz.ratio, qs, ps, 100)
    f['n_skel_ratio'] = rf(fuzz.ratio, qk, pk, 100)
    alt1 = rf(fuzz.ratio, qc, palt, 100); alt1[obj(palt) == ''] = np.nan
    alt2 = rf(fuzz.ratio, qalt, pc, 100); alt2[obj(qalt) == ''] = np.nan
    f['n_alt_ratio'] = np.fmax(alt1, alt2)
    f['n_acronym'] = np.fromiter(((len(s2) >= 2 and T.acronym(c1) == s2) or (len(s1) >= 2 and T.acronym(c2) == s1)
                                  for c1, c2, s1, s2 in zip(qc, pc, qs, ps)), dtype=np.float32, count=len(qc))
    for name, sc in [('a_ratio', fuzz.ratio), ('a_tset', fuzz.token_set_ratio), ('a_tsort', fuzz.token_sort_ratio),
                     ('a_partial', fuzz.partial_ratio)]:
        v = rf(sc, qac, pac, 100); v[a_empty] = np.nan
        f[name] = v
    _tok_block(f, 'n_tok', Qm_['ntok'], Pm_['ntok'], qi, pi, R, 'ntok', rare=True, sizes=True, seteq=True)
    _tok_block(f, 'n_skl', Qm_['nskel'], Pm_['nskel'], qi, pi, R, 'nskel')
    f['n_char_cos'] = rowdot(Qm_['nchar'], Pm_['nchar'], qi, pi)
    _tok_block(f, 'a_tok', Qm_['atok'], Pm_['atok'], qi, pi, R, 'atok', rare=True, seteq=True)
    _tok_block(f, 'a_big', Qm_['abig'], Pm_['abig'], qi, pi, R, 'abig')
    sh, na, nb = _tok_block(f, 'a_num', Qm_['anum'], Pm_['anum'], qi, pi, R, 'anum', sizes=True)
    f['a_num_conflict'] = ((na > 0) & (nb > 0) & (sh == 0)).astype(np.float32)
    f['a_num_subset'] = ((sh == np.minimum(na, nb)) & (np.minimum(na, nb) > 0)).astype(np.float32)
    a_, b_ = Qm_['apost'][qi], Pm_['apost'][pi]
    pa_, pb_ = np.diff(a_.indptr), np.diff(b_.indptr)
    psh = np.diff(a_.multiply(b_).tocsr().indptr)
    f['a_post_eq'] = (psh > 0).astype(np.float32)
    f['a_post_conflict'] = ((pa_ > 0) & (pb_ > 0) & (psh == 0)).astype(np.float32)
    f['a_post_missing'] = ((pa_ == 0) | (pb_ == 0)).astype(np.float32)
    q0, p0 = Q_.num0.to_numpy()[qi], P_.num0.to_numpy()[pi]
    f['a_num0_eq'] = np.where((q0 < 0) | (p0 < 0), np.nan, (q0 == p0)).astype(np.float32)
    f['comb_cos'] = (alpha * np.nan_to_num(f['n_tok_cos']) + (1 - alpha) * np.nan_to_num(f['a_tok_cos'])).astype(np.float32)
    f['name_x_addr'] = (np.nan_to_num(f['n_tset']) * np.nan_to_num(f['a_tset'], nan=0.5)).astype(np.float32)
    for c in REC_COLS:
        f['q_' + c] = Q_[c].to_numpy()[qi].astype(np.float32)
        f['p_' + c] = P_[c].to_numpy()[pi].astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        f['n_len_ratio'] = (np.minimum(f['q_n_len'], f['p_n_len']) / np.maximum(np.maximum(f['q_n_len'], f['p_n_len']), 1)).astype(np.float32)
        f['a_len_ratio'] = np.where(a_empty, np.nan, np.minimum(f['q_a_len'], f['p_a_len']) /
                                    np.maximum(np.maximum(f['q_a_len'], f['p_a_len']), 1)).astype(np.float32)
    f['both_addr_missing'] = (f['q_a_missing'] * f['p_a_missing']).astype(np.float32)
    f['src_s3'] = (P_.src.to_numpy()[pi] == 3).astype(np.float32)
    for name, (Qe, Pe) in (emb or {}).items():
        f[name + '_cos'] = (Qe[qi].astype(np.float32) * Pe[pi].astype(np.float32)).sum(1)
    return pd.DataFrame(f)


GROUP_BASE = ['n_tset', 'n_char_cos', 'n_tok_wjacc', 'a_tset', 'a_tok_cos', 'comb_cos', 'name_x_addr',
              'embG_cos', 'embS_cos', 'pr_p']
CTX_PREFIX = ('blk_',)
CTX_SUFFIX = ('_gap', '_rank', '_z')


def add_context(F, qi, bits, passes):
    for p in passes:
        F['blk_' + p] = ((bits & PASS_BITS[p]) > 0).astype(np.float32)
    _, inv, cnt = np.unique(qi, return_inverse=True, return_counts=True)
    F['q_ncand'] = cnt[inv.ravel()].astype(np.float32)
    for c in GROUP_BASE:
        if c not in F:
            continue
        v = pd.Series(F[c].fillna(-1).to_numpy(np.float32))
        grp = v.groupby(qi)
        mx, mu, sd = grp.transform('max').to_numpy(), grp.transform('mean').to_numpy(), grp.transform('std').fillna(0).to_numpy()
        F[c + '_gap'] = (mx - v.to_numpy()).astype(np.float32)
        F[c + '_rank'] = grp.rank(ascending=False, method='min').to_numpy(np.float32)
        F[c + '_z'] = ((v.to_numpy() - mu) / (sd + 1e-3)).astype(np.float32)
    return F


def drop_context(F):
    return F.drop(columns=[c for c in F.columns if c.startswith(CTX_PREFIX) or c.endswith(CTX_SUFFIX) or c == 'q_ncand'])


def safe_zone(F):
    name_sim = np.fmax.reduce([F['n_tok_jacc'].fillna(0).to_numpy(), F['n_skl_jacc'].fillna(0).to_numpy(),
                               F['n_char_cos'].fillna(0).to_numpy()])
    addr_sim = np.fmax(F['a_tok_jacc'].fillna(0).to_numpy(), F['a_num_jacc'].fillna(0).to_numpy())
    return (name_sim < 0.2) & (addr_sim < 0.2)


def group_chunks(qi_sorted, chunk):
    n, bounds, s = len(qi_sorted), [], 0
    while s < n:
        e = min(s + chunk, n)
        if e < n:
            e = int(np.searchsorted(qi_sorted, qi_sorted[e - 1], side='right'))
        bounds.append((s, e))
        s = e
    return bounds


def compute_features(cand, W, passes, emb=None, prune=False, chunk=1_000_000, scorer=None, keep_cols=None, alpha=0.6,
                     extra=None):
    """scorer=None → (features, keep mask); else streams: (scores, keep mask, kept columns frame).
    extra: {column: array aligned with cand} appended as features (e.g. the stage-1 pruner score pr_p)."""
    qi_all, pi_all, bits_all = cand.qi.to_numpy(), cand.pi.to_numpy(), cand.bits.to_numpy()
    keep = np.ones(len(cand), bool)
    outs, kept = [], []
    bounds = group_chunks(qi_all, chunk)
    for j, (s, e) in enumerate(bounds):
        F = pair_features(qi_all[s:e], pi_all[s:e], W, emb, alpha)
        for c, v in (extra or {}).items():
            F[c] = np.asarray(v[s:e], np.float32)
        if prune:
            k = ~safe_zone(F)
            keep[s:e] = k
            F = F[k].reset_index(drop=True)
        add_context(F, qi_all[s:e][keep[s:e]], bits_all[s:e][keep[s:e]], passes)
        if scorer is None:
            outs.append(F)
        else:
            outs.append(np.asarray(scorer(F), np.float32))
            if keep_cols:
                kept.append(F[keep_cols].astype(np.float32))
        if j % 10 == 0 or j == len(bounds) - 1:
            log(f'   features chunk {j + 1}/{len(bounds)} ({e:,}/{len(cand):,} pairs)')
    if scorer is None:
        return pd.concat(outs, ignore_index=True), keep
    return np.concatenate(outs) if outs else np.empty(0, np.float32), keep, (pd.concat(kept, ignore_index=True) if kept else None)


# ---------------------------------------------------------------- stage 2: cross-source confirmation
def stage2_chunk(qi, pi, s3, p1, P_, Pm_):
    n = len(qi)
    s3 = s3.astype(np.int64)
    df = pd.DataFrame({'qi': qi, 'p1': p1, 'h': (p1 >= 0.5).astype(np.float32)})
    g = df.groupby('qi')
    F = {'p1': p1.astype(np.float32)}
    F['p1_rank'] = g.p1.rank(ascending=False, method='min').to_numpy(np.float32)
    F['p1_qmax'] = g.p1.transform('max').to_numpy(np.float32)
    F['p1_gap'] = F['p1_qmax'] - F['p1']
    F['p1_q_n50'] = g.h.transform('sum').to_numpy(np.float32)
    F['p1_src_n50'] = df.groupby([qi, s3]).h.transform('sum').to_numpy(np.float32)
    order = np.lexsort((-p1, s3, qi))
    q_o, s_o = qi[order], s3[order]
    first = np.r_[True, (q_o[1:] != q_o[:-1]) | (s_o[1:] != s_o[:-1])]
    start = np.flatnonzero(first)
    size = np.diff(np.r_[start, n])
    best_row = order[start]
    second_row = np.where(size > 1, order[np.minimum(start + 1, n - 1)], -1)
    gid = np.empty(n, np.int64); gid[order] = np.cumsum(first) - 1
    gkey = q_o[start].astype(np.int64) * 2 + s_o[start]
    okey = qi.astype(np.int64) * 2 + (1 - s3)
    pos = np.minimum(np.searchsorted(gkey, okey), len(gkey) - 1)
    other_best = np.where(gkey[pos] == okey, best_row[pos], -1)
    own_best, own_second = best_row[gid], second_row[gid]
    same_other = np.where(own_best == np.arange(n), own_second, own_best)
    for tag, ref in (('xsrc', other_best), ('ssrc', same_other)):
        ok = ref >= 0
        a, b = pi[ok], pi[ref[ok]]
        rp = np.full(n, np.nan, np.float32); rp[ok] = p1[ref[ok]]
        nt = np.full(n, np.nan, np.float32); at = np.full(n, np.nan, np.float32); cc = np.full(n, np.nan, np.float32)
        if ok.any():
            nt[ok] = rf(fuzz.token_set_ratio, P_.n_core.take(a).tolist(), P_.n_core.take(b).tolist(), 100)
            ac1, ac2 = P_.a_canon.take(a).tolist(), P_.a_canon.take(b).tolist()
            v = rf(fuzz.token_set_ratio, ac1, ac2, 100)
            v[(obj(ac1) == '') | (obj(ac2) == '')] = np.nan
            at[ok] = v
            cc[ok] = rowdot(Pm_['nchar'], Pm_['nchar'], a, b)
        F[tag + '_p1'], F[tag + '_n_tset'], F[tag + '_a_tset'], F[tag + '_char'] = rp, nt, at, cc
        F[tag + '_support'] = (np.nan_to_num(rp) * np.fmax(np.nan_to_num(nt), np.nan_to_num(at))).astype(np.float32)
    return pd.DataFrame(F)


def stage2_frame(qi, pi, s3, p1, P_, Pm_, keep_df, chunk=2_000_000):
    parts = [stage2_chunk(qi[s:e], pi[s:e], s3[s:e], p1[s:e], P_, Pm_) for s, e in group_chunks(qi, chunk)]
    S = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return pd.concat([S, keep_df.reset_index(drop=True)], axis=1) if keep_df is not None else S
