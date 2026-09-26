"""Label-free candidate generation. Passes:
A exact squashed core name | B name-skeleton prefix | E house#+street key | H postal + name-skeleton token  (exact keys)
C TF-IDF name tokens | D TF-IDF address tokens | F combined name+address | I name char-3gram TF-IDF  (sparse top-K)
G learned hash-encoder | S sentence-transformer  (dense top-K on GPU)
All passes are computed within country (0 cross-country links in the ground truth)."""
import time
import gc

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize as sk_normalize

from . import text as T
from .utils import umap, log, Progress

KEY_PASSES = ['A', 'B', 'E', 'H']
SPARSE_PASSES = ['C', 'D', 'F', 'I']
DENSE_PASSES = ['G', 'S']
PASS_NAMES = {'A': 'A exact squashed name', 'B': 'B name-skeleton prefix', 'C': 'C TF-IDF name', 'D': 'D TF-IDF address',
              'E': 'E house#+street key', 'F': 'F TF-IDF name+address', 'G': 'G hash-encoder dense',
              'H': 'H postal+name key', 'I': 'I char-3gram name', 'S': 'S sentence-transformer dense'}
PASS_BITS = {p: 1 << i for i, p in enumerate('ABCDEFGHIS')}          # 10 bits → uint16
RETR_FIELDS = {'name': ['ntok', 'nskel'], 'addr': ['atok', 'anum', 'abig'], 'char': ['nchar']}


def _topk_rows_py(indptr, indices, data, k):
    """Reference implementation (also the numba kernel source)."""
    n = indptr.shape[0] - 1
    tot = 0
    for i in range(n):
        m = indptr[i + 1] - indptr[i]
        tot += m if m < k else k
    out_r = np.empty(tot, np.int32); out_c = np.empty(tot, np.int32)
    out_s = np.empty(tot, np.float32); out_k = np.empty(tot, np.int16)
    p = 0
    for i in range(n):
        s = indptr[i]; e = indptr[i + 1]; m = e - s
        if m == 0:
            continue
        order = np.argsort(-data[s:e])
        kk = m if m < k else k
        for j in range(kk):
            t = s + order[j]
            out_r[p] = i; out_c[p] = indices[t]; out_s[p] = data[t]; out_k[p] = j
            p += 1
    return out_r, out_c, out_s, out_k


def _topk_rows_np(indptr, indices, data, k):
    """Vectorised-per-row numpy fallback (used when numba is unavailable)."""
    R_, C_, S_, K_ = [], [], [], []
    for i in range(len(indptr) - 1):
        s, e = indptr[i], indptr[i + 1]
        if e == s:
            continue
        d = data[s:e]
        part = np.argpartition(-d, k - 1)[:k] if e - s > k else np.arange(e - s)
        part = part[np.argsort(-d[part], kind='stable')]
        R_.append(np.full(len(part), i, np.int32)); C_.append(indices[s:e][part]); S_.append(d[part])
        K_.append(np.arange(len(part), dtype=np.int16))
    if not R_:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32), np.empty(0, np.int16)
    return (np.concatenate(R_), np.concatenate(C_).astype(np.int32), np.concatenate(S_).astype(np.float32),
            np.concatenate(K_))


def _spmm_topk_py(q_indptr, q_indices, q_data, pt_indptr, pt_indices, pt_data, n_pool, k, nthreads):
    """Fused sparse retrieval: for every query row, accumulate scores over the posting lists of its tokens into a dense
    per-thread buffer, keep the top-k. Parallel over query blocks (numba prange); never materialises the score matrix."""
    nq = len(q_indptr) - 1
    out_c = np.full((nq, k), -1, np.int32)
    out_s = np.zeros((nq, k), np.float32)
    out_n = np.zeros(nq, np.int32)
    per = (nq + nthreads - 1) // nthreads
    for t in prange(nthreads):
        acc = np.zeros(n_pool, np.float32)
        touched = np.empty(n_pool, np.int32)
        for i in range(t * per, min(nq, (t + 1) * per)):
            nt = 0
            for jj in range(q_indptr[i], q_indptr[i + 1]):
                tok = q_indices[jj]
                w = q_data[jj]
                for pp in range(pt_indptr[tok], pt_indptr[tok + 1]):
                    c = pt_indices[pp]
                    if acc[c] == 0.0:
                        touched[nt] = c
                        nt += 1
                    acc[c] += w * pt_data[pp]
            if nt > 0:
                vals = np.empty(nt, np.float32)
                for u in range(nt):
                    vals[u] = acc[touched[u]]
                m = k if nt > k else nt
                order = np.argsort(-vals, kind='mergesort')
                for u in range(m):
                    out_c[i, u] = touched[order[u]]
                    out_s[i, u] = vals[order[u]]
                out_n[i] = m
                for u in range(nt):
                    acc[touched[u]] = 0.0
    return out_c, out_s, out_n


try:
    import numba
    from numba import prange
    _topk_rows = numba.njit(_topk_rows_py)
    _spmm_topk = numba.njit(parallel=True, cache=True)(_spmm_topk_py)
    HAS_NUMBA = True
except Exception:
    prange = range
    _topk_rows = _topk_rows_np
    _spmm_topk = None
    HAS_NUMBA = False


def spmm_topk(Q, PT, k, nthreads=None):
    """Top-k of Q @ PT per row (Q: queries x tokens CSR, PT: tokens x pool CSR) with the parallel kernel."""
    Q, PT = Q.tocsr(), PT.tocsr()
    nthreads = nthreads or numba.get_num_threads()
    c, v, n = _spmm_topk(Q.indptr.astype(np.int64), Q.indices.astype(np.int32), Q.data.astype(np.float32),
                         PT.indptr.astype(np.int64), PT.indices.astype(np.int32), PT.data.astype(np.float32),
                         PT.shape[1], k, max(1, min(nthreads, Q.shape[0])))
    return c, v, n


def topk_csr(S, k):
    S = S.tocsr()
    return _topk_rows(S.indptr, S.indices, S.data.astype(np.float32, copy=False), k)


def scale_cols(X, w):
    X = sp.csr_matrix((X.data * w[X.indices], X.indices.copy(), X.indptr.copy()), shape=X.shape)
    X.eliminate_zeros()
    return X


def retr_mat(M, rows, fields, W, side='q', bm25=False):
    X = sp.hstack([scale_cols(M[f][rows], W[f]) for f in fields], format='csr')
    if bm25 and side == 'p':                     # BM25 document-length saturation (k1=1.2, b=0.75), binary tf
        L = np.diff(X.indptr).astype(np.float32)
        f = (2.2 / (1 + 1.2 * (0.25 + 0.75 * L / max(L.mean(), 1e-6)))).astype(np.float32)
        return sp.diags(f) @ X
    if bm25 and side == 'q':
        X.data[:] = 1.0
        return X
    return sk_normalize(X, norm='l2', copy=False)


def _empty():
    return pd.DataFrame({'qi': np.empty(0, np.int32), 'pi': np.empty(0, np.int32),
                         'score': np.empty(0, np.float32), 'rank': np.empty(0, np.int16)})


def country_groups(q_rows, q_ck, p_slices, n_pool):
    ck = q_ck[q_rows]
    groups = []
    for c, (s, e) in p_slices.items():
        r = q_rows[ck == c]
        if len(r):
            groups.append((c, r, s, e))
    unknown = ~np.isin(ck, np.array(list(p_slices.keys()), dtype=object))
    if unknown.any():
        groups.append(('<country-not-in-pool>', q_rows[unknown], 0, n_pool))
    return groups


def sparse_passes(Qm, Pm, q_rows, q_ck, p_slices, R, k, which, cfg):
    """TF-IDF (or BM25) top-K retrieval for passes C/D/F/I; chunked sparse products, never a Cartesian product."""
    rc = cfg['retr']
    chunk, alpha, bm25 = rc['chunk'], rc['alpha_name'], rc['bm25']
    which = [p for p in which if p in SPARSE_PASSES]
    out = {p: [] for p in which}
    need = {'name': any(p in which for p in 'CF'), 'addr': any(p in which for p in 'DF'), 'char': 'I' in which}
    parallel = HAS_NUMBA and rc.get('parallel', True)
    for c, rows_c, ps, pe in country_groups(np.asarray(q_rows), q_ck, p_slices, Pm['ntok'].shape[0]):
        t = time.time()
        log(f'   sparse {which} [{c}]: preparing pool matrices ({pe - ps:,} pool records)')
        PT = {f: retr_mat(Pm, slice(ps, pe), RETR_FIELDS[f], R['W'], 'p', bm25).T.tocsr() for f in need if need[f]}
        pg = Progress(f'sparse {which} [{c}]' + (' parallel' if parallel else ''), len(rows_c), 'queries')
        if parallel:        # fused parallel kernel on all cores; F = [a*name, (1-a)*addr] . [name; addr]
            if 'F' in which:
                PT['comb'] = sp.vstack([PT['name'], PT['addr']], format='csr')
            pchunk = max(chunk, 200_000)
            for s in range(0, len(rows_c), pchunk):
                qi = rows_c[s:s + pchunk]
                QM = {f: retr_mat(Qm, qi, RETR_FIELDS[f], R['W'], 'q', bm25) for f in need if need[f]}
                todo = [(p_, QM['name'], PT['name']) for p_ in 'C' if p_ in which]
                todo += [(p_, QM['addr'], PT['addr']) for p_ in 'D' if p_ in which]
                todo += [(p_, QM['char'], PT['char']) for p_ in 'I' if p_ in which]
                if 'F' in which:
                    todo.append(('F', sp.hstack([QM['name'] * alpha, QM['addr'] * (1 - alpha)], format='csr'), PT['comb']))
                for p_, Qx, PTx in todo:
                    cc_, vv, nn = spmm_topk(Qx, PTx, k)
                    r = np.repeat(np.arange(len(qi)), nn)
                    j = np.arange(len(r)) - np.repeat(np.cumsum(nn) - nn, nn)          # rank within the query
                    out[p_].append(pd.DataFrame({'qi': qi[r].astype(np.int32), 'pi': (cc_[r, j].astype(np.int64) + ps).astype(np.int32),
                                                 'score': vv[r, j], 'rank': j.astype(np.int16)}))
                pg.update(min(s + pchunk, len(rows_c)), force=True)
            log(f'   sparse {which} [{c}] {len(rows_c):,} q x {pe - ps:,} pool: {time.time() - t:.0f}s')
            del PT
            gc.collect()
            continue
        for s in range(0, len(rows_c), chunk):
            qi = rows_c[s:s + chunk]
            SC = {f: (retr_mat(Qm, qi, RETR_FIELDS[f], R['W'], 'q', bm25) @ PT[f]).tocsr() for f in PT}
            mats = {}
            if 'C' in which: mats['C'] = SC['name']
            if 'D' in which: mats['D'] = SC['addr']
            if 'F' in which: mats['F'] = (SC['name'] * alpha + SC['addr'] * (1 - alpha)).tocsr()
            if 'I' in which: mats['I'] = SC['char']
            for p, S in mats.items():
                r, cix, sc, rk = topk_csr(S, k)
                out[p].append(pd.DataFrame({'qi': qi[r].astype(np.int32), 'pi': (cix.astype(np.int64) + ps).astype(np.int32),
                                            'score': sc, 'rank': rk}))
            pg.update(min(s + chunk, len(rows_c)))
        log(f'   sparse {which} [{c}] {len(rows_c):,} q x {pe - ps:,} pool: {time.time() - t:.0f}s')
        del PT
        gc.collect()
    return {p: (pd.concat(v, ignore_index=True) if v else _empty()) for p, v in out.items()}


def record_keys(df, kind):
    ck = df.country_key.astype(object).to_numpy()
    if kind in ('A', 'B', 'K'):
        if kind == 'A':
            k, minlen = df.n_squash, 3
        else:
            k, minlen = df.n_skel.str.replace(' ', '', regex=False), 4
            if kind == 'B':
                k = k.str[:8]
        rows = np.flatnonzero(k.str.len().to_numpy(np.int64) >= minlen)
        vals = ck[rows] + '|' + k.astype(object).to_numpy()[rows]
    elif kind == 'S':
        k = umap(df.a_canon, T.addr_set_key).astype(object).to_numpy()
        rows = np.flatnonzero(df.a_ntok.to_numpy() >= 3)
        vals = ck[rows] + '|' + k[rows]
    elif kind in ('E', 'H'):
        ks = umap(df.a_canon, T.addr_keys if kind == 'E' else T.postal_str).astype(object).str.split().explode()
        ks = ks[ks.notna() & (ks != '')]
        rows = ks.index.to_numpy()
        suffix = ks.to_numpy(object)
        if kind == 'H':
            first = df.n_skel.str.split(' ').str[0].str[:3].astype(object).to_numpy()
            suffix = suffix + '|' + first[rows]
        vals = ck[rows] + '|' + suffix
    else:
        raise ValueError(kind)
    h = pd.util.hash_array(np.asarray(vals, dtype=object)) if len(rows) else np.empty(0, np.uint64)
    return np.asarray(rows, np.int64), h


def key_pass(kind, Qdf, Pdf, q_rows, max_block, cache=None):
    if cache is not None and kind in cache:
        (qr, qh), (pr, ph) = cache[kind]
    else:
        (qr, qh), (pr, ph) = record_keys(Qdf, kind), record_keys(Pdf, kind)
        if cache is not None:
            cache[kind] = ((qr, qh), (pr, ph))
    sel = np.isin(qr, q_rows)
    qdf = pd.DataFrame({'h': qh[sel], 'qi': qr[sel]})
    pdf_ = pd.DataFrame({'h': ph, 'pi': pr})
    pdf_ = pdf_[pdf_.h.isin(qdf.h.unique())]
    vc = pdf_.h.value_counts()
    ok = vc.index[vc.to_numpy() <= max_block]
    m = qdf[qdf.h.isin(ok)].merge(pdf_, on='h')[['qi', 'pi']].drop_duplicates()
    return pd.DataFrame({'qi': m.qi.to_numpy(np.int32), 'pi': m.pi.to_numpy(np.int32),
                         'score': np.ones(len(m), np.float32), 'rank': np.zeros(len(m), np.int16)})


def dense_pass(Qe, Pe, q_rows, q_ck, p_slices, k, chunk=512):
    """Exact dense top-K on GPU (fp16) within country; CPU fallback with numpy for small pools."""
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    R_, C_, S_, K_ = [], [], [], []
    for c, rows_c, ps, pe in country_groups(np.asarray(q_rows), q_ck, p_slices, len(Pe)):
        t = time.time()
        pg = Progress(f'dense [{c}]', len(rows_c), 'queries')
        Pt = torch.from_numpy(Pe[ps:pe]).to(dev)
        if dev == 'cpu':
            Pt = Pt.float()
        kk = min(k, pe - ps)
        with torch.no_grad():
            for s in range(0, len(rows_c), chunk):
                qi = rows_c[s:s + chunk]
                q = torch.from_numpy(Qe[qi]).to(dev)
                if dev == 'cpu':
                    q = q.float()
                v, ix = torch.topk(q @ Pt.T, kk, dim=1)
                R_.append(np.repeat(qi, kk).astype(np.int32)); C_.append((ix.cpu().numpy().ravel() + ps).astype(np.int32))
                S_.append(v.float().cpu().numpy().ravel()); K_.append(np.tile(np.arange(kk, dtype=np.int16), len(qi)))
                pg.update(min(s + chunk, len(rows_c)))
        del Pt
        if dev == 'cuda':
            torch.cuda.empty_cache()
        log(f'   dense [{c}] {len(rows_c):,} q x {pe - ps:,} pool: {time.time() - t:.0f}s')
    if not R_:
        return _empty()
    return pd.DataFrame({'qi': np.concatenate(R_), 'pi': np.concatenate(C_), 'score': np.concatenate(S_),
                         'rank': np.concatenate(K_)})


def union_candidates(pass_dfs, K_sel, n_pool, scores=True):
    """Union of the selected passes with a pass bitmask; for ranked passes also the retrieval score s_X and rank r_X
    (1-based, NaN when the pass did not retrieve the pair)."""
    keys, bits, kept = [], [], {}
    for p, K in K_sel.items():
        d = pass_dfs[p]
        if p not in KEY_PASSES and K is not None:
            d = d[d['rank'] < K]
        kept[p] = d
        keys.append(d.qi.to_numpy(np.int64) * n_pool + d.pi.to_numpy(np.int64))
        bits.append(np.full(len(d), PASS_BITS[p], np.uint16))
    keys, bits = np.concatenate(keys), np.concatenate(bits)
    uk, inv = np.unique(keys, return_inverse=True)
    ob = np.bincount(inv.ravel(), weights=bits, minlength=len(uk)).astype(np.uint16)   # distinct pass bits: sum == OR
    out = pd.DataFrame({'qi': (uk // n_pool).astype(np.int32), 'pi': (uk % n_pool).astype(np.int32), 'bits': ob})
    if scores:
        for p, d in kept.items():
            if p in KEY_PASSES:
                continue
            k = d.qi.to_numpy(np.int64) * n_pool + d.pi.to_numpy(np.int64)
            pos = np.searchsorted(uk, k)
            sc = np.full(len(uk), np.nan, np.float32); rk = np.full(len(uk), np.nan, np.float32)
            sc[pos] = d.score.to_numpy(np.float32); rk[pos] = d['rank'].to_numpy(np.float32) + 1
            out['s_' + p], out['r_' + p] = sc, rk
    return out
