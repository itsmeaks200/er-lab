"""Cached pipeline stages. A 'world' = (Q = query S1 records, P = pool of S2+S3 records, matrices, labels if train).
Stages and their cache keys:
  world (normalised frames + sparse matrices)   ← cfg.world, cfg.vec, maps
  pass_<X> (top-K candidates of one blocking pass) ← world key + cfg.retr/cfg.block (+ encoder/st keys)
  features (pair features of the candidate union)  ← passes + K + prune + feature version
  st1_* (stage-1 pruner scores + model)           ← features key + cfg.prune1
  feats1_* (matcher features of stage-1 survivors) ← st1 key + the cut
Everything lives in cfg.paths.cache_dir; only small reports go to cfg.paths.results_dir."""
import os
import gc
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import text as T
from . import data as D
from . import vec as V
from . import blocking as B
from . import features as FT
from . import prune as PR
from .utils import Timer, log, cfg_hash, save_json, load_json, STR

WORLD_COLS = V.TEXT_COLS + V.REC_COLS + ['num0']


def _cache(cfg, *parts):
    p = os.path.join(cfg['paths']['cache_dir'], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# ============================================================ worlds
def world_key(cfg, split):
    return f"{split}_" + cfg_hash(dict(w=cfg['world'], v=cfg['vec'], data=os.path.abspath(cfg['paths']['data_dir']),
                                       ver=3))


def _save_frame(df, path, cols):
    df[[c for c in cols if c in df]].to_parquet(path, index=False)


def _load_frame(path):
    df = pd.read_parquet(path)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(STR)
    return df


def save_world(W, d):
    os.makedirs(d, exist_ok=True)
    _save_frame(W['Q'], os.path.join(d, 'Q.parquet'), WORLD_COLS + ['s1_row'])
    _save_frame(W['P'], os.path.join(d, 'P.parquet'), WORLD_COLS + ['src'])
    for side in ('Qm', 'Pm'):
        for k, M in W[side].items():
            sp.save_npz(os.path.join(d, f'{side}_{k}.npz'), M, compressed=False)
    meta = {k: W[k] for k in ('split', 'NQ', 'NP', 'n_s1_total') if k in W}
    meta['maps'] = dict(NAME_MAP=T.NAME_MAP, ADDR_MAP=T.ADDR_MAP)
    save_json(meta, os.path.join(d, 'meta.json'))
    if 'gt_qi' in W:
        np.savez(os.path.join(d, 'labels.npz'), gt_qi=W['gt_qi'], gt_pi=W['gt_pi'], q_fold=W['q_fold'].astype('U4'),
                 n_true=W['n_true'], n_true_all_singleton_rate=np.array([W.get('singleton_rate', np.nan)]))
    open(os.path.join(d, 'DONE'), 'w').close()


def load_world(d, cfg):
    W = load_json(os.path.join(d, 'meta.json'))
    T.NAME_MAP.clear(); T.NAME_MAP.update(W['maps']['NAME_MAP'])
    T.ADDR_MAP.clear(); T.ADDR_MAP.update(W['maps']['ADDR_MAP'])
    W['Q'], W['P'] = _load_frame(os.path.join(d, 'Q.parquet')), _load_frame(os.path.join(d, 'P.parquet'))
    W['Qm'] = {k: sp.load_npz(os.path.join(d, f'Qm_{k}.npz')).tocsr() for k in V.MAT_SRC}
    W['Pm'] = {k: sp.load_npz(os.path.join(d, f'Pm_{k}.npz')).tocsr() for k in V.MAT_SRC}
    lp = os.path.join(d, 'labels.npz')
    if os.path.exists(lp):
        z = np.load(lp)
        W.update(gt_qi=z['gt_qi'], gt_pi=z['gt_pi'], q_fold=z['q_fold'].astype(object), n_true=z['n_true'],
                 singleton_rate=float(z['n_true_all_singleton_rate'][0]))
    return _finish_world(W, cfg, d)


def _finish_world(W, cfg, d):
    W['dir'] = d
    W['NQ'], W['NP'] = len(W['Q']), len(W['P'])
    W['q_ck'] = W['Q'].country_key.astype(object).to_numpy()
    W['p_slices'] = D.country_slices(W['P'])
    W['R'] = V.fit_idf(W['Pm'], cfg)
    if 'gt_qi' in W:
        W['gt_keys'] = W['gt_qi'].astype(np.int64) * W['NP'] + W['gt_pi'].astype(np.int64)
        W['gt_fold'] = W['q_fold'][W['gt_qi']]
        W['fold_q'] = {f: np.flatnonzero(W['q_fold'] == f) for f in ('fit', 'tune', 'hold')}
    return W


def build_train_world(cfg):
    key = world_key(cfg, 'train')
    d = _cache(cfg, key, 'x')[:-2]
    if os.path.exists(os.path.join(d, 'DONE')):
        log(f'world {key}: loading cache')
        return load_world(d, cfg)
    wc = cfg['world']
    paths = D.discover(cfg['paths']['data_dir'])
    with Timer('train world: load'):
        S1 = D.load_tsv(paths['train_s1'])
        S1['country_key'] = S1.country.str.strip().str.lower()
        P = D.build_pool(D.load_tsv(paths['train_s2']), D.load_tsv(paths['train_s3']))
        gt = D.ground_truth_pairs(D.load_tsv(paths['train_gt']))
        s1_idx = pd.Index(S1.entity_id.astype(object)).get_indexer(gt.s1_id)
        pi = pd.Index(P.entity_id.astype(object)).get_indexer(gt.o_id)
        ok = (s1_idx >= 0) & (pi >= 0)
        s1_idx, pi = s1_idx[ok], pi[ok]
        NS1 = len(S1)
        n_true_all = np.bincount(s1_idx, minlength=NS1)
        fold_all = D.entity_folds(S1.entity_id.astype(object).to_numpy(), wc['fold_pcts'])
        rng = np.random.RandomState(cfg['seed'])
        sample = np.sort(rng.choice(NS1, min(wc['n_s1_sample'], NS1), replace=False))
    if wc['pool'] == 'mini':
        with Timer('mini pool'):
            insample = np.zeros(NS1, bool); insample[sample] = True
            keep = np.zeros(len(P), bool)
            keep[pi[insample[s1_idx]]] = True
            keep[rng.rand(len(P)) < wc['mini_distractors'] / max(len(P), 1)] = True
            newpos = np.cumsum(keep) - 1
            P = P[keep].reset_index(drop=True)
            m = keep[pi]
            s1_idx, pi = s1_idx[m], newpos[pi[m]]
    with Timer('train world: basic normalisation + learned maps'):
        V.prepare_basic(S1); V.prepare_basic(P)
        T.NAME_MAP.clear(); T.ADDR_MAP.clear()
        fitpos = np.flatnonzero(fold_all[s1_idx] == 'fit')
        pick = rng.choice(fitpos, min(wc['map_pairs'], len(fitpos)), replace=False)
        lb, rb = S1.n_basic.take(s1_idx[pick]).tolist(), P.n_basic.take(pi[pick]).tolist()
        T.NAME_MAP.update(T.learn_token_map([T.name_core(x) for x in lb], [T.name_core(x) for x in rb],
                                            wc['map_min_count'], wc['map_min_prob']))
        hand = lambda s: ' '.join(T.ADDR_HAND_MAP.get(t, t) for t in s.split())
        la, ra = S1.a_basic.take(s1_idx[pick]).tolist(), P.a_basic.take(pi[pick]).tolist()
        T.ADDR_MAP.update(T.learn_token_map([hand(x) for x in la], [hand(x) for x in ra], wc['map_min_count'], wc['map_min_prob']))
        log(f'   learned NAME_MAP={len(T.NAME_MAP)} ADDR_MAP={len(T.ADDR_MAP)}; e.g. {list(T.ADDR_MAP.items())[:12]}')
    with Timer('train world: prepare frames'):
        V.prepare_frame(S1); V.prepare_frame(P); V.add_freq_features(S1, P)
    Q = S1.iloc[sample].reset_index(drop=True)
    Q['s1_row'] = sample
    s1_to_q = np.full(NS1, -1, np.int64); s1_to_q[sample] = np.arange(len(sample))
    m = s1_to_q[s1_idx] >= 0
    W = dict(split='train', Q=Q, P=P, gt_qi=s1_to_q[s1_idx[m]].astype(np.int32), gt_pi=pi[m].astype(np.int32),
             q_fold=fold_all[sample], n_true=np.bincount(s1_to_q[s1_idx[m]], minlength=len(sample)),
             n_s1_total=NS1, singleton_rate=float((n_true_all == 0).mean()))
    del S1
    gc.collect()
    W['Qm'] = V.build_matrices(Q, cfg, 'Q train'); W['Pm'] = V.build_matrices(P, cfg, 'pool train')
    save_world(W, d)
    return _finish_world(W, cfg, d)


def build_test_world(cfg, maps):
    key = world_key(cfg, 'test') + '_' + cfg_hash(maps)
    d = _cache(cfg, key, 'x')[:-2]
    T.NAME_MAP.clear(); T.NAME_MAP.update(maps['NAME_MAP'])
    T.ADDR_MAP.clear(); T.ADDR_MAP.update(maps['ADDR_MAP'])
    if os.path.exists(os.path.join(d, 'DONE')):
        log(f'world {key}: loading cache')
        return load_world(d, cfg)
    paths = D.discover(cfg['paths']['data_dir'])
    with Timer('test world: load + prepare'):
        Q = D.load_tsv(paths['test_s1'])
        Q['country_key'] = Q.country.str.strip().str.lower()
        P = D.build_pool(D.load_tsv(paths['test_s2']), D.load_tsv(paths['test_s3']))
        V.prepare_frame(Q); V.prepare_frame(P); V.add_freq_features(Q, P)
        Q['s1_row'] = np.arange(len(Q))
    W = dict(split='test', Q=Q, P=P)
    W['Qm'] = V.build_matrices(Q, cfg, 'Q test'); W['Pm'] = V.build_matrices(P, cfg, 'pool test')
    save_world(W, d)
    return _finish_world(W, cfg, d)


# ============================================================ dense embeddings
def encoder_key(cfg):
    return 'enc_' + cfg_hash(dict(e=cfg['encoder'], w=world_key(cfg, 'train')))


def get_hash_encoder(cfg, Wtr):
    import torch
    from . import encoders as E
    ec = cfg['encoder']
    path = _cache(cfg, 'encoders', encoder_key(cfg) + '.pt')
    model = E.build_hash_encoder(len(E.ENC_FIELDS) * 2 ** ec['bucket_bits'], ec['dim'])
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    fit = Wtr['q_fold'][Wtr['gt_qi']] == 'fit'
    with Timer('train hash encoder'):
        model = E.train_hash_encoder(Wtr['Qm'], Wtr['Pm'], Wtr['R'], Wtr['gt_qi'][fit], Wtr['gt_pi'][fit], ec, cfg['seed'])
    torch.save(model.state_dict(), path)
    return model


def get_embeddings(cfg, W, name, Wtr=None):
    """name 'embG' = hash encoder, 'embS' = sentence-transformer. Cached per world."""
    from . import encoders as E
    if name == 'embG':
        key = encoder_key(cfg)
    else:
        key = 'st_' + cfg_hash(dict(s=cfg['st'], w=world_key(cfg, 'train') if cfg['st']['finetune'] else ''))
    pq, pp = os.path.join(W['dir'], f'{name}_{key}_Q.npy'), os.path.join(W['dir'], f'{name}_{key}_P.npy')
    if os.path.exists(pq) and os.path.exists(pp):
        return np.load(pq), np.load(pp)
    with Timer(f'embeddings {name} on {W["split"]} world'):
        if name == 'embG':
            model = get_hash_encoder(cfg, Wtr if Wtr is not None else W)
            ec = cfg['encoder']
            Qe = E.embed_hash(model, W['Qm'], W['R'], W['NQ'], ec['bucket_bits'], ec['dim'])
            Pe = E.embed_hash(model, W['Pm'], W['R'], W['NP'], ec['bucket_bits'], ec['dim'])
        else:
            sc = cfg['st']
            model = get_st_model(cfg, Wtr if Wtr is not None else W)
            Qe = E.embed_st(model, E.st_texts(W['Q'], sc['prefix']), sc['batch'])
            Pe = E.embed_st(model, E.st_texts(W['P'], sc['prefix']), sc['batch'])
    np.save(pq, Qe); np.save(pp, Pe)
    return Qe, Pe


def get_st_model(cfg, Wtr):
    from . import encoders as E
    sc = cfg['st']
    model = E.load_st(sc['model'], sc['max_len'])
    if not sc['finetune']:
        return model
    path = _cache(cfg, 'encoders', 'st_ft_' + cfg_hash(dict(s=sc, w=world_key(cfg, 'train'))))
    if os.path.exists(os.path.join(path, 'DONE')):
        return E.load_st(path, sc['max_len'])
    fit = Wtr['q_fold'][Wtr['gt_qi']] == 'fit'
    qi, pi = Wtr['gt_qi'][fit], Wtr['gt_pi'][fit]
    rng = np.random.RandomState(cfg['seed'])
    order = rng.permutation(len(qi))
    _, first = np.unique(qi[order], return_index=True)       # one positive per S1 → no in-batch false negatives
    sel = order[first]
    qt = E.st_texts(Wtr['Q'].iloc[qi[sel]], sc['prefix'])
    pt = E.st_texts(Wtr['P'].iloc[pi[sel]], sc['prefix'])
    with Timer(f'fine-tune {sc["model"]} on {len(sel):,} pairs'):
        model = E.finetune_st(model, qt, pt, sc, cfg['seed'])
    model.save(path)
    open(os.path.join(path, 'DONE'), 'w').close()
    return model


# ============================================================ blocking
def pass_key(cfg, W, p):
    rel = dict(p=p, w=W['dir'], retr=cfg['retr'], k=cfg['block']['topk_max'], mb=cfg['block']['max_block'])
    if p == 'G':
        rel['enc'] = encoder_key(cfg)
    if p == 'S':
        rel['st'] = cfg['st']
    return f'pass_{p}_' + cfg_hash(rel)


def run_passes(cfg, W, passes, k=None, Wtr=None):
    k = k or cfg['block']['topk_max']
    out, todo = {}, []
    for p in passes:
        path = os.path.join(W['dir'], pass_key(cfg, W, p) + '.parquet')
        if os.path.exists(path):
            out[p] = pd.read_parquet(path)
        else:
            todo.append(p)
    allq = np.arange(W['NQ'])
    keycache = {}
    save = lambda ps: [out[p].to_parquet(os.path.join(W['dir'], pass_key(cfg, W, p) + '.parquet'), index=False) for p in ps]
    for p in [x for x in todo if x in B.KEY_PASSES]:        # each pass is cached as soon as it is done (restart-safe)
        with Timer(f'pass {p}'):
            out[p] = B.key_pass(p, W['Q'], W['P'], allq, cfg['block']['max_block'], keycache)
        save([p])
    sparse = [x for x in todo if x in B.SPARSE_PASSES]
    if sparse:
        with Timer(f'passes {sparse}'):
            out.update(B.sparse_passes(W['Qm'], W['Pm'], allq, W['q_ck'], W['p_slices'], W['R'], k, sparse, cfg))
        save(sparse)
    for p in [x for x in todo if x in B.DENSE_PASSES]:
        Qe, Pe = get_embeddings(cfg, W, 'embG' if p == 'G' else 'embS', Wtr)
        with Timer(f'pass {p}'):
            out[p] = B.dense_pass(Qe, Pe, allq, W['q_ck'], W['p_slices'], k)
        save([p])
    return out


def keys_of(d, NP):
    return np.unique(d.qi.to_numpy(np.int64) * NP + d.pi.to_numpy(np.int64))


def cand_stats(W, name, keys):
    NP, NQ = W['NP'], W['NQ']
    hit = np.isin(W['gt_keys'], keys)
    per_q = np.bincount((keys // NP).astype(np.int64), minlength=NQ)
    pool_c = {c: e - s for c, (s, e) in W['p_slices'].items()}
    total = float(sum(pool_c.get(c, NP) for c in W['q_ck']))
    r = dict(strategy=name, pairs=len(keys), recall=hit.mean())
    for f in ('fit', 'tune', 'hold'):
        r['recall_' + f] = hit[W['gt_fold'] == f].mean()
    r.update(avg_per_S1=per_q.mean(), p50_per_S1=float(np.median(per_q)), p95_per_S1=np.percentile(per_q, 95), max_per_S1=int(per_q.max()),
             S1_no_cand_pct=100 * (per_q == 0).mean(), reduction=1 - len(keys) / max(total, 1), missed=int((~hit).sum()))
    return r


def select_blocking(cfg, W, pass_dfs):
    """K per ranked pass from recall@K; passes chosen greedily on FIT-fold recall (labels only measure recall)."""
    bc, NP = cfg['block'], W['NP']
    rows, curves, K_sel = [], {}, {}
    for p, d in pass_dfs.items():
        rows.append(cand_stats(W, B.PASS_NAMES[p], keys_of(d, NP)))
        if p in B.KEY_PASSES:
            K_sel[p] = None
            continue
        grid = [K for K in bc['k_grid'] if K <= bc['topk_max']]
        curves[p] = [np.isin(W['gt_keys'], keys_of(d[d['rank'] < K], NP)).mean() for K in grid]
        if p in bc['k']:
            K_sel[p] = bc['k'][p]
        else:
            tgt = bc['k_keep'] * curves[p][-1]
            K_sel[p] = next(K for K, r in zip(grid, curves[p]) if r >= tgt)
    table = pd.DataFrame(rows).set_index('strategy')
    curve_df = pd.DataFrame(curves, index=[K for K in bc['k_grid'] if K <= bc['topk_max']])
    is_fit_q = W['q_fold'] == 'fit'
    fit_gt = W['gt_fold'] == 'fit'
    pk = {}
    for p, d in pass_dfs.items():
        dd = d if K_sel[p] is None else d[d['rank'] < K_sel[p]]
        k = keys_of(dd, NP)
        pk[p] = k[is_fit_q[(k // NP).astype(np.int64)]]
    if bc['select'] == 'all':
        selected, hist = list(pass_dfs), []
    else:
        selected, cur, cur_rec, hist = [], np.empty(0, np.int64), 0.0, []
        while True:
            best = None
            for p in pk:
                if p in selected:
                    continue
                u = np.union1d(cur, pk[p])
                r = np.isin(W['gt_keys'][fit_gt], u).mean()
                if best is None or r > best[1]:
                    best = (p, r, u)
            if best is None or best[1] - cur_rec < bc['greedy_min_gain']:
                break
            selected.append(best[0]); cur, cur_rec = best[2], best[1]
            hist.append(dict(step=len(selected), added=B.PASS_NAMES[best[0]], fit_recall=round(cur_rec, 5),
                             pairs_per_S1=round(len(cur) / max(is_fit_q.sum(), 1), 1)))
    return dict(K={p: K_sel[p] for p in selected}, passes=selected, table=table, curves=curve_df, greedy=pd.DataFrame(hist))


# ============================================================ features
def feat_key(cfg, W, plan):
    return 'feats_' + cfg_hash(dict(w=W['dir'], passes=plan['passes'], K=plan['K'], retr=cfg['retr'],
                                    enc=encoder_key(cfg) if 'embG' in plan['emb'] else '',
                                    st=cfg['st'] if 'embS' in plan['emb'] else '', v=cfg['feats']['version'],
                                    pk={p: pass_key(cfg, W, p) for p in plan['passes']}))


def get_features(cfg, W, pass_dfs, plan, Wtr=None):
    """Candidate union + pair features (cached as float32 .npy + columns json + candidate parquet)."""
    key = feat_key(cfg, W, plan)
    fx, fc, fcand = (os.path.join(W['dir'], key + s) for s in ('.npy', '.cols.json', '.cand.parquet'))
    if os.path.exists(fx):
        X = np.load(fx, mmap_mode='r')
        cols = load_json(fc)
        cand = pd.read_parquet(fcand)
        return cand, pd.DataFrame(np.asarray(X), columns=cols)
    cand = B.union_candidates(pass_dfs, plan['K'], W['NP'])
    emb = {n: get_embeddings(cfg, W, n, Wtr) for n in plan['emb']}
    with Timer(f'features for {len(cand):,} candidate pairs'):
        F, _ = FT.compute_features(cand, W, plan['passes'], emb, prune=False, chunk=cfg['feats']['chunk'],
                                   alpha=cfg['retr']['alpha_name'])
    np.save(fx, F.to_numpy(np.float32)); save_json(list(F.columns), fc); cand.to_parquet(fcand, index=False)
    return cand, F


def label_candidates(W, cand):
    keys = cand.qi.to_numpy(np.int64) * W['NP'] + cand.pi.to_numpy(np.int64)
    return np.isin(keys, W['gt_keys']).astype(np.int8)


def apply_prune(cand, F, y, plan):
    keep = ~FT.safe_zone(F)
    cand, F, y = cand[keep].reset_index(drop=True), F[keep].reset_index(drop=True), y[keep]
    F = FT.drop_context(F)
    FT.add_context(F, cand.qi.to_numpy(), cand.bits.to_numpy(), plan['passes'])
    return cand, F, y


# ============================================================ stage 1: learned blocking (erlab/prune.py)
def stage1_key(cfg, W, plan):
    pc = {k: v for k, v in cfg['prune1'].items() if k not in ('max_oracle_loss', 't_grid', 'n_grid', 'm_grid', 'enabled')}
    return 'st1_' + cfg_hash(dict(f=feat_key(cfg, W, plan), pc=pc, seed=cfg['seed'], v=1))


def run_stage1(cfg, W, pdfs, plan, Wtr=None):
    """Train world: stage-0 union → cheap features → pruner (out-of-fold on fit) → per-S1 / per-pool ranks. Cached."""
    pc = cfg['prune1']
    base = os.path.join(W['dir'], stage1_key(cfg, W, plan))
    cand = B.union_candidates(pdfs, plan['K'], W['NP'])
    y = label_candidates(W, cand)
    qi, pi = cand.qi.to_numpy(), cand.pi.to_numpy()
    if os.path.exists(base + '.DONE'):
        p, m = np.load(base + '.p.npy'), PR.load_pruner(base + '.lgb')
        cols, imp = load_json(base + '.cols.json'), load_json(base + '.imp.json')
    else:
        emb = {n: get_embeddings(cfg, W, n, Wtr) for n in plan['emb']}
        with Timer(f'stage-1 cheap features for {len(cand):,} stage-0 pairs'):
            X, cols = PR.cheap_features(cand, W, emb, plan['passes'], cfg['retr']['alpha_name'], pc['pool_ctx'])
        fold = W['q_fold'][qi]
        rows = {f: np.flatnonzero(fold == f) for f in ('fit', 'tune', 'hold')}
        q_hash = (pd.util.hash_array(W['Q'].entity_id.astype(object).to_numpy()) % 2).astype(np.int64)
        p, m = PR.fit_pruner(X, y, qi, rows, q_hash, pc, cfg['seed'])
        del X
        imp = dict(zip(cols, map(float, m.importance())))
        np.save(base + '.p.npy', p); PR.save_pruner(m, base + '.lgb')
        save_json(cols, base + '.cols.json'); save_json(imp, base + '.imp.json')
        open(base + '.DONE', 'w').close()
    qr, pr = PR.ranks_of(p, qi, pi, pc['pool_ctx'])
    return dict(key=os.path.basename(base), cand=cand, y=y, p=p, qr=qr, pr=pr, model=m, cols=cols, imp=imp)


def stage1_features(cfg, W, plan, S1, keep, tag, Wtr=None):
    """Matcher features (incl. pr_p) for the stage-1 survivors `keep` (bool mask over the stage-0 union). Cached."""
    key = 'feats1_' + cfg_hash(dict(s=S1['key'], tag=tag, n=int(keep.sum())))
    fx, fc, fcand = (os.path.join(W['dir'], key + s) for s in ('.npy', '.cols.json', '.cand.parquet'))
    if os.path.exists(fx):
        return pd.read_parquet(fcand), pd.DataFrame(np.asarray(np.load(fx, mmap_mode='r')), columns=load_json(fc))
    cand = S1['cand'][keep].reset_index(drop=True)
    emb = {n: get_embeddings(cfg, W, n, Wtr) for n in plan['emb']}
    with Timer(f'matcher features for {len(cand):,} stage-1 survivors'):
        F, _ = FT.compute_features(cand, W, plan['passes'], emb, prune=False, chunk=cfg['feats']['chunk'],
                                   alpha=cfg['retr']['alpha_name'], extra={'pr_p': S1['p'][keep]})
    np.save(fx, F.to_numpy(np.float32)); save_json(list(F.columns), fc); cand.to_parquet(fcand, index=False)
    return cand, F


def subset_context(cand, F, sub, passes):
    """Rows `sub` of a survivor set; recomputes the within-S1 context features on the smaller set (as at inference)."""
    cand, F = cand.iloc[sub].reset_index(drop=True), FT.drop_context(F.iloc[sub].reset_index(drop=True))
    FT.add_context(F, cand.qi.to_numpy(), cand.bits.to_numpy(), passes)
    return cand, F
