"""Approach pipelines: holdout evaluation → refit on all labelled rows → test inference → submission files + validation
(own checks + official validator). Each approach writes to results/submissions/<EXP>/."""
import os
import sys
import time
import subprocess

import numpy as np
import pandas as pd

from . import pipeline as PL
from . import features as FT
from . import evaluate as EV
from . import prune as PR
from .models import Model
from .utils import Timer, log, save_json, load_json, cfg_hash


def write_id_lists(path, header, q_ids, qi_sorted, pi_sorted, pool_ids):
    n = len(q_ids)
    starts = np.searchsorted(qi_sorted, np.arange(n), side='left')
    ends = np.searchsorted(qi_sorted, np.arange(n), side='right')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\t'.join(header) + '\n')
        for b in range(0, n, 200_000):
            lines = [q_ids[q] + '\t' + ','.join(pool_ids[pi_sorted[starts[q]:ends[q]]]) for q in range(b, min(n, b + 200_000))]
            fh.write('\n'.join(lines) + '\n')


def validate_files(match_path, cand_path, s1_ids, pool_ids):
    s1_set, pool_set = set(s1_ids), set(pool_ids)
    rep, M = {}, {}
    bad = dup_list = unknown = n_m = dup_m = 0
    with open(match_path, encoding='utf-8') as fh:
        rep['matching header'] = fh.readline().rstrip('\n').split('\t') == ['source1_entity_id', 'matched_entity_ids']
        for line in fh:
            if not line.strip():
                continue
            s1, _, rest = line.rstrip('\n').partition('\t')
            ids = rest.split(',') if rest else []
            n_m += 1; dup_m += s1 in M; M[s1] = ids
            dup_list += len(ids) != len(set(ids))
            for i in ids:
                if not i.startswith(('S2-', 'S3-')):
                    bad += 1
                elif i not in pool_set:
                    unknown += 1
    seen, n_c, dup_c, cbad, cdup, notsub, tot = set(), 0, 0, 0, 0, 0, 0
    with open(cand_path, encoding='utf-8') as fh:
        rep['candidate header'] = fh.readline().rstrip('\n').split('\t') == ['source1_entity_id', 'candidate_entity_ids']
        for line in fh:
            if not line.strip():
                continue
            s1, _, rest = line.rstrip('\n').partition('\t')
            ids = rest.split(',') if rest else []
            n_c += 1; dup_c += s1 in seen; seen.add(s1)
            cs = set(ids)
            cdup += len(ids) != len(cs)
            cbad += sum(1 for i in ids if i not in pool_set)
            notsub += bool(set(M.get(s1, [])) - cs)
            tot += len(ids)
    rep['one row per test S1 (matching)'] = set(M) == s1_set and n_m == len(s1_set) and dup_m == 0
    rep['one row per test S1 (candidates)'] = seen == s1_set and n_c == len(s1_set) and dup_c == 0
    rep['only S2/S3 ids'] = bad == 0
    rep['no duplicate ids in a list'] = dup_list == 0 and cdup == 0
    rep['matched ids exist'] = unknown == 0
    rep['candidate ids exist'] = cbad == 0
    rep['matches subset of candidates'] = notsub == 0
    stats = dict(n_S1=len(s1_set), with_matches=sum(1 for v in M.values() if v), singletons=sum(1 for v in M.values() if not v),
                 total_matches=sum(len(v) for v in M.values()), total_candidates=tot)
    return all(rep.values()), rep, stats


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST = {}


def refit_rounds(best_iter, mult=1.15, lo=50):
    """Rounds for the refit on ~2x more rows than the holdout-phase fit fold."""
    return max(lo, int(round((best_iter or 0) * mult)))


# ============================================================ holdout reporting
def _country_table(ctx, qi_sel, y_sel, q_eval):
    ck, rows = ctx.W['q_ck'], []
    for c in np.unique(ck[q_eval]):
        qe = q_eval[ck[q_eval] == c]
        mm = np.isin(qi_sel, qe)
        rows.append(dict(country=c, **EV.s1_metrics(qi_sel[mm], y_sel[mm], ctx.W['n_true'], qe)))
    return pd.DataFrame(rows).set_index('country')


def holdout_report(ctx, rep, res, rule, ph, out, extra):
    W, rh = ctx.W, ctx.rows['hold']
    prm = res[rule]['prm']
    phd = EV.apply_calibrator(res[rule]['cal'], ph) if rule == 'expf' else ph
    sel = EV.apply_rule(prm, ctx.qi[rh], ctx.pi[rh], phd, ctx.s3[rh], W['NQ'])
    q_eval = W['fold_q']['hold']
    m = res[rule]['hold']
    cand_per = np.bincount(ctx.qi[rh], minlength=W['NQ'])[q_eval]
    summ = dict(hold_F05=m['F05'], hold_P_micro=m['P_micro'], hold_R_micro=m['R_micro'], hold_P_macro=m['P_macro'],
                hold_R_macro=m['R_macro'], singleton_acc=m['singleton_acc'], false_merge_rate=m['false_merge_rate'],
                avg_pred=m['avg_pred'], tune_F05=res[rule]['tune'], oracle_hold=ctx.oracle['hold'], auc=res['auc'],
                hold_cand_per_S1=float(cand_per.mean()), n_hold_S1=int(len(q_eval)), rule=rule, rule_prm=str(prm), **extra)
    rep.table(f'HOLDOUT {rep.exp_id} (hold S1 never used for training, early stopping or rule tuning)',
              pd.Series({k: (round(v, 5) if isinstance(v, float) else v) for k, v in summ.items()}, name='value').to_frame(),
              short=True, short_rows=30)
    rep.table('holdout F0.5 by decision rule (rule picked on tune)',
              pd.DataFrame([dict(rule=k, tune_F05=res[k]['tune'], hold_F05=res[k]['hold']['F05'],
                                 hold_P=res[k]['hold']['P_micro'], hold_R=res[k]['hold']['R_micro'])
                            for k in ('global', 'staged', 'expf') if k in res]).set_index('rule'), short=True)
    per = _country_table(ctx, ctx.qi[rh][sel], ctx.y[rh][sel], q_eval)
    rep.table('holdout per country', per, short=True)
    nt = W['n_true'][q_eval]
    npred = np.bincount(ctx.qi[rh][sel], minlength=W['NQ'])[q_eval]
    rep.table('holdout: true (rows) vs predicted (cols) matches per S1', pd.crosstab(np.minimum(nt, 8), np.minimum(npred, 8)))
    save_json(dict(summary=summ, per_country=per.reset_index().to_dict('records')), os.path.join(out, 'holdout.json'))
    return summ


# ============================================================ stage 2 / cross-encoder helpers
def ce_name(j):
    return 'ce' if j == 0 else f'ce{j + 1}'


def stage2_matrix(ctx, p1, keep, ces, rows):
    S = FT.stage2_frame(ctx.qi[rows], ctx.pi[rows], ctx.s3[rows].astype(np.int64), p1[rows], ctx.W['P'], ctx.W['Pm'],
                        ctx.F[keep].iloc[rows])
    for j, ce in enumerate(ces or []):
        S[ce_name(j)] = ce[rows]
    return S


def oof_fixed(ctx, mc, feats, rows, rounds, X_of):
    """2-fold (by S1 hash) out-of-fold scores on `rows` with a fixed number of rounds."""
    h = (pd.util.hash_array(ctx.W['Q'].entity_id.astype(object).to_numpy()) % 2).astype(np.int64)[ctx.qi[rows]]
    p = np.empty(len(rows), np.float32)
    for k in (0, 1):
        a, b = rows[h != k], rows[h == k]
        m = Model(mc['kind'], mc['params'], rounds, 10 ** 9, ctx.cfg['seed'], feats).fit_fixed(X_of(ctx, a, feats), ctx.y[a], ctx.qi[a])
        p[h == k] = m.predict(X_of(ctx, b, feats))
    return p


def crossenc_scores(ctx, cc, rep):
    """Cross-encoder trained on the 'enc' fold only (the matcher never trains on it → no in-sample scores). Rows to
    score are chosen by the stage-1 probability pr_p (fixed across the holdout, refit and test phases)."""
    from . import crossenc as CE
    if 'pr_p' not in ctx.colidx:
        raise ValueError('cross-encoder approaches need stage-1 learned blocking (prune1.enabled=True)')
    enc = np.flatnonzero(ctx.fold == 'enc')
    if not len(enc):
        raise ValueError("cross-encoder approaches need the 'enc' fold (world.enc_frac > 0)")
    W, y, pr = ctx.W, ctx.y, ctx.F['pr_p'].to_numpy(np.float32)
    rng = np.random.RandomState(ctx.cfg['seed'])
    pos, neg = enc[y[enc] == 1], enc[y[enc] == 0]
    rk = pd.Series(pr[neg]).groupby(ctx.qi[neg]).rank(ascending=False, method='first').to_numpy()
    hard = neg[rk <= 4]
    tr = np.r_[pos, rng.choice(hard, min(len(hard), cc['neg_per_pos'] * len(pos)), replace=False)]
    if len(tr) > cc['max_train_pairs']:
        tr = rng.choice(tr, cc['max_train_pairs'], replace=False)
    with Timer(f"cross-encoder {cc['model']} train on {len(tr):,} enc-fold pairs"):
        model, tok = CE.train_crossenc(cc, CE.serialise(W['Q'], ctx.qi[tr]), CE.serialise(W['P'], ctx.pi[tr]), y[tr], ctx.cfg['seed'])
    scope = np.flatnonzero(ctx.fold != 'enc')
    sel = scope[CE.select_rows(ctx.qi[scope], pr[scope], cc['band'], cc['top_n'], cc.get('max_pred_pairs', 0))]
    ce = np.full(len(y), np.nan, np.float32)
    with Timer(f'cross-encoder score {len(sel):,} train-world pairs'):
        ce[sel] = CE.predict_crossenc(model, tok, CE.serialise(W['Q'], ctx.qi[sel]), CE.serialise(W['P'], ctx.pi[sel]), cc)
    from sklearn.metrics import roc_auc_score
    h = np.intersect1d(sel, ctx.rows['hold'])
    if len(h) and 0 < y[h].mean() < 1:
        rep.text(f"cross-encoder {cc['model']}: {len(tr):,} training pairs (enc fold) | scored {len(sel):,} pairs | "
                 f"hold AUC {roc_auc_score(y[h], ce[h]):.5f} vs stage-1 pruner {roc_auc_score(y[h], pr[h]):.5f} (n={len(h):,})", short=True)
    return ce, dict(model=model, tok=tok, n_scope_q=len(np.unique(ctx.qi[scope])), n_sel=len(sel))


# ============================================================ test world (cached, shared by all approaches)
def test_world(cfg, ctx):
    if 'W' not in _TEST:
        from . import text as T
        maps = dict(NAME_MAP=dict(T.NAME_MAP), ADDR_MAP=dict(T.ADDR_MAP))
        _TEST['W'] = PL.build_test_world(cfg, maps)
    return _TEST['W']


def test_candidates_and_features(cfg, ctx, rep):
    """Stage 0 + stage 1 (+ safe zone when stage 1 is off) + matcher features on the test world. Cached on disk and in
    memory, so every approach scores exactly the same candidate set."""
    from . import blocking as B
    plan, Wtr = ctx.plan, ctx.W
    Wte = test_world(cfg, ctx)
    st = ctx.stage1
    key = 'testset_' + cfg_hash(dict(w=Wte['dir'], passes=plan['passes'], K=plan['K'], emb=plan['emb'], prune=plan.get('prune'),
                                     st1=st['key'] if st else None, cut=st['cut'] if st else None, v=cfg['feats']['version'],
                                     fe2=PL.fe2_on(cfg), pool=PL.pool_feats_on(cfg),
                                     enc=PL.encoder_key(cfg), pk={p: PL.pass_key(cfg, Wte, p) for p in plan['passes']}))
    if _TEST.get('key') == key:
        return Wte, _TEST['cand'], _TEST['F'], _TEST['crows']
    base = os.path.join(Wte['dir'], key)
    if os.path.exists(base + '.DONE'):
        log('test candidates + features: loading cache')
        cand = pd.read_parquet(base + '.cand.parquet')
        F = pd.DataFrame(np.asarray(np.load(base + '.npy', mmap_mode='r')), columns=load_json(base + '.cols.json'))
        crows = load_json(base + '.crows.json')
    else:
        pdfs = PL.run_passes(cfg, Wte, plan['passes'], Wtr=Wtr)
        cand = B.union_candidates(pdfs, plan['K'], Wte['NP'])
        del pdfs
        emb = {n: PL.get_embeddings(cfg, Wte, n, Wtr) for n in plan['emb']}
        per_q = lambda c: np.bincount(c.qi.to_numpy(), minlength=Wte['NQ'])
        crow = lambda name, c: dict(stage=name, pairs=len(c), avg_per_S1=float(per_q(c).mean()), p50=float(np.median(per_q(c))),
                                    p95=float(np.percentile(per_q(c), 95)), max=int(per_q(c).max()),
                                    S1_empty_pct=float(100 * (per_q(c) == 0).mean()))
        crows = [crow('stage 0 retrieval union', cand)]
        extra = None
        if st:
            with Timer(f'test stage-1 pruner on {len(cand):,} pairs'):
                X1, cols1 = PR.cheap_features(cand, Wte, emb, plan['passes'], cfg['retr']['alpha_name'], cfg['prune1']['pool_ctx'])
                assert cols1 == st['cols'], 'stage-1 feature columns differ between train and test'
                p1s = PR.predict_rows(st['model'], X1, np.arange(len(cand)))
                del X1
                qr, pr = PR.ranks_of(p1s, cand.qi.to_numpy(), cand.pi.to_numpy(), cfg['prune1']['pool_ctx'])
                keep1 = PR.cut_mask(p1s, qr, pr, st['cut'])
                cand, extra = cand[keep1].reset_index(drop=True), {'pr_p': p1s[keep1]}
            crows.append(crow(f"stage 1 pruner {st['cut']}", cand))
        with Timer(f'test matcher features for {len(cand):,} pairs'):
            F, keepm = FT.compute_features(cand, Wte, plan['passes'], emb, prune=plan.get('prune', False),
                                           chunk=cfg['feats']['chunk'], alpha=cfg['retr']['alpha_name'], extra=extra,
                                           fe2=PL.fe2_on(cfg))
        if not keepm.all():
            cand = cand[keepm].reset_index(drop=True)
            crows.append(crow('after safe-zone pruning', cand))
        if PL.pool_feats_on(cfg):
            FT.add_pool_context(F, cand.pi.to_numpy())
        np.save(base + '.npy', F.to_numpy(np.float32)); save_json(list(F.columns), base + '.cols.json')
        cand.to_parquet(base + '.cand.parquet', index=False); save_json(crows, base + '.crows.json')
        open(base + '.DONE', 'w').close()
    _TEST.update(key=key, cand=cand, F=F, crows=crows)
    return Wte, cand, F, crows


def crossenc_both(cfg, ctx, cc, rep, Wte, cand_te, F_te):
    """Train-world and test scores of one cross-encoder / LLM, cached on disk (the most expensive step; approaches that
    share a text model reuse it). The model is freed right after scoring so the next, larger one fits on the GPU."""
    key = cfg_hash(dict(st=ctx.stage1['key'] if ctx.stage1 else None, cut=ctx.stage1['cut'] if ctx.stage1 else None, cc=cc,
                        ef=cfg['world'].get('enc_frac', 0), seed=cfg['seed'], n=len(ctx.y), test=Wte['dir'], nt=len(cand_te), v=1))
    path = os.path.join(cfg['paths']['cache_dir'], 'crossenc', f'ce_{key}.npz')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        z = np.load(path)
        ce, ce_t = z['ce'], z['ce_t']
        log(f"cross-encoder {cc['model']}: loaded cached scores")
    else:
        ce, obj = crossenc_scores(ctx, cc, rep)
        ce_t = crossenc_test(obj, cc, Wte, cand_te, F_te)
        np.savez(path, ce=ce, ce_t=ce_t)
        del obj
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
    from sklearn.metrics import roc_auc_score
    h = ctx.rows['hold'][~np.isnan(ce[ctx.rows['hold']])]
    auc = roc_auc_score(ctx.y[h], ce[h]) if len(h) and 0 < ctx.y[h].mean() < 1 else float('nan')
    rep.text(f"text model {cc['model']} ({cc.get('arch', 'encoder')}): hold AUC {auc:.5f} on {len(h):,} scored hold pairs | "
             f"test pairs scored {int((~np.isnan(ce_t)).sum()):,}", short=True)
    return ce, ce_t


def crossenc_test(ce_obj, cc, Wte, cand, F):
    from . import crossenc as CE
    qi, pi = cand.qi.to_numpy(), cand.pi.to_numpy()
    pr = F['pr_p'].to_numpy(np.float32)
    cap = cc.get('max_pred_pairs', 0)
    cap = int(cap * Wte['NQ'] / max(ce_obj['n_scope_q'], 1)) if cap else 0      # same share of S1 as in training
    sel = CE.select_rows(qi, pr, cc['band'], cc['top_n'], cap)
    ce = np.full(len(qi), np.nan, np.float32)
    with Timer(f'cross-encoder score {len(sel):,} test pairs'):
        ce[sel] = CE.predict_crossenc(ce_obj['model'], ce_obj['tok'], CE.serialise(Wte['Q'], qi[sel]), CE.serialise(Wte['P'], pi[sel]), cc)
    return ce


# ============================================================ one approach = holdout → refit → test
def submit(cfg, rep, args, build_ctx, feature_list, X_of, cap_rows, evaluate_scores, oof_p1):
    """Each approach writes to results/submissions/<EXP>/ (holdout.json, matching_results.tsv, candidate_pairs.tsv, meta).
    1 HOLDOUT: train on the fit fold, early stopping + decision rule on tune, score hold once.
    2 REFIT: retrain on fit+tune+hold (every labelled row except the 'enc' fold, which trained the encoders; using it
      would leak their memorised pairs) with the holdout-phase round counts x1.15; decision rule unchanged.
    3 TEST: shared cached test candidates/features → score → rule → write → validate (own + official validator).
    args: models=[{kind, params}] averaged | stage2 (cross-source reranker) | crossenc (needs stage 1) | rule."""
    t0 = time.time()
    ctx = build_ctx(cfg, rep)
    W, y, mcfg = ctx.W, ctx.y, cfg['model']
    out = os.path.join(cfg['paths']['results_dir'], 'submissions', rep.exp_id)
    os.makedirs(out, exist_ok=True)
    feats = feature_list(ctx.F, mcfg['feature_drop'])
    specs = [dict(s) for s in args.get('models', [dict(kind=mcfg['kind'], params=mcfg['params'])])]
    for s in specs:
        pf = s.pop('params_file', None)
        if pf:
            pf = os.path.join(cfg['paths']['results_dir'], pf)
            if os.path.exists(pf):
                s['params'] = dict(s.get('params', {}), **load_json(pf))
                rep.text(f'using tuned params from {pf}: {s["params"]}', short=True)
            else:
                rep.text(f'{pf} not found → default params', short=True)
    use_s2, use_ce = bool(args.get('stage2')), bool(args.get('crossenc'))
    from .config import deep_merge
    ce_cfgs = ([cfg['crossenc']] if use_ce else []) + [deep_merge(cfg['crossenc'], ex) for ex in args.get('extra_crossenc', [])]
    if ce_cfgs and not use_s2:
        raise ValueError('cross-encoder scores are stacked by the stage-2 model: set stage2=True')
    rf, rt, rh = ctx.rows['fit'], ctx.rows['tune'], ctx.rows['hold']
    all_rows = np.sort(np.r_[rf, rt, rh])
    rep.text(f"approach {rep.exp_id}: models={[s['kind'] for s in specs]} stage2={use_s2} "
             f"text models={[c['model'] for c in ce_cfgs]} | "
             f"{len(feats)} features | rows fit {len(rf):,} / tune {len(rt):,} / hold {len(rh):,} / enc (excluded) "
             f"{int((ctx.fold == 'enc').sum()):,} | train S1 {W['NQ']:,}", short=True)
    ces, ces_t = [], []
    for cc in ce_cfgs:
        Wte, cand_te, F_te, _ = test_candidates_and_features(cfg, ctx, rep)
        ce, ce_t = crossenc_both(cfg, ctx, cc, rep, Wte, cand_te, F_te)
        ces.append(ce); ces_t.append(ce_t)

    # ---------------- phase 1: holdout
    t1 = time.time()
    with Timer(f'{rep.exp_id} phase 1: holdout models'):
        if use_s2:
            mc = dict(mcfg, **specs[0])
            p1, m1 = oof_p1(ctx, mc, feats)
            imp = m1.importance()
            keep = (list(pd.Series(imp, index=feats).sort_values(ascending=False).index[:cfg['stage2']['keep_feats']])
                    if imp is not None else feats[:cfg['stage2']['keep_feats']])
            S2 = stage2_matrix(ctx, p1, keep, ces, np.arange(len(y)))
            s2cols = list(S2.columns)
            X2 = lambda r: S2.iloc[r].to_numpy(np.float32)
            tr = cap_rows(ctx, rf, mcfg['max_train_rows'])
            m2 = Model('lgb', dict(num_leaves=63), mcfg['rounds'], mcfg['early_stop'], cfg['seed'], s2cols)
            m2.fit(X2(tr), y[tr], None, X2(rt), y[rt], None)
            pt, ph = m2.predict(X2(rt)), m2.predict(X2(rh))
            r1 = evaluate_scores(ctx, p1[rt], p1[rh])
            rep.text(f"stage-1 matcher alone: hold F0.5 {r1[r1['best']]['hold']['F05']:.5f} (rule {r1['best']})", short=True)
            iters = dict(m1=int(m1.best_iter), m2=int(m2.best_iter))
            del S2
        else:
            tr = cap_rows(ctx, rf, mcfg['max_train_rows'])
            models_h, member_rows, P_t, P_h = [], [], [], []
            for s in specs:
                m = Model(s['kind'], s.get('params', {}), mcfg['rounds'], mcfg['early_stop'], cfg['seed'], feats)
                with Timer(f"  holdout model {s['kind']}"):
                    m.fit(X_of(ctx, tr, feats), y[tr], ctx.qi[tr], X_of(ctx, rt, feats), y[rt], ctx.qi[rt])
                models_h.append(m)
                P_t.append(m.predict(X_of(ctx, rt, feats)).astype(np.float32)); P_h.append(m.predict(X_of(ctx, rh, feats)).astype(np.float32))
                if len(specs) > 1:
                    r = evaluate_scores(ctx, P_t[-1], P_h[-1])
                    member_rows.append(dict(member=s['kind'], hold_F05=r[r['best']]['hold']['F05'], tune_F05=r[r['best']]['tune'],
                                            rule=r['best'], best_iter=int(getattr(m, 'best_iter', 0))))
            pt, ph = np.mean(P_t, axis=0), np.mean(P_h, axis=0)
            if member_rows:
                rep.table('ensemble members alone (holdout)', pd.DataFrame(member_rows).set_index('member'), short=True)
            iters = {f"{s['kind']}#{j}": int(getattr(m, 'best_iter', 0)) for j, (s, m) in enumerate(zip(specs, models_h))}
    res = evaluate_scores(ctx, pt.astype(np.float32), ph.astype(np.float32))
    rule = args.get('rule', 'best')
    rule = res['best'] if rule == 'best' else rule
    prm, cal = res[rule]['prm'], res[rule].get('cal')
    summ = holdout_report(ctx, rep, res, rule, ph, out, dict(iters=str(iters), minutes_holdout=round((time.time() - t1) / 60, 1)))
    rep.metric(hold_F05=summ['hold_F05'], hold_P=summ['hold_P_micro'], hold_R=summ['hold_R_micro'], tune_F05=summ['tune_F05'],
               rule=rule, oracle_hold=summ['oracle_hold'])

    # ---------------- phase 2: refit on every labelled non-enc row
    t2 = time.time()
    tr_all = cap_rows(ctx, all_rows, mcfg['max_train_rows'])
    with Timer(f'{rep.exp_id} phase 2: refit on {len(tr_all):,} rows (fit+tune+hold)'):
        if use_s2:
            R1 = refit_rounds(m1.best_iter)
            m1f = Model(mc['kind'], mc['params'], R1, 10 ** 9, cfg['seed'], feats).fit_fixed(X_of(ctx, tr_all, feats), y[tr_all], ctx.qi[tr_all])
            p1r = np.zeros(len(y), np.float32)
            p1r[tr_all] = oof_fixed(ctx, mc, feats, tr_all, R1, X_of)
            S2r = stage2_matrix(ctx, p1r, keep, ces, tr_all)
            m2f = Model('lgb', dict(num_leaves=63), refit_rounds(m2.best_iter), 10 ** 9, cfg['seed'], s2cols)
            m2f.fit_fixed(S2r[s2cols].to_numpy(np.float32), y[tr_all])
            del S2r
            refit_desc = dict(m1_rounds=R1, m2_rounds=int(m2f.rounds))
        else:
            final = []
            for s, m in zip(specs, models_h):
                mf = Model(s['kind'], s.get('params', {}), refit_rounds(getattr(m, 'best_iter', 0)), 10 ** 9, cfg['seed'], feats)
                with Timer(f"  refit {s['kind']} ({mf.rounds} rounds)"):
                    mf.fit_fixed(X_of(ctx, tr_all, feats), y[tr_all], ctx.qi[tr_all])
                final.append(mf)
            refit_desc = {f"{s['kind']}#{j}": int(mf.rounds) for j, (s, mf) in enumerate(zip(specs, final))}
    rep.text(f'refit on {len(tr_all):,} labelled rows ({len(np.unique(ctx.qi[tr_all])):,} S1) with rounds {refit_desc} '
             f'[{(time.time() - t2) / 60:.1f} min]', short=True)

    # ---------------- phase 3: test
    t3 = time.time()
    Wte, cand, F, crows = test_candidates_and_features(cfg, ctx, rep)
    rep.table('test candidates per S1 (last row = candidate_pairs.tsv)', pd.DataFrame(crows).set_index('stage'), short=True)
    qi, pi = cand.qi.to_numpy(), cand.pi.to_numpy()
    s3b = Wte['P'].src.to_numpy()[pi] == 3
    with Timer(f'{rep.exp_id} phase 3: score {len(cand):,} test pairs'):
        if use_s2:
            p1t = m1f.predict(F[feats].to_numpy(np.float32)).astype(np.float32)
            S2t = FT.stage2_frame(qi, pi, s3b.astype(np.int64), p1t, Wte['P'], Wte['Pm'], F[keep])
            for j, ce_t in enumerate(ces_t):
                S2t[ce_name(j)] = ce_t
            PT = m2f.predict(S2t[s2cols].to_numpy(np.float32)).astype(np.float32)
            del S2t
        else:
            Xt = F[feats].to_numpy(np.float32)
            PT = np.mean([m.predict(Xt) for m in final], axis=0).astype(np.float32)
            del Xt
    PTd = EV.apply_calibrator(cal, PT) if rule == 'expf' else PT
    sel = EV.apply_rule(prm, qi, pi, PTd, s3b, Wte['NQ'])
    fm, fc = os.path.join(out, 'matching_results.tsv'), os.path.join(out, 'candidate_pairs.tsv')
    qids, pids = Wte['Q'].entity_id.astype(object).to_numpy(), Wte['P'].entity_id.astype(object).to_numpy()
    with Timer('write submission files'):
        write_id_lists(fc, ['source1_entity_id', 'candidate_entity_ids'], qids, qi, pi, pids)
        write_id_lists(fm, ['source1_entity_id', 'matched_entity_ids'], qids, qi[sel], pi[sel], pids)

    # ---------------- test diagnostics + validation
    ck = Wte['q_ck']
    nc = np.bincount(qi, minlength=Wte['NQ']); npred = np.bincount(qi[sel], minlength=Wte['NQ'])
    diag = []
    for c in np.unique(ck):
        mk = ck == c
        diag.append(dict(country=c, S1=int(mk.sum()), cand_per_S1=nc[mk].mean(), pred_per_S1=npred[mk].mean(),
                         pct_S1_with_match=100 * (npred[mk] > 0).mean(), pct_S1_no_candidates=100 * (nc[mk] == 0).mean()))
    rep.table('test predictions per country (compare pred_per_S1 with holdout avg_pred; France = unseen country)',
              pd.DataFrame(diag).set_index('country'), short=True)
    q = np.quantile(PT, [0.1, 0.25, 0.5, 0.75, 0.9]) if len(PT) else []
    rep.text(f'test score quantiles (10/25/50/75/90%): {np.round(q, 4).tolist()} | matched pairs {int(sel.sum()):,} '
             f'of {len(PT):,} candidates', short=True)
    ok, checks, stats = validate_files(fm, fc, qids.tolist(), pids.tolist())
    rep.table('own validator', pd.Series(checks, name='pass').to_frame())
    vpath = os.path.join(ROOT, 'reference', 'validate_submission.py')
    from . import data as D
    tdir = os.path.dirname(D.discover(cfg['paths']['data_dir'])['test_s1'])
    try:
        r = subprocess.run([sys.executable, vpath, '--matching', fm, '--candidate', fc, '--test-dir', tdir, '--check-ids'],
                           capture_output=True, text=True, timeout=3600)
        rep.text('official validator (--check-ids): exit ' + str(r.returncode) + '\n```\n' + r.stdout[-1200:] + r.stderr[-600:] + '\n```', short=True)
        ok = ok and r.returncode == 0
    except Exception as e:
        rep.text(f'official validator could not run: {e}', short=True)
    minutes = dict(holdout=round((t2 - t1) / 60, 1), refit=round((t3 - t2) / 60, 1), test=round((time.time() - t3) / 60, 1),
                   total=round((time.time() - t0) / 60, 1))
    save_json(dict(exp=rep.exp_id, desc=rep.desc, plan={k: v for k, v in ctx.plan.items()}, stage1_cut=ctx.stage1['cut'] if ctx.stage1 else None,
                   models=specs, stage2=use_s2, text_models=[c['model'] for c in ce_cfgs], rule=rule, prm=prm, holdout=summ, refit=refit_desc,
                   test_candidates=crows, test_stats=stats, valid=ok, minutes=minutes), os.path.join(out, 'submission_meta.json'))
    row = dict(exp=rep.exp_id, desc=rep.desc, hold_F05=round(summ['hold_F05'], 5), hold_P=round(summ['hold_P_micro'], 4),
               hold_R=round(summ['hold_R_micro'], 4), singleton_acc=round(summ['singleton_acc'], 4), tune_F05=round(summ['tune_F05'], 5),
               oracle_hold=round(summ['oracle_hold'], 4), rule=rule, test_cand_per_S1=round(crows[-1]['avg_per_S1'], 3),
               test_pred_per_S1=round(float(npred.mean()), 3), valid=ok, minutes=minutes['total'], folder=out)
    sp = os.path.join(cfg['paths']['results_dir'], 'submissions', 'SUMMARY.csv')
    pd.DataFrame([row]).to_csv(sp, mode='a', header=not os.path.exists(sp), index=False)
    rep.text(f"**{rep.exp_id}: holdout F0.5 {summ['hold_F05']:.5f}** | files in {out} | valid={ok} | minutes {minutes}", short=True)
    rep.metric(submission_valid=ok, test_cand_per_S1=row['test_cand_per_S1'], test_pred_per_S1=row['test_pred_per_S1'], **stats)
    log(f'SUBMISSION FILES: {fm}  {fc}  valid={ok}')
