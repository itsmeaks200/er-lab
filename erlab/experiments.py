"""Experiment runners + registry. Each experiment = (kind, config overrides, args). Results go to results/."""
import os
import gc
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import pipeline as PL
from . import features as FT
from . import evaluate as EV
from . import blocking as B
from . import prune as PR
from .models import Model, blend_weights, rank01
from .report import Report
from .utils import Timer, log, cfg_hash, save_json, load_json, TIMINGS
from .config import make_cfg

# ------------------------------------------------------------------ feature groups for ablations
GROUPS = {
    'address': ('a_', 'q_a', 'p_a', 'q_af', 'p_af', 'comb_cos', 'name_x_addr', 'both_addr', 'blk_D', 'blk_E', 'blk_H'),
    'numeric_postal': ('a_num', 'a_post'),
    'context': ('blk_', 'q_ncand'),   # plus *_gap/_rank/_z handled below
    'commonness': ('q_nf', 'p_nf', 'q_af', 'p_af'),
    'fuzzy': ('n_ratio', 'n_lev', 'n_jw', 'n_tsort', 'n_tset', 'n_partial', 'n_sq_ratio', 'n_skel_ratio', 'n_alt_ratio',
              'a_ratio', 'a_tset', 'a_tsort', 'a_partial'),
    'embeddings': ('embG', 'embS', 'blk_G', 'blk_S'),
    'noise_flags': ('q_f_', 'p_f_'),
}


def feature_list(F, drop=()):
    cols = list(F.columns)
    out = []
    for c in cols:
        bad = False
        for g in drop:
            pref = GROUPS.get(g, (g,))
            if c.startswith(pref):
                bad = True
            if g == 'context' and c.endswith(FT.CTX_SUFFIX):
                bad = True
        if not bad:
            out.append(c)
    return out


# ------------------------------------------------------------------ context (world + candidates + features)
_CTX = {}


def _torch_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def base_parts(cfg):
    """Train world + blocking passes + selected plan (shared by build_ctx and the stage-1 runner)."""
    passes = list(cfg['block']['passes'])
    if 'S' not in passes and cfg['st']['enabled']:
        passes.append('S')
    if not _torch_gpu():
        dropped = [p for p in passes if p in B.DENSE_PASSES]
        passes = [p for p in passes if p not in B.DENSE_PASSES]
        if dropped:
            log(f'no GPU/torch → dense passes {dropped} skipped')
    emb = []
    if cfg['encoder']['enabled'] and _torch_gpu():
        emb.append('embG')
    if cfg['st']['enabled'] and _torch_gpu():
        emb.append('embS')
    key = cfg_hash(dict(w=PL.world_key(cfg, 'train'), b=cfg['block'], r=cfg['retr'], passes=passes, emb=emb,
                        e=cfg['encoder'], s=cfg['st'], f=cfg['feats'], pc=cfg['prune1'], ef=cfg['world'].get('enc_frac', 0)))
    return passes, emb, key


def build_ctx(cfg, rep=None):
    passes, emb, key = base_parts(cfg)
    if key in _CTX:
        return _CTX[key]
    W = PL.build_train_world(cfg)
    pdfs = PL.run_passes(cfg, W, passes)
    blk = PL.select_blocking(cfg, W, pdfs)
    plan = dict(passes=blk['passes'], K=blk['K'], emb=emb)
    if cfg['prune1']['enabled']:
        S1 = PL.run_stage1(cfg, W, pdfs, plan)
        info = stage1_info(cfg, W, S1)
        prm = PR.choose_cut(info['frontier'], info['oracle0']['tune'], cfg['prune1']['max_oracle_loss'])
        ctx = next(stage1_ctxs(cfg, W, pdfs, blk, plan, S1, info, [prm], key))
    else:
        cand, F = PL.get_features(cfg, W, pdfs, plan)
        y = PL.label_candidates(W, cand)
        sz = FT.safe_zone(F)
        loss = float(y[sz].sum() / max(len(W['gt_keys']), 1))
        prune = bool(cfg['feats']['prune'] and loss <= cfg['feats']['prune_max_loss'])
        if prune:
            cand, F, y = PL.apply_prune(cand, F, y, plan)
        plan['prune'] = prune
        ctx = make_ctx(cfg, W, pdfs, blk, plan, cand, F, y, key,
                       dict(pruned_share=float(sz.mean()), recall_loss=loss, applied=prune))
    _CTX.clear()
    _CTX[key] = ctx
    return ctx


def make_ctx(cfg, W, pdfs, blk, plan, cand, F, y, key, prune_info, stage1=None):
    qi, pi = cand.qi.to_numpy(), cand.pi.to_numpy()
    fold = W['q_fold'][qi]
    ctx = SimpleNamespace(cfg=cfg, W=W, pdfs=pdfs, blk=blk, plan=plan, cand=cand, F=F, y=y, qi=qi, pi=pi,
                          s3=(W['P'].src.to_numpy()[pi] == 3), fold=fold, key=key,
                          rows={f: np.flatnonzero(fold == f) for f in ('fit', 'tune', 'hold')},
                          prune_info=prune_info, stage1=stage1)
    ctx.colidx = {c: i for i, c in enumerate(F.columns)}
    ctx.oracle = {f: EV.s1_metrics(qi[ctx.rows[f]][y[ctx.rows[f]] == 1], np.ones(int(y[ctx.rows[f]].sum())),
                                   W['n_true'], W['fold_q'][f])['F05'] for f in ctx.rows}
    return ctx


def stage1_info(cfg, W, S1):
    """Stage-0 oracle per fold + the tune-fold frontier of pruner cuts."""
    qi0, y0 = S1['cand'].qi.to_numpy(), S1['y']
    fold0 = W['q_fold'][qi0]
    rows0 = {f: np.flatnonzero(fold0 == f) for f in ('fit', 'tune', 'hold')}
    oracle0 = {f: EV.s1_metrics(qi0[r][y0[r] == 1], np.ones(int(y0[r].sum())), W['n_true'], W['fold_q'][f])['F05']
               for f, r in rows0.items()}
    with Timer('stage-1 cut frontier (tune)'):
        fr = PR.frontier(S1['p'], y0, qi0, S1['qr'], S1['pr'], rows0['tune'], W['fold_q']['tune'], W['n_true'], cfg['prune1'])
    keys0 = qi0.astype(np.int64) * W['NP'] + S1['cand'].pi.to_numpy(np.int64)
    from sklearn.metrics import roc_auc_score
    h = rows0['hold']
    auc = float(roc_auc_score(y0[h], S1['p'][h])) if 0 < y0[h].mean() < 1 else float('nan')
    return dict(oracle0=oracle0, frontier=fr, stats0=PL.cand_stats(W, 'stage 0 union', keys0), rows0=rows0, auc=auc)


def stage1_ctxs(cfg, W, pdfs, blk, plan, S1, info, prms, key):
    """One ctx per cut. Matcher features are computed once on the union of all survivors, then subset per cut."""
    masks = [PR.cut_mask(S1['p'], S1['qr'], S1['pr'], prm) for prm in prms]
    U = np.logical_or.reduce(masks)
    cand_u, F_u = PL.stage1_features(cfg, W, plan, S1, U, str(prms))
    plan = dict(plan, prune=False, stage1=True)
    for prm, mk in zip(prms, masks):
        if len(prms) == 1:
            cand, F = cand_u, F_u
        else:
            cand, F = PL.subset_context(cand_u, F_u, np.flatnonzero(mk[U]), plan['passes'])
        y = PL.label_candidates(W, cand)
        keys = cand.qi.to_numpy(np.int64) * W['NP'] + cand.pi.to_numpy(np.int64)
        st = dict(cut=prm, info=info, model=S1['model'], cols=S1['cols'], imp=S1['imp'],
                  stats=PL.cand_stats(W, f"stage 1 cut {prm}", keys))
        yield make_ctx(cfg, W, pdfs, blk, plan, cand, F, y, key + '_' + cfg_hash(prm), dict(stage1_cut=prm), st)


def X_of(ctx, rows, feats):
    return ctx.F.iloc[rows, [ctx.colidx[c] for c in feats]].to_numpy(np.float32)


def cap_rows(ctx, rows, max_rows, seed=42):
    if len(rows) <= max_rows:
        return rows
    qs = np.unique(ctx.qi[rows])
    keep = np.random.RandomState(seed).choice(qs, int(len(qs) * max_rows / len(rows)), replace=False)
    return rows[np.isin(ctx.qi[rows], keep)]


def subset_rows(ctx, rows, n_s1, seed=42):
    qs = np.unique(ctx.qi[rows])
    if len(qs) <= n_s1:
        return rows
    keep = np.random.RandomState(seed).choice(qs, n_s1, replace=False)
    return rows[np.isin(ctx.qi[rows], keep)]


# ------------------------------------------------------------------ evaluation of a score vector
def evaluate_scores(ctx, pt, ph, modes=('global', 'staged', 'expf')):
    W, rt, rh = ctx.W, ctx.rows['tune'], ctx.rows['hold']
    evT = EV.RuleEval(ctx.qi[rt], ctx.pi[rt], pt, ctx.y[rt], ctx.s3[rt], W['fold_q']['tune'], W['n_true'])
    evH = EV.RuleEval(ctx.qi[rh], ctx.pi[rh], ph, ctx.y[rh], ctx.s3[rh], W['fold_q']['hold'], W['n_true'])
    out = {}
    if 'global' in modes:
        prm, ft = EV.tune_global(evT)
        out['global'] = dict(prm=prm, tune=ft, hold=evH.metrics(prm))
    if 'staged' in modes:
        prm, ft, _ = EV.tune_staged(evT)
        out['staged'] = dict(prm=prm, tune=ft, hold=evH.metrics(prm))
    if 'expf' in modes:
        cal = EV.fit_calibrator(pt, ctx.y[rt])
        ptc, phc = EV.apply_calibrator(cal, pt), EV.apply_calibrator(cal, ph)
        prm, ft, _ = EV.tune_expf(ctx.qi[rt], ctx.pi[rt], ptc, ctx.y[rt], W['fold_q']['tune'], W['n_true'])
        sel = EV.apply_rule(prm, ctx.qi[rh], ctx.pi[rh], phc, ctx.s3[rh], W['NQ'])
        out['expf'] = dict(prm=prm, tune=ft, hold=EV.s1_metrics(ctx.qi[rh][sel], ctx.y[rh][sel], W['n_true'], W['fold_q']['hold']),
                           cal=cal)
    best = max(out, key=lambda k: out[k]['tune'])
    out['best'] = best
    from sklearn.metrics import roc_auc_score, log_loss
    yh = ctx.y[rh]
    out['auc'] = float(roc_auc_score(yh, ph)) if 0 < yh.mean() < 1 else float('nan')
    out['logloss'] = float(log_loss(yh, np.clip(ph, 1e-6, 1 - 1e-6))) if 0 < yh.mean() < 1 else float('nan')
    return out


def score_row(label, res, extra=None):
    b = res['best']
    r = dict(model=label, best_rule=b, tune_F05=round(res[b]['tune'], 5), hold_F05=round(res[b]['hold']['F05'], 5),
             hold_P=round(res[b]['hold']['P_micro'], 4), hold_R=round(res[b]['hold']['R_micro'], 4),
             singleton_acc=round(res[b]['hold']['singleton_acc'], 4), avg_pred=round(res[b]['hold']['avg_pred'], 3),
             auc=round(res['auc'], 5))
    for m in ('global', 'staged', 'expf'):
        if m in res:
            r[f'hold_{m}'] = round(res[m]['hold']['F05'], 5)
    r.update(extra or {})
    return r


def _pred_path(ctx, label):
    return os.path.join(ctx.cfg['paths']['cache_dir'], 'preds', f'{ctx.key}_{label}.npz')


def train_eval(ctx, mcfg, feats, label, rep=None, tr_rows=None):
    tr = tr_rows if tr_rows is not None else cap_rows(ctx, ctx.rows['fit'], mcfg['max_train_rows'])
    rt, rh = ctx.rows['tune'], ctx.rows['hold']
    with Timer(f'train {label} ({mcfg["kind"]}, {len(feats)} feats, {len(tr):,} rows)'):
        m = Model(mcfg['kind'], mcfg['params'], mcfg['rounds'], mcfg['early_stop'], ctx.cfg['seed'], feats)
        m.fit(X_of(ctx, tr, feats), ctx.y[tr], ctx.qi[tr], X_of(ctx, rt, feats), ctx.y[rt], ctx.qi[rt])
        pt = m.predict(X_of(ctx, rt, feats)).astype(np.float32)
        ph = m.predict(X_of(ctx, rh, feats)).astype(np.float32)
    res = evaluate_scores(ctx, pt, ph)
    os.makedirs(os.path.dirname(_pred_path(ctx, label)), exist_ok=True)
    np.savez(_pred_path(ctx, label), pt=pt, ph=ph)
    row = score_row(label, res, dict(n_feats=len(feats), best_iter=getattr(m, 'best_iter', 0), train_rows=len(tr)))
    if rep is not None:
        rep.leaderboard(row)
    return m, res, row


# ------------------------------------------------------------------ runners
def run_env(cfg, rep, args):
    import platform
    import importlib
    rows = []
    for mod in ['numpy', 'pandas', 'pyarrow', 'scipy', 'sklearn', 'lightgbm', 'xgboost', 'catboost', 'rapidfuzz',
                'numba', 'torch', 'transformers', 'sentence_transformers', 'optuna', 'psutil', 'tabulate']:
        try:
            v = importlib.import_module(mod).__version__
        except Exception as e:
            v = f'MISSING ({type(e).__name__})'
        rows.append(dict(package=mod, version=v))
    rep.table('packages', pd.DataFrame(rows).set_index('package'), short=True, short_rows=30)
    info = dict(python=platform.python_version(), os=platform.platform(), cpus=os.cpu_count())
    try:
        import psutil
        info['ram_gb'] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        import torch
        info['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
        if torch.cuda.is_available():
            info['gpu_mem_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        info['gpu'] = 'torch missing'
    try:
        from . import data as D
        info['data_files'] = {k: f'{os.path.getsize(v) / 1e6:.0f}MB' for k, v in D.discover(cfg['paths']['data_dir']).items()}
    except Exception as e:
        info['data_files'] = f'NOT FOUND: {e}'
    rep.text('```\n' + '\n'.join(f'{k}: {v}' for k, v in info.items()) + '\n```', short=True)
    rep.metric(**{k: v for k, v in info.items() if k != 'data_files'})


def run_blocking(cfg, rep, args):
    ctx = build_ctx(cfg, rep)
    blk = ctx.blk
    rep.table('per-pass candidate quality (K = topk_max)', blk['table'], short=True)
    rep.table('recall@K (ranked passes)', blk['curves'], short=True)
    rep.table('greedy union (fit-fold recall)', blk['greedy'], short=True)
    final = PL.cand_stats(ctx.W, 'UNION ' + '+'.join(ctx.plan['passes']),
                          ctx.cand.qi.to_numpy(np.int64) * ctx.W['NP'] + ctx.cand.pi.to_numpy(np.int64))
    if ctx.stage1:
        report_stage1(rep, ctx.stage1)
    else:
        rep.table('final union after pruning', pd.DataFrame([final]).set_index('strategy'), short=True)
    rep.text(f"K selected: {ctx.plan['K']} | prune: {ctx.prune_info} | oracle F0.5 per fold: "
             f"{ {k: round(v, 4) for k, v in ctx.oracle.items()} }", short=True)
    missed = ctx.W['gt_keys'][~np.isin(ctx.W['gt_keys'], ctx.cand.qi.to_numpy(np.int64) * ctx.W['NP'] + ctx.cand.pi.to_numpy(np.int64))]
    ex = []
    for k in missed[:12]:
        q, p = int(k // ctx.W['NP']), int(k % ctx.W['NP'])
        ex.append(dict(s1_name=ctx.W['Q'].business_name.iloc[q], s1_addr=ctx.W['Q'].business_address.iloc[q][:60],
                       pool_name=ctx.W['P'].business_name.iloc[p], pool_addr=ctx.W['P'].business_address.iloc[p][:60]))
    rep.table('examples of true pairs lost by blocking', pd.DataFrame(ex), short=True, short_rows=8)
    rep.metric(recall=final['recall'], avg_per_S1=final['avg_per_S1'], oracle_hold=ctx.oracle['hold'], passes=ctx.plan['passes'])


def report_stage1(rep, st, frontier_rows=20, show_cands=True):
    info = st['info']
    if show_cands:
        rep.table('candidates per S1: stage 0 (retrieval union) → stage 1 (learned pruner) = what the matcher scores',
                  pd.DataFrame([info['stats0'], st['stats']]).set_index('strategy'), short=True)
    rep.table('stage-1 cut frontier on tune (Pareto: candidates/S1 vs oracle F0.5)', PR.pareto(info['frontier']),
              short=True, max_rows=60, short_rows=frontier_rows)
    imp = pd.Series(st['imp']).sort_values(ascending=False)
    rep.table('stage-1 pruner gain importance (top 15)', (imp / max(imp.sum(), 1e-9)).head(15).to_frame('share'), short=True)
    rep.text(f"stage-0 oracle F0.5 { {k: round(v, 4) for k, v in info['oracle0'].items()} } | pruner hold AUC {info['auc']:.5f} | "
             f"{len(st['cols'])} cheap features" + (f" | chosen cut {st['cut']}" if show_cands else ''), short=True)


def run_prune(cfg, rep, args):
    """Stage-1 learned blocking: frontier of cuts, then the matcher trained/evaluated on the survivors of several
    oracle-loss budgets → hold F0.5 as a function of the candidate-set size."""
    passes, emb, key = base_parts(cfg)
    W = PL.build_train_world(cfg)
    pdfs = PL.run_passes(cfg, W, passes)
    blk = PL.select_blocking(cfg, W, pdfs)
    rep.table('greedy union (fit-fold recall)', blk['greedy'], short=True)
    plan = dict(passes=blk['passes'], K=blk['K'], emb=emb)
    S1 = PL.run_stage1(cfg, W, pdfs, plan)
    info = stage1_info(cfg, W, S1)
    labels, prms = [], []
    if args.get('baseline'):
        labels.append('all stage-0 pairs'); prms.append(dict(t=-1.0, n_max=0, m_pool=0))
    for b in args.get('budgets', [0.0005, 0.001, 0.002, 0.004, 0.008]):
        prm = PR.choose_cut(info['frontier'], info['oracle0']['tune'], b)
        if prm not in prms:
            labels.append(f'budget {b}'); prms.append(prm)
    rows, first, first_row = [], True, True
    for lab, ctx in zip(labels, stage1_ctxs(cfg, W, pdfs, blk, plan, S1, info, prms, key)):
        if first:
            report_stage1(rep, ctx.stage1, show_cands=False)
            first = False
        feats = feature_list(ctx.F, cfg['model']['feature_drop'])
        _, res, row = train_eval(ctx, cfg['model'], feats, f'{rep.exp_id}-{lab}', rep)
        st = ctx.stage1['stats']
        if first_row:
            rows.append(dict(setting='stage 0 retrieval union', cut='-', cand_per_S1=info['stats0']['avg_per_S1'],
                             p50=info['stats0']['p50_per_S1'], p95=info['stats0']['p95_per_S1'], max=info['stats0']['max_per_S1'],
                             pair_recall=info['stats0']['recall'], oracle_hold=info['oracle0']['hold']))
            first_row = False
        rows.append(dict(setting=lab, cut=str(ctx.stage1['cut']), cand_per_S1=st['avg_per_S1'], p50=st['p50_per_S1'],
                         p95=st['p95_per_S1'], max=st['max_per_S1'], pair_recall=st['recall'], oracle_hold=ctx.oracle['hold'],
                         hold_F05=row['hold_F05'], hold_P=row['hold_P'], hold_R=row['hold_R'], singleton_acc=row['singleton_acc'],
                         avg_pred=row['avg_pred'], tune_F05=row['tune_F05'], rule=row['best_rule']))
        del ctx
        gc.collect()
    df = pd.DataFrame(rows).set_index('setting')
    rep.table('matcher (LightGBM) F0.5 vs candidate-set size', df, short=True)
    rep.metric(**{f"F05@{r['cand_per_S1']:.2f}cand": r['hold_F05'] for r in rows if 'hold_F05' in r})


def run_model(cfg, rep, args):
    ctx = build_ctx(cfg, rep)
    feats = feature_list(ctx.F, cfg['model']['feature_drop'])
    m, res, row = train_eval(ctx, cfg['model'], feats, args.get('label', rep.exp_id), rep)
    rep.table('result (hold fold, rule chosen on tune)', pd.DataFrame([row]).set_index('model'), short=True)
    rep.text(f"rules: " + ' | '.join(f"{k}: {res[k]['prm']}" for k in ('global', 'staged', 'expf') if k in res))
    imp = m.importance()
    if imp is not None:
        rep.table('top-25 gain importance', pd.Series(imp, index=feats).sort_values(ascending=False).head(25).to_frame('gain'),
                  short=args.get('show_imp', False))
    rep.metric(hold_F05=row['hold_F05'], tune_F05=row['tune_F05'], rule=row['best_rule'], oracle_hold=ctx.oracle['hold'])


def run_ablation(cfg, rep, args):
    ctx = build_ctx(cfg, rep)
    base_rows = subset_rows(ctx, ctx.rows['fit'], args.get('n_s1', 60_000))
    mc = dict(cfg['model'], rounds=100 if cfg.get('smoke') else args.get('rounds', 1500))
    rows = []
    for drop in [()] + [(g,) for g in args.get('groups', list(GROUPS))]:
        feats = feature_list(ctx.F, drop)
        if not feats:
            continue
        _, _, row = train_eval(ctx, mc, feats, f"{rep.exp_id}-no_{'_'.join(drop) or 'NONE'}", rep, tr_rows=base_rows)
        rows.append(dict(dropped='+'.join(drop) or '(none)', n_feats=len(feats), hold_F05=row['hold_F05'], tune_F05=row['tune_F05']))
    df = pd.DataFrame(rows).set_index('dropped')
    df['delta_vs_full'] = df.hold_F05 - df.loc['(none)', 'hold_F05']
    rep.table('feature-group ablation (quick LightGBM, same rows)', df, short=True)
    rep.metric(full=float(df.loc['(none)', 'hold_F05']))


def run_blend(cfg, rep, args):
    ctx = build_ctx(cfg, rep)
    preds = {}
    for lab in args['members']:
        p = _pred_path(ctx, lab)
        if os.path.exists(p):
            z = np.load(p)
            preds[lab] = (z['pt'], z['ph'])
        else:
            rep.text(f'missing predictions for {lab} (run it first with the same world/blocking config)', short=True)
    if len(preds) < 2:
        return
    corr = pd.DataFrame({k: v[1] for k, v in preds.items()}).corr()
    rep.table('hold prediction correlation', corr, short=True)
    rt = ctx.rows['tune']
    evT = lambda p: EV.tune_global(EV.RuleEval(ctx.qi[rt], ctx.pi[rt], p, ctx.y[rt], ctx.s3[rt], ctx.W['fold_q']['tune'], ctx.W['n_true']))[1]
    rows = []
    for name, tf in [('prob', lambda x: x), ('rank', rank01)]:
        pt = {k: tf(v[0]) for k, v in preds.items()}
        ph = {k: tf(v[1]) for k, v in preds.items()}
        w, f = blend_weights(pt, None, evT)
        bt = sum(w[k] * pt[k] for k in w); bh = sum(w[k] * ph[k] for k in w)
        res = evaluate_scores(ctx, bt, bh)
        row = score_row(f'blend-{name}', res, dict(weights=str({k: round(v, 2) for k, v in w.items()})))
        rep.leaderboard(row)
        rows.append(row)
    rep.table('blends (hill-climbing weights on tune)', pd.DataFrame(rows).set_index('model'), short=True)
    rep.metric(best_blend=max(r['hold_F05'] for r in rows))


def _oof_p1(ctx, mcfg, feats, n_folds=2):
    """Out-of-fold stage-1 scores on fit rows (by S1 hash), in-sample-free scores for tune/hold from a full-fit model."""
    fit = ctx.rows['fit']
    h = (pd.util.hash_array(ctx.W['Q'].entity_id.astype(object).to_numpy()) % n_folds).astype(np.int64)[ctx.qi]
    p1 = np.zeros(len(ctx.y), np.float32)
    full = Model(mcfg['kind'], mcfg['params'], mcfg['rounds'], mcfg['early_stop'], ctx.cfg['seed'], feats)
    tr = cap_rows(ctx, fit, mcfg['max_train_rows'])
    full.fit(X_of(ctx, tr, feats), ctx.y[tr], ctx.qi[tr], X_of(ctx, ctx.rows['tune'], feats), ctx.y[ctx.rows['tune']], ctx.qi[ctx.rows['tune']])
    for rows in (ctx.rows['tune'], ctx.rows['hold'], np.flatnonzero(ctx.fold == 'enc')):   # enc: unseen by `full`
        if len(rows):
            p1[rows] = full.predict(X_of(ctx, rows, feats))
    rounds = max(100, int(getattr(full, 'best_iter', 500) * 1.05))
    for k in range(n_folds):
        trk, prk = tr[h[tr] != k], fit[h[fit] == k]
        mk = Model(mcfg['kind'], mcfg['params'], rounds, 10 ** 9, ctx.cfg['seed'], feats)
        mk.fit(X_of(ctx, trk, feats), ctx.y[trk], ctx.qi[trk], X_of(ctx, ctx.rows['tune'][:1000], feats),
               ctx.y[ctx.rows['tune'][:1000]], ctx.qi[ctx.rows['tune'][:1000]])
        p1[prk] = mk.predict(X_of(ctx, prk, feats))
    return p1, full


def run_stage2(cfg, rep, args):
    ctx = build_ctx(cfg, rep)
    mc = cfg['model']
    feats = feature_list(ctx.F, mc['feature_drop'])
    with Timer('stage-1 OOF'):
        p1, m1 = _oof_p1(ctx, mc, feats)
    res1 = evaluate_scores(ctx, p1[ctx.rows['tune']], p1[ctx.rows['hold']])
    row1 = score_row(f'{rep.exp_id}-stage1', res1)
    rep.leaderboard(row1)
    imp = m1.importance()
    keep = list(pd.Series(imp, index=feats).sort_values(ascending=False).index[:cfg['stage2']['keep_feats']]) if imp is not None else feats[:15]
    with Timer('stage-2 features'):
        S2 = FT.stage2_frame(ctx.qi, ctx.pi, ctx.s3.astype(np.int64), p1, ctx.W['P'], ctx.W['Pm'], ctx.F[keep])
    if args.get('crossenc'):
        S2['ce'] = _crossenc_feature(ctx, p1, rep)
    cols = list(S2.columns)
    X = lambda r: S2.iloc[r].to_numpy(np.float32)
    tr = cap_rows(ctx, ctx.rows['fit'], mc['max_train_rows'])
    m2 = Model('lgb', dict(num_leaves=63), mc['rounds'], mc['early_stop'], cfg['seed'], cols)
    m2.fit(X(tr), ctx.y[tr], None, X(ctx.rows['tune']), ctx.y[ctx.rows['tune']], None)
    pt, ph = m2.predict(X(ctx.rows['tune'])), m2.predict(X(ctx.rows['hold']))
    res2 = evaluate_scores(ctx, pt, ph)
    row2 = score_row(f'{rep.exp_id}-stage2', res2)
    rep.leaderboard(row2)
    os.makedirs(os.path.dirname(_pred_path(ctx, 'x')), exist_ok=True)
    np.savez(_pred_path(ctx, f'{rep.exp_id}-stage2'), pt=pt, ph=ph)
    rep.table('stage 1 vs stage 2 (hold)', pd.DataFrame([row1, row2]).set_index('model'), short=True)
    rep.table('stage-2 gain importance', pd.Series(m2.importance(), index=cols).sort_values(ascending=False).head(15).to_frame('gain'), short=True)
    rep.metric(stage1=row1['hold_F05'], stage2=row2['hold_F05'])


def _crossenc_feature(ctx, p1, rep):
    from . import crossenc as CE
    cc = ctx.cfg['crossenc']
    W = ctx.W
    enc = np.flatnonzero(ctx.fold == 'enc')
    src = enc if len(enc) else ctx.rows['fit']      # train on the encoder fold -> no in-sample CE scores on fit rows
    rng = np.random.RandomState(ctx.cfg['seed'])
    pos = src[ctx.y[src] == 1]
    neg = src[ctx.y[src] == 0]
    rk = pd.Series(p1[neg]).groupby(ctx.qi[neg]).rank(ascending=False, method='first').to_numpy()
    hard = neg[rk <= 4]
    nneg = min(len(hard), cc['neg_per_pos'] * len(pos))
    tr = np.r_[pos, rng.choice(hard, nneg, replace=False)]
    if len(tr) > cc['max_train_pairs']:
        tr = rng.choice(tr, cc['max_train_pairs'], replace=False)
    rep.text(f"cross-encoder {cc['model']}: {len(tr):,} training pairs from the "
             f"{'enc' if len(enc) else 'fit'} fold; scoring {int((ctx.fold != 'enc').sum()):,} candidate rows in scope", short=True)
    with Timer(f'cross-encoder train on {len(tr):,} pairs'):
        model, tok = CE.train_crossenc(cc, CE.serialise(W['Q'], ctx.qi[tr]), CE.serialise(W['P'], ctx.pi[tr]), ctx.y[tr], ctx.cfg['seed'])
    ce = np.full(len(p1), np.nan, np.float32)
    sel = CE.select_rows(ctx.qi, p1, cc['band'], cc['top_n'], cc.get('max_pred_pairs', 0))
    sel = sel[ctx.fold[sel] != 'enc']
    with Timer(f'cross-encoder predict {len(sel):,} pairs'):
        ce[sel] = CE.predict_crossenc(model, tok, CE.serialise(W['Q'], ctx.qi[sel]), CE.serialise(W['P'], ctx.pi[sel]), cc)
    from sklearn.metrics import roc_auc_score
    h = np.intersect1d(sel, ctx.rows['hold'])
    if len(h) and 0 < ctx.y[h].mean() < 1:
        rep.text(f'cross-encoder AUC on scored hold pairs: {roc_auc_score(ctx.y[h], ce[h]):.4f} vs stage-1 '
                 f'{roc_auc_score(ctx.y[h], p1[h]):.4f} (n={len(h):,})', short=True)
    return ce


def run_hpo(cfg, rep, args):
    import optuna
    ctx = build_ctx(cfg, rep)
    feats = feature_list(ctx.F, cfg['model']['feature_drop'])
    tr = subset_rows(ctx, ctx.rows['fit'], args.get('n_s1', 60_000))
    rt = ctx.rows['tune']
    Xtr, Xt = X_of(ctx, tr, feats), X_of(ctx, rt, feats)

    def objective(trial):
        prm = dict(num_leaves=trial.suggest_int('num_leaves', 15, 511, log=True),
                   min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 20, 2000, log=True),
                   feature_fraction=trial.suggest_float('feature_fraction', 0.3, 1.0),
                   learning_rate=trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
                   lambda_l2=trial.suggest_float('lambda_l2', 1e-3, 30, log=True))
        m = Model('lgb', prm, 3000, 100, cfg['seed'], feats).fit(Xtr, ctx.y[tr], None, Xt, ctx.y[rt], None)
        ev = EV.RuleEval(ctx.qi[rt], ctx.pi[rt], m.predict(Xt), ctx.y[rt], ctx.s3[rt], ctx.W['fold_q']['tune'], ctx.W['n_true'])
        return EV.tune_global(ev)[1]
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=cfg['seed']))
    study.optimize(objective, n_trials=3 if cfg.get('smoke') else args.get('trials', 40))
    df = study.trials_dataframe()[['number', 'value'] + [c for c in study.trials_dataframe().columns if c.startswith('params_')]]
    rep.table('optuna trials (top 10 by tune F0.5)', df.sort_values('value', ascending=False).head(10), short=True)
    best = study.best_params
    rep.text(f'best params: {best}', short=True)
    mc = dict(cfg['model'], params=dict(cfg['model']['params'], **best))
    _, _, row = train_eval(ctx, mc, feats, f'{rep.exp_id}-best-full', rep)
    rep.table('best params retrained on all fit rows', pd.DataFrame([row]).set_index('model'), short=True)
    save_json(best, os.path.join(cfg['paths']['results_dir'], f'{rep.exp_id}_best_params.json'))
    rep.metric(hold_F05=row['hold_F05'], best_params=best)


def run_decision(cfg, rep, args):
    """Deep dive into decision rules for a saved prediction (default: M01)."""
    ctx = build_ctx(cfg, rep)
    lab = args.get('pred', 'M01')
    p = _pred_path(ctx, lab)
    if not os.path.exists(p):
        rep.text(f'run {lab} first', short=True)
        return
    z = np.load(p)
    res = evaluate_scores(ctx, z['pt'], z['ph'])
    rows = [dict(rule=k, tune=res[k]['tune'], **{kk: v for kk, v in res[k]['hold'].items() if kk != 'n_S1'}, prm=str(res[k]['prm']))
            for k in ('global', 'staged', 'expf')]
    rep.table('decision rules (hold)', pd.DataFrame(rows).set_index('rule'), short=True)
    rh = ctx.rows['hold']
    b = res['best']
    ph = z['ph'] if b != 'expf' else EV.apply_calibrator(res['expf']['cal'], z['ph'])
    sel = EV.apply_rule(res[b]['prm'], ctx.qi[rh], ctx.pi[rh], ph, ctx.s3[rh], ctx.W['NQ'])
    q_eval = ctx.W['fold_q']['hold']
    ck = ctx.W['q_ck']
    per = []
    for c in np.unique(ck[q_eval]):
        qe = q_eval[ck[q_eval] == c]
        mm = np.isin(ctx.qi[rh][sel], qe)
        per.append(dict(country=c, **EV.s1_metrics(ctx.qi[rh][sel][mm], ctx.y[rh][sel][mm], ctx.W['n_true'], qe)))
    rep.table('per-country hold metrics (best rule)', pd.DataFrame(per).set_index('country'), short=True)
    nt = ctx.W['n_true'][q_eval]
    npred = np.bincount(ctx.qi[rh][sel], minlength=ctx.W['NQ'])[q_eval]
    rep.table('true vs predicted multiplicity (hold)', pd.crosstab(np.minimum(nt, 8), np.minimum(npred, 8)), short=True)
    rep.metric(best_rule=b, hold_F05=res[b]['hold']['F05'])


RUNNERS = {'env': run_env, 'blocking': run_blocking, 'prune': run_prune, 'model': run_model, 'ablation': run_ablation, 'blend': run_blend,
           'stage2': run_stage2, 'hpo': run_hpo, 'decision': run_decision}


def run_submit(cfg, rep, args):
    from .submit import submit
    submit(cfg, rep, args, build_ctx, feature_list, X_of, cap_rows, evaluate_scores, _oof_p1)


RUNNERS['submit'] = run_submit


def run_experiment(exp_id, spec, cli_sets, base_over=None):
    from .config import deep_merge
    cfg = make_cfg(deep_merge(base_over or {}, spec.get('cfg', {})), cli_sets)
    rep = Report(cfg, exp_id, spec['desc'])
    t = time.time()
    try:
        RUNNERS[spec['kind']](cfg, rep, dict(spec.get('args', {})))
        rep.metric(minutes=round((time.time() - t) / 60, 1))
        rep.close('OK')
        return True
    except Exception:
        rep.metric(minutes=round((time.time() - t) / 60, 1))
        rep.close('FAILED', traceback.format_exc())
        return False
