"""Final training + test inference + submission writing + validation (own checks + official validator)."""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

from . import pipeline as PL
from . import features as FT
from . import evaluate as EV
from .models import Model
from .utils import Timer, log, save_json


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


def submit(cfg, rep, args, build_ctx, feature_list, X_of, cap_rows, evaluate_scores, oof_p1):
    """args: models=[{kind, params}] (averaged), stage2=bool, rule='best'|'staged'|'expf'|'global'."""
    ctx = build_ctx(cfg, rep)
    feats = feature_list(ctx.F, cfg['model']['feature_drop'])
    specs = args.get('models', [dict(kind=cfg['model']['kind'], params=cfg['model']['params'])])
    use_s2 = bool(args.get('stage2', False))
    tr = cap_rows(ctx, ctx.rows['fit'], cfg['model']['max_train_rows'])
    rt, rh = ctx.rows['tune'], ctx.rows['hold']
    models = []
    if use_s2:
        mc = dict(cfg['model'], **specs[0])
        p1, m1 = oof_p1(ctx, mc, feats)
        models = [m1]
        imp = m1.importance()
        keep = list(pd.Series(imp, index=feats).sort_values(ascending=False).index[:cfg['stage2']['keep_feats']])
        S2 = FT.stage2_frame(ctx.qi, ctx.pi, ctx.s3.astype(np.int64), p1, ctx.W['P'], ctx.W['Pm'], ctx.F[keep])
        s2cols = list(S2.columns)
        m2 = Model('lgb', dict(num_leaves=63), cfg['model']['rounds'], cfg['model']['early_stop'], cfg['seed'], s2cols)
        m2.fit(S2.iloc[tr].to_numpy(np.float32), ctx.y[tr], None, S2.iloc[rt].to_numpy(np.float32), ctx.y[rt], None)
        pt, ph = m2.predict(S2.iloc[rt].to_numpy(np.float32)), m2.predict(S2.iloc[rh].to_numpy(np.float32))
        del S2
    else:
        for sp_ in specs:
            with Timer(f"final model {sp_['kind']}"):
                m = Model(sp_['kind'], sp_.get('params', {}), cfg['model']['rounds'], cfg['model']['early_stop'], cfg['seed'], feats)
                m.fit(X_of(ctx, tr, feats), ctx.y[tr], ctx.qi[tr], X_of(ctx, rt, feats), ctx.y[rt], ctx.qi[rt])
                models.append(m)
        pt = np.mean([m.predict(X_of(ctx, rt, feats)) for m in models], axis=0)
        ph = np.mean([m.predict(X_of(ctx, rh, feats)) for m in models], axis=0)
    res = evaluate_scores(ctx, pt.astype(np.float32), ph.astype(np.float32))
    rule = args.get('rule', 'best')
    rule = res['best'] if rule == 'best' else rule
    prm, cal = res[rule]['prm'], res[rule].get('cal')
    rep.text(f"validation (hold) with rule '{rule}': F0.5={res[rule]['hold']['F05']:.4f} "
             f"P={res[rule]['hold']['P_micro']:.4f} R={res[rule]['hold']['R_micro']:.4f} | prm={prm}", short=True)
    rep.metric(val_hold_F05=res[rule]['hold']['F05'], rule=rule)

    # ---------------- test world
    Wtr = ctx.W
    maps = dict(NAME_MAP=dict(__import__('erlab.text', fromlist=['x']).NAME_MAP), ADDR_MAP=dict(__import__('erlab.text', fromlist=['x']).ADDR_MAP))
    Wte = PL.build_test_world(cfg, maps)
    plan = ctx.plan
    pdfs = PL.run_passes(cfg, Wte, plan['passes'], Wtr=Wtr)
    from . import blocking as B
    cand = B.union_candidates(pdfs, plan['K'], Wte['NP'])
    del pdfs
    emb = {n: PL.get_embeddings(cfg, Wte, n, Wtr) for n in plan['emb']}
    log(f'test candidates: {len(cand):,} ({len(cand) / Wte["NQ"]:.1f}/S1)')
    with Timer('test features + scoring'):
        if use_s2:
            s1m = models[0]
            p1t, keepm, KEEP = FT.compute_features(cand, Wte, plan['passes'], emb, prune=plan['prune'], chunk=cfg['feats']['chunk'],
                                                   scorer=lambda F: s1m.predict(F[feats].to_numpy(np.float32)), keep_cols=keep,
                                                   alpha=cfg['retr']['alpha_name'])
            cand = cand[keepm].reset_index(drop=True)
            qi, pi = cand.qi.to_numpy(), cand.pi.to_numpy()
            s3 = (Wte['P'].src.to_numpy()[pi] == 3).astype(np.int64)
            PT = np.empty(len(cand), np.float32)
            for s, e in FT.group_chunks(qi, 2_000_000):
                S = pd.concat([FT.stage2_chunk(qi[s:e], pi[s:e], s3[s:e], p1t[s:e], Wte['P'], Wte['Pm']),
                               KEEP.iloc[s:e].reset_index(drop=True)], axis=1)
                PT[s:e] = m2.predict(S[s2cols].to_numpy(np.float32))
        else:
            PT, keepm, _ = FT.compute_features(cand, Wte, plan['passes'], emb, prune=plan['prune'], chunk=cfg['feats']['chunk'],
                                               scorer=lambda F: np.mean([m.predict(F[feats].to_numpy(np.float32)) for m in models], axis=0),
                                               alpha=cfg['retr']['alpha_name'])
            cand = cand[keepm].reset_index(drop=True)
            qi, pi = cand.qi.to_numpy(), cand.pi.to_numpy()
    s3b = Wte['P'].src.to_numpy()[pi] == 3
    PTd = EV.apply_calibrator(cal, PT) if rule == 'expf' else PT
    sel = EV.apply_rule(prm, qi, pi, PTd, s3b, Wte['NQ'])
    out = os.path.join(cfg['paths']['results_dir'], 'submission')
    os.makedirs(out, exist_ok=True)
    fm, fc = os.path.join(out, 'matching_results.tsv'), os.path.join(out, 'candidate_pairs.tsv')
    qids, pids = Wte['Q'].entity_id.astype(object).to_numpy(), Wte['P'].entity_id.astype(object).to_numpy()
    with Timer('write submission'):
        write_id_lists(fc, ['source1_entity_id', 'candidate_entity_ids'], qids, qi, pi, pids)
        write_id_lists(fm, ['source1_entity_id', 'matched_entity_ids'], qids, qi[sel], pi[sel], pids)
    ok, checks, stats = validate_files(fm, fc, qids.tolist(), pids.tolist())
    rep.table('own validator', pd.Series(checks, name='pass').to_frame(), short=True)
    rep.text(f'submission stats: {stats}', short=True)
    ck = Wte['q_ck']
    per = pd.Series(ck[np.unique(qi[sel])]).value_counts().rename('S1 with matches').to_frame()
    per['S1 total'] = pd.Series(ck).value_counts()
    rep.table('matched S1 by country (test)', per, short=True)
    vpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reference', 'validate_submission.py')
    from . import data as D
    tdir = os.path.dirname(D.discover(cfg['paths']['data_dir'])['test_s1'])
    try:
        r = subprocess.run([sys.executable, vpath, '--matching', fm, '--candidate', fc, '--test-dir', tdir],
                           capture_output=True, text=True, timeout=3600)
        rep.text('official validator: exit ' + str(r.returncode) + '\n```\n' + r.stdout[-1500:] + r.stderr[-800:] + '\n```', short=True)
        ok = ok and r.returncode == 0
    except Exception as e:
        rep.text(f'official validator could not run: {e}', short=True)
    save_json(dict(plan=plan, rule=rule, prm=prm, feats=feats, stats=stats), os.path.join(out, 'submission_meta.json'))
    rep.metric(submission_valid=ok, **stats)
    log(f'SUBMISSION FILES: {fm}  {fc}  valid={ok}')
