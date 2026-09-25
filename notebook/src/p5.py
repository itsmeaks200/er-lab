# %% [markdown]
# ## 16. Threshold optimisation
# There is no arbitrary `p > 0.5`. Thresholds are searched on the **tune** fold, and the same curve is shown on **hold** to confirm that the operating region is stable.

# %%
EV_T = RuleEval(ROWS['tune'], FINAL_P['tune'], 'tune')
EV_H = RuleEval(ROWS['hold'], FINAL_P['hold'], 'hold')
_curve = []
for t in T_GRID:
    mt, mh = EV_T.metrics(base_prm(t)), EV_H.metrics(base_prm(t))
    _curve.append(dict(threshold=t, tune_F05=mt['F05'], hold_F05=mh['F05'], precision=mh['P_micro'], recall=mh['R_micro'],
                       false_merge_rate=mh['false_merge_rate'], singleton_acc=mh['singleton_acc'], avg_matches_per_S1=mh['avg_pred_per_S1']))
CURVE = pd.DataFrame(_curve)
display(CURVE[np.isclose(CURVE.threshold * 20, np.round(CURVE.threshold * 20))].set_index('threshold').round(4))
_bt = CURVE.threshold[CURVE.tune_F05.idxmax()]
fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
ax[0].plot(CURVE.threshold, CURVE.tune_F05, label='tune'); ax[0].plot(CURVE.threshold, CURVE.hold_F05, label='hold')
ax[0].axvline(_bt, ls='--', c='k'); ax[0].set_title('threshold vs macro F0.5'); ax[0].legend(); ax[0].grid(alpha=.3)
for c in ['precision', 'recall', 'singleton_acc', 'false_merge_rate']:
    ax[1].plot(CURVE.threshold, CURVE[c], label=c)
ax[1].set_title('hold: pair precision / recall, singleton accuracy, false-merge rate'); ax[1].legend(); ax[1].grid(alpha=.3)
plt.show()
_region = CURVE[CURVE.tune_F05 >= CURVE.tune_F05.max() - 0.002].threshold
print(f'best global threshold (tune) = {_bt:.2f}; operating region within 0.002 F0.5: [{_region.min():.2f}, {_region.max():.2f}]')

# per-source threshold surface
_g = np.round(np.clip(np.arange(_bt - 0.2, _bt + 0.201, 0.04), 0.05, 0.97), 2)
_g = np.unique(_g)
SURF = np.array([[EV_T.metrics(dict(base_prm(_bt), t_s2=float(a), t_s3=float(b)))['F05'] for b in _g] for a in _g])
plt.figure(figsize=(7, 5.5))
plt.imshow(SURF, origin='lower', cmap='magma', extent=[_g[0], _g[-1], _g[0], _g[-1]], aspect='auto')
plt.colorbar(label='tune macro F0.5'); plt.xlabel('threshold S1–S3'); plt.ylabel('threshold S1–S2')
plt.title('Does each source pair need its own threshold?'); plt.show()
_ia, _ib = np.unravel_index(SURF.argmax(), SURF.shape)
finding('Single vs per-source thresholds',
        f'best single t={_bt:.2f} (tune F0.5 {CURVE.tune_F05.max():.4f}); best (t_S2, t_S3)=({_g[_ia]:.2f}, {_g[_ib]:.2f}) '
        f'(tune F0.5 {SURF.max():.4f})',
        'S2 and S3 have different noise profiles (casing, scripts, domains), so their score calibration can differ.',
        'Per-source thresholds are kept only if they beat the single threshold in the staged search of §17.')

# %% [markdown]
# ## 17. Multi-match decision logic
# The output is a *set* per S1, so the rule is searched for in stages on **tune**, each stage accepted only if macro F0.5 improves:
# 1. global threshold
# 2. **exclusivity**: each S2/S3 record goes only to its best-scoring S1. The ground truth guarantees at most one owner, and the stage is preferred unless it hurts.
# 3. per-source thresholds
# 4. **anchor + expansion**: an S1 receives matches only if its best pair ≥ `t_anchor`; once anchored, further records need only the (lower) expansion threshold. This is the principled singleton gate.
# 5. relative margin to the best pair
#
# Top-1 selection is evaluated for comparison only, because the problem is one-to-many.

# %%
def tune_full(ev):
    H = []

    def run(prm, stage):
        m = ev.metrics(prm)
        H.append(dict(stage=stage, **prm, F05=m['F05'], P=m['P_micro'], R=m['R_micro'], singleton_acc=m['singleton_acc']))
        return m['F05']
    best, bf = None, -1.0
    for t in T_GRID:
        f = run(base_prm(t), '1 global')
        if f > bf:
            best, bf = base_prm(t), f
    be, bfe = None, -1.0
    for t in T_GRID:
        prm = dict(base_prm(t), exclusive=True)
        f = run(prm, '2 exclusive')
        if f > bfe:
            be, bfe = prm, f
    if bfe >= bf - 1e-4:
        best, bf = be, bfe
    base3 = dict(best)
    grid = np.unique(np.round(np.clip(np.arange(base3['t_s2'] - 0.2, base3['t_s2'] + 0.201, 0.02), 0.05, 0.97), 2))
    for a in grid:
        for b in grid:
            prm = dict(base3, t_s2=float(a), t_s3=float(b))
            f = run(prm, '3 per-source')
            if f > bf + 1e-6:
                best, bf = prm, f
    base4 = dict(best)
    lo = max(base4['t_s2'], base4['t_s3'])
    for ta in np.round(np.arange(lo, 0.99, 0.01), 2):
        for d in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
            prm = dict(base4, t_anchor=float(ta), t_s2=float(max(0.05, base4['t_s2'] - d)), t_s3=float(max(0.05, base4['t_s3'] - d)))
            f = run(prm, '4 anchor+expansion')
            if f > bf + 1e-6:
                best, bf = prm, f
    base5 = dict(best)
    for r in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        prm = dict(base5, rel=r)
        f = run(prm, '5 relative margin')
        if f > bf + 1e-6:
            best, bf = prm, f
    return best, bf, pd.DataFrame(H)


with Timer('decision-rule search (tune)'):
    FINAL_PRM, _ftune, RULE_HIST = tune_full(EV_T)
print('FINAL decision rule:', FINAL_PRM, f'(tune F0.5 {_ftune:.4f})')
display(RULE_HIST.groupby('stage').F05.max().round(4).to_frame('best tune F0.5 in stage'))

_stage_best = {s: RULE_HIST[RULE_HIST.stage == s].sort_values('F05').iloc[-1] for s in RULE_HIST.stage.unique()}
_keys = ['t_s2', 't_s3', 't_anchor', 'exclusive', 'rel', 'top_n']
_top1_t = max(T_GRID, key=lambda t: EV_T.metrics(dict(base_prm(t), top_n=1))['F05'])
RULES = {
    'fixed t = 0.5 (naive)': base_prm(0.5),
    'tuned global threshold': {k: _stage_best['1 global'][k] for k in _keys},
    'top-1 only (tuned t)': dict(base_prm(_top1_t), top_n=1),
    'FINAL staged rule': FINAL_PRM,
}
RULE_TABLE = pd.DataFrame({k: EV_H.metrics(v) for k, v in RULES.items()}).T[
    ['F05', 'P_micro', 'R_micro', 'singleton_acc', 'false_merge_rate', 'avg_pred_per_S1']]
display(RULE_TABLE.round(4))
add_experiment('E6', 'Threshold/decision tuning (per-source, exclusivity, anchor+expansion) vs fixed 0.5 '
               f'(0.5 → {RULE_TABLE.loc["fixed t = 0.5 (naive)", "F05"]:.4f})',
               EV_H.metrics(FINAL_PRM), 'use FINAL staged rule')

# behaviour by true multiplicity on hold
_ok = EV_H.select(FINAL_PRM)
_npred = np.bincount(EV_H.qi[_ok], minlength=NQ)[FOLD_Q['hold']]
_ntrue = N_TRUE_Q[FOLD_Q['hold']]
_tp = np.bincount(EV_H.qi[_ok], weights=EV_H.y[_ok], minlength=NQ)[FOLD_Q['hold']]
with np.errstate(divide='ignore', invalid='ignore'):
    _p = np.where(_npred > 0, _tp / np.maximum(_npred, 1), 1.0); _r = np.where(_ntrue > 0, _tp / np.maximum(_ntrue, 1), 1.0)
    _f = np.where((_npred == 0) & (_ntrue == 0), 1.0, np.where(_tp > 0, 1.25 * _p * _r / (0.25 * _p + _r), 0.0))
MULT = pd.DataFrame({'true': np.minimum(_ntrue, 8), 'pred': np.minimum(_npred, 8), 'F05': _f})
display(MULT.groupby('true').agg(S1=('F05', 'size'), mean_F05=('F05', 'mean'), mean_pred=('pred', 'mean')).round(3))
display(pd.crosstab(MULT.true, MULT.pred, rownames=['#true (capped 8)'], colnames=['#predicted (capped 8)']))
finding('Multi-match rule',
        f"hold F0.5: top-1 {RULE_TABLE.loc['top-1 only (tuned t)', 'F05']:.4f} vs tuned global {RULE_TABLE.loc['tuned global threshold', 'F05']:.4f} "
        f"vs final {RULE_TABLE.loc['FINAL staged rule', 'F05']:.4f}; singleton accuracy {RULE_TABLE.loc['FINAL staged rule', 'singleton_acc']:.4f}",
        'Most S1 entities have 2-6 true records; top-1 caps recall. Anchor+expansion separates "is there a match at all" '
        '(singleton control) from "which records belong".',
        f'FINAL rule = {FINAL_PRM}')

# %% [markdown]
# ## 18. Error analysis on hold
# False positives and false negatives under the final rule are assigned to a taxonomy, first matching rule wins.
# Missed true pairs that were never candidates are attributed to blocking, and their features are computed separately.
# Each category is paired with a concrete fix.

# %%
_hold_rows = EV_H.rows
_sel_rows = _hold_rows[_ok]
FP = FE.iloc[_sel_rows[C_Y[_sel_rows] == 0]].copy()
FP['qi'], FP['pi'] = C_QI[_sel_rows[C_Y[_sel_rows] == 0]], C_PI[_sel_rows[C_Y[_sel_rows] == 0]]
_sel_keys = CAND_KEYS[_sel_rows]
_hold_gt = GT_Q[GT_Q_FOLD == 'hold']
_hold_gt_keys = _hold_gt.qi.to_numpy(np.int64) * NP + _hold_gt.pi.to_numpy(np.int64)
_fn = _hold_gt[~np.isin(_hold_gt_keys, _sel_keys)].sort_values('qi')
_fn_in_cand = np.isin(_fn.qi.to_numpy(np.int64) * NP + _fn.pi.to_numpy(np.int64), CAND_KEYS)
FN = pair_features(_fn.qi.to_numpy(), _fn.pi.to_numpy(), Q, P, Qm, Pm, RET, Qe, Pe)
FN['qi'], FN['pi'], FN['in_candidates'] = _fn.qi.to_numpy(), _fn.pi.to_numpy(), _fn_in_cand


def _v(F, c, fill=0):
    return F[c].fillna(fill).to_numpy()


fp_rules = [
    ('normalisation collision', (_v(FP, 'n_skel_eq') == 1) & (_v(FP, 'n_basic_eq') == 0) & (_v(FP, 'n_tset') < 0.8)),
    ('same name, other branch/location', (_v(FP, 'n_core_eq') == 1) & (_v(FP, 'a_tset', 1) < 0.6)),
    ('common business name', _v(FP, 'p_nf_s1') >= np.log1p(5)),
    ('same address, different business', (_v(FP, 'a_tset') >= 0.9) & (_v(FP, 'n_tset') < 0.6)),
    ('missing address (name-only evidence)', (_v(FP, 'q_a_missing') + _v(FP, 'p_a_missing')) > 0),
    ('only common tokens shared', (_v(FP, 'n_tok_rare_shared') == 0) & (_v(FP, 'n_tok_shared') > 0)),
]
fn_rules = [
    ('blocking miss (never a candidate)', ~FN.in_candidates.to_numpy()),
    ('missing address', _v(FN, 'p_a_missing') > 0),
    ('transliteration', _v(FN, 'p_f_nonascii') > 0),
    ('domain / handle name', _v(FN, 'p_f_domain') > 0),
    ('word reorder', (_v(FN, 'n_tsort') >= 0.95) & (_v(FN, 'n_ratio') < 0.95)),
    ('typo', _v(FN, 'n_ratio') >= 0.8),
    ('abbreviation / acronym', (_v(FN, 'n_acronym') == 1) | (_v(FN, 'n_tset') >= 0.8)),
    ('DBA / trade name', (_v(FN, 'n_tset') < 0.5) & (_v(FN, 'a_tset') >= 0.8)),
]
FIXES = {
    'normalisation collision': 'down-weight skeleton equality when basic names differ; require address agreement',
    'same name, other branch/location': 'stronger number/postal conflict veto; per-S1 exclusivity already applied',
    'common business name': 'raise anchor threshold when name commonness is high (commonness × similarity interaction)',
    'same address, different business': 'name-similarity floor for address-only matches (multi-tenant buildings)',
    'missing address (name-only evidence)': 'separate threshold for pairs without address evidence',
    'only common tokens shared': 'more weight on rare-token overlap; IDF-weighted Jaccard already present',
    'blocking miss (never a candidate)': 'larger K / extra pass (char-n-gram LSH, acronym key) — check recall@K curve',
    'missing address': 'name-only acceptance needs high name similarity + distinctive name',
    'transliteration': 'richer transliteration (schwa deletion, learned char map)',
    'domain / handle name': 'squash/partial-ratio key already; add domain→token segmentation',
    'word reorder': 'token-sort features exist; check threshold for these',
    'typo': 'char n-gram + skeleton similarities; encoder features',
    'abbreviation / acronym': 'acronym blocking key; learned abbreviation map',
    'DBA / trade name': 'address-driven acceptance when numbers/postal agree',
    'severe corruption / other': 'accept loss (precision-weighted metric)',
    'other': 'inspect examples',
}


def taxonomy(F, rules, default):
    lab = np.full(len(F), default, dtype=object)
    done = np.zeros(len(F), bool)
    for name, m in rules:
        m = np.asarray(m, bool) & ~done
        lab[m] = name
        done |= m
    return lab


FP['category'] = taxonomy(FP, fp_rules, 'other')
FN['category'] = taxonomy(FN, fn_rules, 'severe corruption / other')
ERR = pd.concat([
    FP.category.value_counts().rename('count').to_frame().assign(kind='false positive'),
    FN.category.value_counts().rename('count').to_frame().assign(kind='false negative')])
ERR['share_within_kind_%'] = 100 * ERR['count'] / ERR.groupby('kind')['count'].transform('sum')
ERR['proposed_fix'] = [FIXES.get(i, '') for i in ERR.index]
display(ERR.round(2))
for kind, F in [('FALSE POSITIVE', FP), ('FALSE NEGATIVE', FN)]:
    for cat in F.category.value_counts().index[:4]:
        ex = F[F.category == cat].head(2)
        for qi_, pi_ in zip(ex.qi, ex.pi):
            print(f'[{kind} | {cat}]\n   S1 : {Q.business_name.iloc[qi_]!r} | {Q.business_address.iloc[qi_]!r}\n'
                  f'   S{P.src.iloc[pi_]} : {P.business_name.iloc[pi_]!r} | {P.business_address.iloc[pi_]!r}')
ERR.to_csv(os.path.join(ART_DIR, 'error_taxonomy.csv'))

# %% [markdown]
# ## 19. Frozen final pipeline
# Everything selected above is frozen into `FINAL`: normalisation maps, blocking passes and K, pruning, feature list, scorer and decision rule.
# The test run in §20 calls **the same functions** (`prepare_frame`, `build_matrices`, `fit_idf`, `key_pass`/`sparse_passes`/`dense_pass`,
# `union_candidates`, `compute_features`, `predict_model`/`stage2_chunk`, `decide`), so validation and inference cannot drift apart.

# %%
FINAL = dict(
    passes=SELECTED, K={p: K_SEL[p] for p in SELECTED}, prune=PRUNE, scorer=FINAL_SCORER, stage1=STAGE1_MODEL,
    stage2=(dict(stage1=STAGE2['stage1'], keep_feats=STAGE2['keep_feats'], feats=STAGE2['feats']) if USE_STAGE2 else None),
    use_emb=bool(USE_EMB), decision=FINAL_PRM,
    validation=dict(hold=EV_H.metrics(FINAL_PRM), oracle_hold=ORACLE['hold'], blocking=FINAL_BLOCK),
    cfg=CFG,
)
with open(os.path.join(ART_DIR, 'final_pipeline.json'), 'w') as fh:
    json.dump(FINAL, fh, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
with open(os.path.join(ART_DIR, 'learned_maps.json'), 'w') as fh:
    json.dump(dict(NAME_MAP=NAME_MAP, ADDR_MAP=ADDR_MAP), fh, indent=1)
np.save(os.path.join(ART_DIR, 'hold_final_scores.npy'), FINAL_P['hold'])
print(f"""
FINAL PIPELINE
  normalisation : translit + basic/core/squash/skeleton names ({len(NAME_MAP)} learned name synonyms),
                  basic/canon addresses ({len(ADDR_MAP)} learned address synonyms), postal/number anchors
  blocking      : {' + '.join(PASS_NAMES[p] + (f' (K={K_SEL[p]})' if K_SEL[p] else '') for p in SELECTED)}  | safe pruning: {PRUNE}
  features      : {len(MODELS[STAGE1_MODEL]['feats']) if MODELS[STAGE1_MODEL]['kind'] != 'ens' else len(FEATS_BASE)} stage-1 features
  scorer        : {FINAL_SCORER}  (stage-1 = {STAGE1_MODEL})
  decision      : {FINAL_PRM}
  hold macro F0.5 = {FINAL['validation']['hold']['F05']:.4f}  (oracle ceiling given candidates {ORACLE['hold']['F05']:.4f})
""")

# %% [markdown]
# ## 20. Test inference
# Training-phase memory is released first. The test files are then reloaded and pushed through the frozen pipeline.
# The IDF statistics are **recomputed on the test pool**. This is label-free and matches the training procedure, which computed them on the train pool.
# Candidates are generated for **every** test S1, France included. Features and scores are streamed in chunks, and the exclusivity rule runs across all test S1.

# %%
for _name in ['FE', 'Qm', 'Pm', 'PASS_DFS', 'CAND', 'CAND_KEYS', 'POSF', 'HNF', 'Qe', 'Pe', 'FP', 'FN', 'SG', 'sg',
              'S1', 'P', 'Q', 'gt_pairs', 'PASS_KEYS', '_fit_keys', 'cur', 'EV_T', 'EV_H']:
    if _name in globals():
        del globals()[_name]
KEY_CACHE.clear()
gc.collect()
log('training-phase memory released')

with Timer('load + prepare test'):
    T1 = load_tsv(PATHS['test_s1'])
    TP = build_pool(load_tsv(PATHS['test_s2']), load_tsv(PATHS['test_s3']))
    T1['country_key'] = T1.country.str.strip().str.lower()
    prepare_frame(T1)
    prepare_frame(TP)
    add_freq_features(T1, TP)
    NQT, NPT = len(T1), len(TP)
    T_SLICES = country_slices(TP)
    T_CK = T1.country_key.astype(object).to_numpy()
    print(f'test S1={NQT:,}  test pool={NPT:,}  countries in pool: { {c: e - s for c, (s, e) in T_SLICES.items()} }')
    print('test S1 countries:', pd.Series(T_CK).value_counts().to_dict())

Tm = build_matrices(T1, 'test S1')
TPm = build_matrices(TP, 'test pool')
with Timer('test pool IDF'):
    RT = fit_idf(TPm)
TQe = TPe = None
if ENCODER is not None and ('G' in SELECTED or USE_EMB or (USE_STAGE2 and FINAL['stage2'] and 'emb' in FINAL['stage2']['stage1'])):
    with Timer('test embeddings (GPU)'):
        TQe = embed_all(ENCODER, Tm, RT, NQT)
        TPe = embed_all(ENCODER, TPm, RT, NPT)

with Timer('test candidate generation'):
    T_ALL = np.arange(NQT)
    TPASS = {}
    for p in SELECTED:
        if p in KEY_PASSES:
            TPASS[p] = key_pass(p, T1, TP, T_ALL, 'test')
    _ranked = tuple(p for p in SELECTED if p in ('C', 'D', 'F'))
    if _ranked:
        TPASS.update(sparse_passes(Tm, TPm, T_ALL, T_CK, T_SLICES, RT, k=max(K_SEL[p] for p in _ranked), which=_ranked))
    if 'G' in SELECTED:
        TPASS['G'] = dense_pass(TQe, TPe, T_ALL, T_CK, T_SLICES, K_SEL['G'])
    TCAND = union_candidates(TPASS, {p: K_SEL[p] for p in SELECTED}, NPT)
    del TPASS
    gc.collect()
    print(f'test candidates (before pruning): {len(TCAND):,} ({len(TCAND) / NQT:.1f} per S1)')

with Timer('test features + scoring (streamed)'):
    if USE_STAGE2:
        _s1 = FINAL['stage2']['stage1']
        _p1, _keep, _KEEPDF = compute_features(
            TCAND, T1, TP, Tm, TPm, RT, SELECTED, TQe, TPe, prune=PRUNE,
            scorer=lambda F: predict_model(_s1, lambda feats: F[feats].to_numpy(np.float32)),
            keep_cols=FINAL['stage2']['keep_feats'])
        TCAND = TCAND[_keep].reset_index(drop=True)
        _qi, _pi = TCAND.qi.to_numpy(), TCAND.pi.to_numpy()
        _s3 = (TP.src.to_numpy()[_pi] == 3).astype(np.int64)
        PT = np.empty(len(TCAND), np.float32)
        _m2 = STAGE2['model']
        for s, e in group_chunks(_qi, 2_000_000):
            S = pd.concat([stage2_chunk(_qi[s:e], _pi[s:e], _s3[s:e], _p1[s:e], TP, TPm),
                           _KEEPDF.iloc[s:e].reset_index(drop=True)], axis=1)
            PT[s:e] = _m2.predict(S[FINAL['stage2']['feats']].to_numpy(np.float32), num_iteration=_m2.best_iteration)
        del _KEEPDF, _p1
    else:
        PT, _keep, _ = compute_features(
            TCAND, T1, TP, Tm, TPm, RT, SELECTED, TQe, TPe, prune=PRUNE,
            scorer=lambda F: predict_model(FINAL_SCORER, lambda feats: F[feats].to_numpy(np.float32)))
        TCAND = TCAND[_keep].reset_index(drop=True)
    gc.collect()

with Timer('test decision rule'):
    _qi, _pi = TCAND.qi.to_numpy(), TCAND.pi.to_numpy()
    _s3b = TP.src.to_numpy()[_pi] == 3
    _bp = best_for_pool(_pi, PT)
    _rk = (pd.Series(PT).groupby(_qi).rank(ascending=False, method='first').to_numpy()
           if FINAL_PRM.get('top_n') else np.ones(len(PT)))
    TSEL = decide(_qi, PT, _s3b, _bp, _rk, FINAL_PRM, NQT)
    print(f'final test matches: {TSEL.sum():,} pairs; S1 with >=1 match: {len(np.unique(_qi[TSEL])):,} / {NQT:,}')
    _cc = pd.DataFrame({'ck': T_CK[_qi[TSEL]]}).ck.value_counts()
    print('matched pairs per country:', _cc.to_dict())


def write_id_lists(path, header, q_ids, qi_sorted, pi_sorted, pool_ids):
    """One row per S1 (in test_source1 order), comma-joined S2/S3 ids, empty when none. qi must be sorted."""
    n = len(q_ids)
    starts = np.searchsorted(qi_sorted, np.arange(n), side='left')
    ends = np.searchsorted(qi_sorted, np.arange(n), side='right')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\t'.join(header) + '\n')
        for b in range(0, n, 200_000):
            lines = [q_ids[q] + '\t' + ','.join(pool_ids[pi_sorted[starts[q]:ends[q]]]) for q in range(b, min(n, b + 200_000))]
            fh.write('\n'.join(lines) + '\n')


with Timer('write submission files'):
    OUT_MATCH = os.path.join(CFG['WORK_DIR'], 'matching_results.tsv')
    OUT_CAND = os.path.join(CFG['WORK_DIR'], 'candidate_pairs.tsv')
    _qids = T1.entity_id.astype(object).to_numpy()
    _pids = TP.entity_id.astype(object).to_numpy()
    write_id_lists(OUT_CAND, ['source1_entity_id', 'candidate_entity_ids'], _qids, _qi, _pi, _pids)
    write_id_lists(OUT_MATCH, ['source1_entity_id', 'matched_entity_ids'], _qids, _qi[TSEL], _pi[TSEL], _pids)
    os.makedirs(os.path.join(CFG['WORK_DIR'], 'output'), exist_ok=True)
    for f in (OUT_MATCH, OUT_CAND):          # also expose the package layout output/ (hard link, no extra disk)
        dst = os.path.join(CFG['WORK_DIR'], 'output', os.path.basename(f))
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.link(f, dst)
        except OSError:
            pass
for f in (OUT_MATCH, OUT_CAND):
    print(f'{f}: exists={os.path.exists(f)} size={os.path.getsize(f) / 1e6:,.1f} MB')

# %% [markdown]
# ## 21. Submission validation
# There are two independent checks:
# 1. An in-notebook validator that re-reads both files and checks every rule: header, one row per test S1, only S2/S3 ids that exist in the test pool,
#    no duplicates, matches ⊆ candidates, and no S1 ids.
# 2. The **official competition validator** (`student_resource/utils/validate_submission.py`, stdlib only), embedded verbatim and run with `--check-ids`.
#    Any validator script found under `/kaggle/input` is run as well.

# %%
def validate_submission(match_path, cand_path, s1_ids, pool_ids):
    """Memory-light re-read of both files: the matching file is held in memory (small), the candidate file is streamed."""
    s1_set, pool_set = set(s1_ids), set(pool_ids)
    report = {}
    M, dup_m, n_m = {}, 0, 0
    bad_prefix = dup_in_list = unknown = 0
    with open(match_path, encoding='utf-8') as fh:
        report['matching header ok'] = fh.readline().rstrip('\n').split('\t') == ['source1_entity_id', 'matched_entity_ids']
        for line in fh:
            if not line.strip():
                continue
            s1, _, rest = line.rstrip('\n').partition('\t')
            ids = rest.split(',') if rest else []
            n_m += 1
            dup_m += s1 in M
            M[s1] = ids
            dup_in_list += len(ids) != len(set(ids))
            for i in ids:
                if not (i.startswith('S2-') or i.startswith('S3-')):
                    bad_prefix += 1
                elif i not in pool_set:
                    unknown += 1
    seen_c, dup_c, n_c, cand_bad, cand_dup, not_subset, with_c, tot_c = set(), 0, 0, 0, 0, 0, 0, 0
    with open(cand_path, encoding='utf-8') as fh:
        report['candidate header ok'] = fh.readline().rstrip('\n').split('\t') == ['source1_entity_id', 'candidate_entity_ids']
        for line in fh:
            if not line.strip():
                continue
            s1, _, rest = line.rstrip('\n').partition('\t')
            ids = rest.split(',') if rest else []
            n_c += 1
            dup_c += s1 in seen_c
            seen_c.add(s1)
            cs = set(ids)
            cand_dup += len(ids) != len(cs)
            cand_bad += sum(1 for i in ids if i not in pool_set or not (i.startswith('S2-') or i.startswith('S3-')))
            not_subset += bool(set(M.get(s1, [])) - cs)
            with_c += bool(ids)
            tot_c += len(ids)
    report['no duplicate S1 rows'] = dup_m == 0 and dup_c == 0
    report['every test S1 exactly once (matching)'] = set(M) == s1_set and n_m == len(s1_set)
    report['every test S1 exactly once (candidates)'] = seen_c == s1_set and n_c == len(s1_set)
    report['only S2/S3 ids in matches'] = bad_prefix == 0
    report['no duplicate ids within a list'] = dup_in_list == 0 and cand_dup == 0
    report['all matched ids exist in test pool'] = unknown == 0
    report['all candidate ids exist in test pool (S2/S3 only)'] = cand_bad == 0
    report['matches subset of candidates'] = not_subset == 0
    stats = dict(n_S1=len(s1_set), with_candidates=with_c, with_matches=sum(1 for v in M.values() if v),
                 singletons=sum(1 for v in M.values() if not v), total_candidate_pairs=tot_c,
                 total_matches=sum(len(v) for v in M.values()))
    return all(report.values()), report, stats


OFFICIAL_VALIDATOR_SRC = r'''__OFFICIAL_VALIDATOR__'''

for _name in ['Tm', 'TPm', 'TQe', 'TPe', 'TCAND', 'PT', '_bp', '_rk', '_s3b']:
    if _name in globals():
        del globals()[_name]
gc.collect()
with Timer('submission validation'):
    VALID_OK, VALID_REPORT, SUB_STATS = validate_submission(OUT_MATCH, OUT_CAND, T1.entity_id.astype(object).tolist(),
                                                            TP.entity_id.astype(object).tolist())
    display(pd.Series(VALID_REPORT, name='check passed').to_frame())
    _vpath = os.path.join(CFG['WORK_DIR'], 'validate_submission.py')
    with open(_vpath, 'w', encoding='utf-8') as fh:
        fh.write(OFFICIAL_VALIDATOR_SRC)
    OFFICIAL_RC = None
    _tdir = os.path.dirname(PATHS['test_s1'])
    # two memory-safe runs (as the validator docstring recommends): id-existence on the scored file, then all rules incl. candidates
    _runs = [['--matching', OUT_MATCH, '--candidate', os.path.join(CFG['WORK_DIR'], '__none__.tsv'), '--test-dir', _tdir, '--check-ids'],
             ['--matching', OUT_MATCH, '--candidate', OUT_CAND, '--test-dir', _tdir]]
    for vp in [_vpath] + [v for v in VALIDATORS if v.endswith('validate_submission.py')]:
        for args in _runs:
            try:
                r = subprocess.run([sys.executable, vp] + args, capture_output=True, text=True, timeout=3600)
                flags = ' '.join(a for a in args if a in ('--check-ids',)) or '(with candidates)'
                print(f'--- official validator {vp} {flags}: exit code {r.returncode}')
                print(r.stdout[-3000:], r.stderr[-2000:])
                OFFICIAL_RC = r.returncode if OFFICIAL_RC is None else max(OFFICIAL_RC, r.returncode)
            except Exception as e:
                print('validator could not run:', e)
SUBMISSION_PASS = VALID_OK and (OFFICIAL_RC in (None, 0))
print(f"""
Submission validation: {'PASS' if SUBMISSION_PASS else 'FAIL'}
Number of S1 entities        : {SUB_STATS['n_S1']:,}
Number with candidates       : {SUB_STATS['with_candidates']:,}
Number with predicted matches: {SUB_STATS['with_matches']:,}
Number predicted as singleton: {SUB_STATS['singletons']:,}
Total candidate pairs        : {SUB_STATS['total_candidate_pairs']:,}
Total final matches          : {SUB_STATS['total_matches']:,}
""")

# %% [markdown]
# ## 22. Compute and memory report

# %%
TIME_TABLE = pd.DataFrame(TIMINGS)
display(TIME_TABLE)
print(f'total runtime {(time.time() - T0) / 60:.1f} min; current RSS {rss_gb():.1f} GB; GPU used: {USE_GPU}')
TIME_TABLE.to_csv(os.path.join(ART_DIR, 'timings.csv'), index=False)

# %% [markdown]
# ## 23. Final competition report
# This section prints the report and experiment table, and saves all findings. It also writes a pre-filled methodology document (`Documentation_template_filled.md`) for the submission package.

# %%
EXP_TABLE = pd.DataFrame(EXPERIMENTS)
EXP_TABLE['_o'] = EXP_TABLE.Experiment.str[1:].astype(int)
EXP_TABLE = EXP_TABLE.sort_values('_o').drop(columns='_o').reset_index(drop=True)
EXP_TABLE.to_csv(os.path.join(ART_DIR, 'experiments.csv'), index=False)
pd.DataFrame(FINDINGS).to_csv(os.path.join(ART_DIR, 'findings.csv'), index=False)
_hold = FINAL['validation']['hold']
try:
    EXP_MD = EXP_TABLE.to_markdown(index=False)
except Exception:
    EXP_MD = EXP_TABLE.to_string(index=False)
REPORT = f"""
DATASET
-------
TRAIN S1: {NS1:,}
TRAIN S2+S3 (pool): {NP:,}
TEST  S1 / pool: {NQT:,} / {NPT:,}
SINGLETON RATE (train): {(N_TRUE_ALL == 0).mean():.2%}

BLOCKING
--------
FINAL BLOCKING RECALL (validation sample): {FINAL_BLOCK['recall']:.4f}  (hold {FINAL_BLOCK['recall_hold']:.4f})
AVERAGE CANDIDATES/S1: {FINAL_BLOCK['avg_per_S1']:.1f}  (test: {SUB_STATS['total_candidate_pairs'] / max(NQT, 1):.1f} after pruning={PRUNE})
CANDIDATE REDUCTION: {FINAL_BLOCK['reduction_ratio']:.6f}

MODEL
-----
MODEL: {FINAL_SCORER} (stage-1: {STAGE1_MODEL})
FEATURE COUNT: {len(FEATS_ALL if USE_EMB else FEATS_BASE)} stage-1{f" + {len(STAGE2['feats'])} stage-2" if USE_STAGE2 else ''}

VALIDATION (hold fold, macro F0.5 per S1)
----------
PRECISION (pair-level): {_hold['P_micro']:.4f}
RECALL    (pair-level): {_hold['R_micro']:.4f}
F0.5: {_hold['F05']:.4f}   (oracle ceiling {ORACLE['hold']['F05']:.4f}; singleton acc {_hold['singleton_acc']:.4f})

FINAL PIPELINE
--------------
NORMALISATION: Unicode-name transliteration, basic/core/squash/skeleton names, legal-form & learned synonym maps, canonical addresses, number/postal anchors
BLOCKING: {' + '.join(PASS_NAMES[p] for p in SELECTED)} (within country) {'+ safe pruning' if PRUNE else ''}
FEATURES: fuzzy (rapidfuzz), TF-IDF/IDF-weighted token & char statistics, numeric/postal agreement, commonness, noise flags, within-S1 context{', learned embeddings' if USE_EMB else ''}
MODEL: {FINAL_SCORER}
THRESHOLD: t_S2={FINAL_PRM['t_s2']:.2f}, t_S3={FINAL_PRM['t_s3']:.2f}, anchor={FINAL_PRM['t_anchor']:.2f}
MULTI-MATCH RULE: all candidates above the source threshold (no top-1), relative margin={FINAL_PRM['rel']}, exclusivity={FINAL_PRM['exclusive']}
SINGLETON RULE: empty list unless the best pair of the S1 reaches the anchor threshold

SUBMISSION
----------
matching_results.tsv: {OUT_MATCH}
candidate_pairs.tsv: {OUT_CAND}
VALIDATION STATUS: {'PASS' if SUBMISSION_PASS else 'FAIL'}
"""
print(REPORT)
display(EXP_TABLE)
display(pd.DataFrame(FINDINGS)[['finding', 'action']])

DOC = f"""# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [fill in]
**Team Members:** [fill in]
**Submission Date:** {time.strftime('%Y-%m-%d')}

---

## 1. Executive Summary
Multi-pass, label-free blocking (exact keys + within-country TF-IDF retrieval{' + learned GPU encoder' if 'G' in SELECTED else ''}) feeds a
gradient-boosted pair classifier trained on hard (blocking) negatives{', re-ranked by a cross-source confirmation stage' if USE_STAGE2 else ''}; a decision rule tuned directly
for macro F0.5 (per-source thresholds, anchor/expansion singleton gate{', exclusivity of S2/S3 records' if FINAL_PRM['exclusive'] else ''}) produces the matches.
Hold-out macro F0.5 = {_hold['F05']:.4f}.

## 2. Methodology
### 2.1 Problem Analysis
""" + '\n'.join(f"- **{r['finding']}** — {r['evidence']}. → {r['action']}" for r in FINDINGS) + f"""

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (+ stacked reranker) + F0.5-optimised decision rule
**Core Innovation:** dependency-free Unicode transliteration + learned token maps; recall-driven greedy blocking union; anchor/expansion decision rule; cross-source confirmation.

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** {', '.join(PASS_NAMES[p] + (f' (K={K_SEL[p]})' if K_SEL[p] else '') for p in SELECTED)}
- **Candidate pairs generated (test):** {SUB_STATS['total_candidate_pairs']:,} ({SUB_STATS['total_candidate_pairs'] / max(NQT, 1):.1f} per S1)
- **How true matches were kept:** passes chosen greedily by fit-fold recall; validation recall {FINAL_BLOCK['recall']:.4f}, reduction ratio {FINAL_BLOCK['reduction_ratio']:.6f}.

## 4. Matching Model
**Features used:** name/address equality at several normalisation levels, rapidfuzz ratios (Levenshtein, Jaro-Winkler, token-set/sort, partial),
IDF-weighted token Jaccard/cosine, char-3gram cosine, rare-token overlap, number/postal agreement & conflict, name/address commonness, noise flags,
within-S1 rank/gap/z-score context{', learned-encoder cosine' if USE_EMB else ''}{', stage-2 cross-source agreement' if USE_STAGE2 else ''}.
**Model type:** {FINAL_SCORER}
**Threshold selection method:** staged search maximising macro F0.5 on a tune fold of S1 entities; reported on an untouched hold fold.

## 5. Results & Error Analysis
- **F_0.5 Score (macro, hold):** {_hold['F05']:.4f} (precision {_hold['P_micro']:.4f}, recall {_hold['R_micro']:.4f})
- **Common false positives:** """ + ', '.join(f'{i} ({r["share_within_kind_%"]:.0f}%)' for i, r in ERR[ERR.kind == 'false positive'].head(4).iterrows()) + """
- **Common false negatives:** """ + ', '.join(f'{i} ({r["share_within_kind_%"]:.0f}%)' for i, r in ERR[ERR.kind == 'false negative'].head(4).iterrows()) + f"""

## 6. Conclusion
Blocking recall sets the ceiling ({ORACLE['hold']['F05']:.4f} oracle F0.5); hard-negative training and an F0.5-tuned set-level decision rule
convert it into {_hold['F05']:.4f}. Only competition data is used; no external lookup, geocoding or pretrained external models.

## Appendix — Experiment log
""" + EXP_MD
try:
    with open(os.path.join(CFG['WORK_DIR'], 'Documentation_template_filled.md'), 'w', encoding='utf-8') as fh:
        fh.write(DOC)
except Exception as e:
    print('documentation not written:', e)
print('Artifacts in', ART_DIR, ':', sorted(os.listdir(ART_DIR)))
print('Final files:', OUT_MATCH, OUT_CAND)
