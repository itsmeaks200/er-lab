# %% [markdown]
# ## 10. Blocking experiments
# Every pass runs on **all** experiment S1 entities against the **full** train pool, restricted to the same country.
# Ranked passes (C, D, F, G) keep the top `TOPK_MAX` so that recall@K curves can be drawn and K chosen afterwards.
#
# **Block G** is a learned encoder that uses the GPU. It is a hashed char-3-gram/token *EmbeddingBag* followed by a small MLP, trained with an in-batch contrastive (InfoNCE) loss
# on **fit-fold positive pairs only**. It uses no pretrained weights and no external data, so there are no licence or download issues. Its embeddings also feed the
# `emb_cos` feature evaluated in §15.

# %%
ENC_FIELDS = ['nchar', 'ntok', 'nskel', 'atok', 'anum', 'abig']


def enc_inputs(M, rows, R):
    B, nf = 2 ** CFG['ENC_BUCKET_BITS'], len(ENC_FIELDS)
    X = None
    for fi, f in enumerate(ENC_FIELDS):
        A = M[f][rows]
        data = A.data if f == 'nchar' else A.data * R['IDF'][f][A.indices]
        A = sp.csr_matrix((data.astype(np.float32), (A.indices % B + fi * B).astype(np.int64), A.indptr.copy()),
                          shape=(A.shape[0], nf * B))
        A.sum_duplicates()
        A = sk_normalize(A, copy=False)
        X = A if X is None else X + A
    X = X.tocsr()
    return (torch.from_numpy(X.indices.astype(np.int64)), torch.from_numpy(X.indptr[:-1].astype(np.int64)),
            torch.from_numpy(X.data.astype(np.float32)))


if torch is not None:
    class HashEncoder(torch.nn.Module):
        def __init__(self, n_buckets, dim):
            super().__init__()
            self.emb = torch.nn.EmbeddingBag(n_buckets, dim, mode='sum', sparse=True)
            torch.nn.init.normal_(self.emb.weight, std=0.05)
            self.mlp = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, 2 * dim), torch.nn.GELU(),
                                           torch.nn.Linear(2 * dim, dim))

        def forward(self, idx, off, w):
            x = self.emb(idx, off, per_sample_weights=w)
            return torch.nn.functional.normalize(x + self.mlp(x), dim=-1)


def _dev(t):
    return [x.cuda(non_blocking=True) for x in t]


def train_encoder(Qm_, Pm_, R, pos_qi, pos_pi):
    B, nf, dim = 2 ** CFG['ENC_BUCKET_BITS'], len(ENC_FIELDS), CFG['ENC_DIM']
    model = HashEncoder(nf * B, dim).cuda()
    opt_s = torch.optim.SparseAdam(list(model.emb.parameters()), lr=CFG['ENC_LR'])
    opt_d = torch.optim.AdamW(model.mlp.parameters(), lr=1e-3)
    rng = np.random.RandomState(SEED)
    order = np.argsort(pos_qi, kind='stable')
    sq, spi = pos_qi[order], pos_pi[order]
    uq, start, counts = np.unique(sq, return_index=True, return_counts=True)
    bs, tau = CFG['ENC_BATCH'], CFG['ENC_TAU']
    for ep in range(CFG['ENC_EPOCHS']):
        pick = start + (rng.rand(len(uq)) * counts).astype(np.int64)      # one positive per S1 per epoch → no in-batch false negatives
        a_rows, b_rows = sq[pick], spi[pick]
        perm = rng.permutation(len(a_rows))
        tot, nb = 0.0, 0
        model.train()
        for s in range(0, len(perm) - bs + 1, bs):
            bi = perm[s:s + bs]
            ea = model(*_dev(enc_inputs(Qm_, a_rows[bi], R)))
            eb = model(*_dev(enc_inputs(Pm_, b_rows[bi], R)))
            logits = ea @ eb.T / tau
            lab = torch.arange(len(bi), device='cuda')
            loss = (torch.nn.functional.cross_entropy(logits, lab) + torch.nn.functional.cross_entropy(logits.T, lab)) / 2
            opt_s.zero_grad(); opt_d.zero_grad()
            loss.backward()
            opt_s.step(); opt_d.step()
            tot += loss.item(); nb += 1
        log(f'   encoder epoch {ep + 1}/{CFG["ENC_EPOCHS"]}: InfoNCE loss {tot / max(nb, 1):.4f}')
    model.eval()
    return model


@torch.no_grad() if torch is not None else (lambda f: f)
def embed_all(model, M, R, n, bs=16384):
    out = np.empty((n, CFG['ENC_DIM']), np.float16)
    for s in range(0, n, bs):
        e = min(n, s + bs)
        out[s:e] = model(*_dev(enc_inputs(M, slice(s, e), R))).half().cpu().numpy()
    return out


ENCODER, Qe, Pe = None, None, None
with Timer('blocking passes A/B/E/H (exact keys)'):
    ALL_Q = np.arange(NQ)
    PASS_DFS = {k: key_pass(k, Q, P, ALL_Q, 'train') for k in KEY_PASSES}
with Timer('blocking passes C/D/F (TF-IDF top-K)'):
    PASS_DFS.update(sparse_passes(Qm, Pm, ALL_Q, Q_CK, P_SLICES, RET, k=CFG['TOPK_MAX']))
if CFG['RUN_ENCODER'] and USE_GPU:
    with Timer('train learned encoder (GPU) + pass G'):
        _fp = GT_Q[Q_FOLD[GT_Q.qi.to_numpy()] == 'fit']
        ENCODER = train_encoder(Qm, Pm, RET, _fp.qi.to_numpy(), _fp.pi.to_numpy())
        Qe = embed_all(ENCODER, Qm, RET, NQ)
        Pe = embed_all(ENCODER, Pm, RET, NP)
        PASS_DFS['G'] = dense_pass(Qe, Pe, ALL_Q, Q_CK, P_SLICES, CFG['TOPK_MAX'])
        torch.save(ENCODER.state_dict(), os.path.join(ART_DIR, 'encoder.pt'))
else:
    print('Block G skipped (no GPU or RUN_ENCODER=False).')

POOL_SIZE_C = {c: e - s for c, (s, e) in P_SLICES.items()}
TOTAL_PAIRS = float(sum(POOL_SIZE_C.get(c, NP) for c in Q_CK))


def keys_of(d):
    return np.unique(d.qi.to_numpy(np.int64) * NP + d.pi.to_numpy(np.int64))


def cand_stats(name, keys, K=None):
    hit = np.isin(GT_Q_KEYS, keys)
    per_q = np.bincount((keys // NP).astype(np.int64), minlength=NQ)
    r = dict(strategy=name, K=K, candidate_pairs=len(keys), recall=hit.mean())
    for f in ['fit', 'tune', 'hold']:
        r[f'recall_{f}'] = hit[GT_Q_FOLD == f].mean()
    r.update(avg_per_S1=per_q.mean(), median_per_S1=np.median(per_q), p95_per_S1=np.percentile(per_q, 95),
             max_per_S1=per_q.max(), S1_without_candidates_pct=100 * (per_q == 0).mean(),
             reduction_ratio=1 - len(keys) / TOTAL_PAIRS, missed_true_pairs=int((~hit).sum()))
    return r


rows = [cand_stats(PASS_NAMES[p], keys_of(d if p in KEY_PASSES else d), None if p in KEY_PASSES else CFG['TOPK_MAX'])
        for p, d in PASS_DFS.items()]
BLOCK_TABLE = pd.DataFrame(rows).set_index('strategy')
display(BLOCK_TABLE.round(4))

# recall@K curves and K selection for ranked passes
RANKED = [p for p in PASS_DFS if p not in KEY_PASSES]
RK = {}
for p in RANKED:
    d = PASS_DFS[p]
    RK[p] = [np.isin(GT_Q_KEYS, keys_of(d[d['rank'] < K])).mean() for K in CFG['K_GRID']]
K_SEL = {p: None for p in KEY_PASSES if p in PASS_DFS}
for p in RANKED:
    target = CFG['K_RECALL_KEEP'] * RK[p][-1]
    K_SEL[p] = next(K for K, r in zip(CFG['K_GRID'], RK[p]) if r >= target)
plt.figure(figsize=(8, 4.2))
for p in RANKED:
    plt.plot(CFG['K_GRID'], RK[p], marker='o', label=f'{PASS_NAMES[p]} (K*={K_SEL[p]})')
plt.xlabel('K (top-K per S1)'); plt.ylabel('true-pair recall'); plt.title('recall@K per ranked blocking pass'); plt.legend(); plt.grid(alpha=.3)
plt.show()
print('Selected K per ranked pass:', {p: K_SEL[p] for p in RANKED})

# greedy multi-pass union selected on FIT-fold recall only
_fit_gt = GT_Q_FOLD == 'fit'
_is_fit_q = Q_FOLD == 'fit'
PASS_KEYS = {}
for p, d in PASS_DFS.items():
    dd = d if p in KEY_PASSES else d[d['rank'] < K_SEL[p]]
    PASS_KEYS[p] = keys_of(dd)
_fit_keys = {p: k[_is_fit_q[(k // NP).astype(np.int64)]] for p, k in PASS_KEYS.items()}
SELECTED, cur, cur_rec, hist = [], np.empty(0, np.int64), 0.0, []
while True:
    best = None
    for p in PASS_KEYS:
        if p in SELECTED:
            continue
        u = np.union1d(cur, _fit_keys[p])
        r = np.isin(GT_Q_KEYS[_fit_gt], u).mean()
        if best is None or r > best[1]:
            best = (p, r, u)
    if best is None or best[1] - cur_rec < CFG['GREEDY_MIN_GAIN']:
        break
    SELECTED.append(best[0]); cur, cur_rec = best[2], best[1]
    hist.append(dict(step=len(SELECTED), added=PASS_NAMES[best[0]], fit_recall=cur_rec,
                     fit_pairs_per_S1=len(cur) / max(_is_fit_q.sum(), 1)))
GREEDY = pd.DataFrame(hist)
display(GREEDY.round(4))
print('Selected blocking architecture:', [PASS_NAMES[p] for p in SELECTED])

with Timer('final candidate union'):
    CAND = union_candidates(PASS_DFS, {p: K_SEL[p] for p in SELECTED}, NP)
    CAND_KEYS = CAND.qi.to_numpy(np.int64) * NP + CAND.pi.to_numpy(np.int64)
FINAL_BLOCK = cand_stats('UNION ' + '+'.join(SELECTED), CAND_KEYS)
BLOCK_TABLE = pd.concat([BLOCK_TABLE, pd.DataFrame([FINAL_BLOCK]).set_index('strategy')])
display(BLOCK_TABLE.round(4))

_missed = GT_Q[~np.isin(GT_Q_KEYS, CAND_KEYS)].head(12)
print('Examples of true pairs lost by blocking:')
for qi_, pi_ in zip(_missed.qi, _missed.pi):
    print(f'  S1: {Q.business_name.iloc[qi_]!r:45} | {Q.business_address.iloc[qi_]!r:60}\n'
          f'  S{P.src.iloc[pi_]}: {P.business_name.iloc[pi_]!r:45} | {P.business_address.iloc[pi_]!r}')
_best_single = BLOCK_TABLE.drop(index=BLOCK_TABLE.index[-1]).recall.idxmax()
finding('Multi-pass blocking',
        f"best single pass '{_best_single}' recall {BLOCK_TABLE.loc[_best_single, 'recall']:.4f}; union of {len(SELECTED)} passes "
        f"recall {FINAL_BLOCK['recall']:.4f} with {FINAL_BLOCK['avg_per_S1']:.1f} candidates/S1 (p95 {FINAL_BLOCK['p95_per_S1']:.0f}), "
        f"reduction ratio {FINAL_BLOCK['reduction_ratio']:.6f}; {FINAL_BLOCK['missed_true_pairs']:,} true pairs lost",
        'Recall lost at blocking can never be recovered by the model — it is the ceiling of the whole system.',
        f"Use the greedy-selected union {SELECTED} with K={ {p: K_SEL[p] for p in SELECTED} } for validation AND test.")

# %% [markdown]
# ## 11. Blocking validation
# The blocking functions never received labels. Ground truth was used only for the recall numbers, and pass selection used **fit-fold** recall only.
# Below is the final architecture per fold, and the question from §5: do singletons look different once candidates exist?

# %%
rows = []
for f in ['fit', 'tune', 'hold']:
    qs = FOLD_Q[f]
    m = np.isin((CAND_KEYS // NP).astype(np.int64), qs)
    hit = np.isin(GT_Q_KEYS[GT_Q_FOLD == f], CAND_KEYS[m])
    rows.append(dict(fold=f, S1=len(qs), candidate_pairs=int(m.sum()), recall=hit.mean(),
                     reduction_ratio=1 - m.sum() / sum(POOL_SIZE_C.get(c, NP) for c in Q_CK[qs]),
                     avg_candidates_per_S1=m.sum() / len(qs)))
display(pd.DataFrame(rows).set_index('fold').round(5))

_fsc = PASS_DFS['F'].groupby('qi').score.max().reindex(range(NQ)).fillna(0).to_numpy()
_ncand = np.bincount(CAND.qi.to_numpy(), minlength=NQ)
SING = pd.DataFrame({'singleton': N_TRUE_Q == 0, 'n_candidates': _ncand, 'best_TFIDF_score': _fsc})
display(SING.groupby('singleton').describe().T.round(3))
finding('Singletons have weaker best candidates, not fewer candidates',
        f"median best TF-IDF score: singletons {SING[SING.singleton].best_TFIDF_score.median():.3f} vs matched "
        f"{SING[~SING.singleton].best_TFIDF_score.median():.3f}; median #candidates {SING[SING.singleton].n_candidates.median():.0f} vs "
        f"{SING[~SING.singleton].n_candidates.median():.0f}",
        'Blocking always returns look-alikes, so "has candidates" says nothing; the *strength of the best* candidate does.',
        'Singleton control = anchor threshold on the best pair score of each S1 (tuned in §17) + group-context features (gap to best).')
BLOCK_TABLE.to_csv(os.path.join(ART_DIR, 'blocking_table.csv'))

# %% [markdown]
# ## 12. Local validation protocol
# ```text
# train S1 entities ──hash split──► fit (60%) / tune (20%) / hold (20%)     (entity-level: no S1 or its GT in two folds)
#        │                         every S2/S3 record belongs to ≤1 S1 → no pair-label leakage across folds
#        ▼
# label-free blocking vs FULL pool ─► candidates ─► features ─► model fit on FIT ─► thresholds / rule on TUNE
#        ─► S1-level predictions on HOLD ─► macro F0.5 (all hold S1, singletons included)
# ```
# The learned token maps and the encoder use **fit** positives only. Early stopping and every threshold use **tune** only. **Hold** is touched only to report.
# The metric follows the official definition: an empty prediction on a singleton scores 1, and any prediction on a singleton scores 0.
# Micro (pair-level) precision and recall are reported next to it.

# %%
with Timer('candidate features (all folds)'):
    FE, _ = compute_features(CAND, Q, P, Qm, Pm, RET, SELECTED, Qe, Pe, prune=False)
C_QI, C_PI = CAND.qi.to_numpy(), CAND.pi.to_numpy()
C_Y = np.isin(CAND_KEYS, GT_Q_KEYS).astype(np.int8)
C_SRC3 = FE.src_s3.to_numpy() == 1
print(f'candidate feature table: {FE.shape}, positives {C_Y.mean():.3%}, memory {FE.memory_usage().sum() / 1e9:.2f} GB')

# ---- safe pruning zone (validated) ----
_sz = safe_zone(FE)
_loss = C_Y[_sz].sum() / len(GT_Q)
PRUNE = bool(_loss <= CFG['PRUNE_MAX_RECALL_LOSS'])
finding('Safe pruning zone',
        f'pairs with name AND address similarity < 0.2: {_sz.mean():.1%} of candidates, containing {C_Y[_sz].sum():,} true pairs '
        f'({100 * _loss:.3f}% of all true pairs)',
        'Pruning junk before scoring shrinks candidate_pairs.tsv / inference cost and sharpens group-context features.',
        f"{'APPLY' if PRUNE else 'DO NOT apply'} pruning (limit {100 * CFG['PRUNE_MAX_RECALL_LOSS']:.2f}% recall loss); "
        'the same rule runs inside the test feature stream.')
CTX_PREFIX = ('blk_',)
CTX_SUFFIX = ('_gap', '_rank', '_z')


def drop_context(F):
    return F.drop(columns=[c for c in F.columns if c.startswith(CTX_PREFIX) or c.endswith(CTX_SUFFIX) or c == 'q_ncand'])


if PRUNE:
    keep = ~_sz
    CAND = CAND[keep].reset_index(drop=True)
    CAND_KEYS, C_Y, C_SRC3 = CAND_KEYS[keep], C_Y[keep], C_SRC3[keep]
    FE = drop_context(FE[keep].reset_index(drop=True))
    add_context_features(FE, CAND.qi.to_numpy(), CAND.bits.to_numpy(), SELECTED)
    C_QI, C_PI = CAND.qi.to_numpy(), CAND.pi.to_numpy()
del _sz
gc.collect()
C_FOLD = Q_FOLD[C_QI]
ROWS = {f: np.flatnonzero(C_FOLD == f) for f in ['fit', 'tune', 'hold']}
C_CK = Q_CK[C_QI]


def s1_metrics(sel_qi, sel_y, n_true, q_eval):
    nq = len(n_true)
    n_pred = np.bincount(sel_qi, minlength=nq)[q_eval].astype(np.float64)
    tp = np.bincount(sel_qi, weights=sel_y.astype(np.float64), minlength=nq)[q_eval]
    nt = n_true[q_eval].astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 1.0)
        r = np.where(nt > 0, tp / np.maximum(nt, 1), 1.0)
        f = np.where((n_pred == 0) & (nt == 0), 1.0, np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0))
    sing = nt == 0
    return dict(F05=float(f.mean()), P_macro=float(p.mean()), R_macro=float(r.mean()),
                P_micro=float(tp.sum() / max(n_pred.sum(), 1)), R_micro=float(tp.sum() / max(nt.sum(), 1)),
                singleton_acc=float((n_pred[sing] == 0).mean()) if sing.any() else float('nan'),
                false_merge_rate=float(((n_pred - tp) > 0).mean()), avg_pred_per_S1=float(n_pred.mean()), n_S1=int(len(q_eval)))


def best_for_pool(pi, p):
    order = np.lexsort((-p, pi))
    ps = pi[order]
    first = np.r_[True, ps[1:] != ps[:-1]]
    flag = np.zeros(len(p), bool)
    flag[order[first]] = True
    return flag


def decide(qi, p, s3, bp, rk, prm, nq):
    """Decision rule: per-source thresholds, optional exclusivity (argmax S1 per pool record), optional top-n,
    anchor (an S1 gets matches only if its best pair >= t_anchor) and relative margin to the best pair."""
    ok = p >= np.where(s3, prm['t_s3'], prm['t_s2'])
    if prm.get('exclusive'):
        ok &= bp
    if prm.get('top_n'):
        ok &= rk <= prm['top_n']
    if prm.get('t_anchor', 0) > 0 or prm.get('rel', 0) > 0:
        mx = np.zeros(nq, np.float32)
        np.maximum.at(mx, qi[ok], p[ok])
        ok &= mx[qi] >= prm.get('t_anchor', 0)
        if prm.get('rel', 0) > 0:
            ok &= p >= prm['rel'] * mx[qi]
    return ok


class RuleEval:
    """Fast repeated evaluation of decision rules on one fold."""
    def __init__(self, rows, p, fold, min_p=0.02):
        qi, pi, p = C_QI[rows], C_PI[rows], np.asarray(p, np.float32)
        bp = best_for_pool(pi, p)
        rk = pd.Series(p).groupby(qi).rank(ascending=False, method='first').to_numpy()
        k = p >= min_p
        self.qi, self.p, self.y, self.s3, self.bp, self.rk = qi[k], p[k], C_Y[rows][k], C_SRC3[rows][k], bp[k], rk[k]
        self.rows = rows[k]
        self.q_eval = FOLD_Q[fold]

    def select(self, prm):
        return decide(self.qi, self.p, self.s3, self.bp, self.rk, prm, NQ)

    def metrics(self, prm):
        ok = self.select(prm)
        return s1_metrics(self.qi[ok], self.y[ok], N_TRUE_Q, self.q_eval)


T_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def base_prm(t):
    return dict(t_s2=float(t), t_s3=float(t), t_anchor=0.0, exclusive=False, rel=0.0, top_n=0)


def tune_global(ev):
    res = [(t, ev.metrics(base_prm(t))['F05']) for t in T_GRID]
    t, f = max(res, key=lambda x: x[1])
    return base_prm(t), f, pd.DataFrame(res, columns=['t', 'F05'])


def eval_scores(p_tune, p_hold, label):
    """Tune a global threshold on TUNE, report HOLD."""
    prm, f_t, _ = tune_global(RuleEval(ROWS['tune'], p_tune, 'tune'))
    m = RuleEval(ROWS['hold'], p_hold, 'hold').metrics(prm)
    print(f'  {label:45s} t*={prm["t_s2"]:.2f}  tune F0.5={f_t:.4f}  HOLD F0.5={m["F05"]:.4f}  P={m["P_micro"]:.4f}  R={m["R_micro"]:.4f}')
    return prm, m


ORACLE = {f: s1_metrics(C_QI[ROWS[f]][C_Y[ROWS[f]] == 1], np.ones(int(C_Y[ROWS[f]].sum())), N_TRUE_Q, FOLD_Q[f]) for f in ROWS}
print('Oracle (perfect model on these candidates) macro F0.5:', {f: round(v['F05'], 4) for f, v in ORACLE.items()})

# %% [markdown]
# ## 13. Baselines — an experiment history from trivial to ML
# All baselines are evaluated on **hold**. Thresholds, where they exist, are tuned on **tune**.

# %%
def _country_code_arrays():
    codes = {c: i + 1 for i, c in enumerate(sorted(set(P_SLICES) | set(Q_CK)))}
    pc = np.zeros(NP, np.uint64)
    for c, (s, e) in P_SLICES.items():
        pc[s:e] = codes[c]
    qc = np.array([codes[c] for c in Q_CK], dtype=np.uint64)
    return qc, pc


_QCC, _PCC = _country_code_arrays()


def exact_join_baseline(col, with_country, q_rows):
    qv = Q[col].take(q_rows).astype(object).to_numpy()
    qh = pd.util.hash_array(qv)
    ph = pd.util.hash_array(P[col].astype(object).to_numpy())
    if with_country:
        qh = qh + _QCC[q_rows] * np.uint64(0x9E3779B97F4A7C15)
        ph = ph + _PCC * np.uint64(0x9E3779B97F4A7C15)
    q = pd.DataFrame({'h': qh, 'qi': q_rows})[qv != '']
    p = pd.DataFrame({'h': ph, 'pi': np.arange(NP)})
    m = q.merge(p[p.h.isin(q.h)], on='h')
    keys = m.qi.to_numpy(np.int64) * NP + m.pi.to_numpy(np.int64)
    y = np.isin(keys, GT_Q_KEYS)
    return s1_metrics(m.qi.to_numpy(), y, N_TRUE_Q, q_rows)


BASE = {}
with Timer('baselines'):
    hq = FOLD_Q['hold']
    BASE['B1 exact raw name'] = exact_join_baseline('business_name', False, hq)
    BASE['B1 exact normalised (basic) name'] = exact_join_baseline('n_basic', False, hq)
    BASE['B2 exact basic name + country'] = exact_join_baseline('n_basic', True, hq)
    BASE['B2+ exact core name + country'] = exact_join_baseline('n_core', True, hq)
    _b3 = (0.5 * FE.n_tset.fillna(0) + 0.5 * FE.a_tset.fillna(FE.n_tset.fillna(0))).to_numpy()
    _, BASE['B3 weighted name/address token-set (tuned t)'] = eval_scores(_b3[ROWS['tune']], _b3[ROWS['hold']], 'B3 weighted name/address similarity')
    _b4 = FE[['n_ratio', 'n_jw', 'n_char_cos', 'n_tset', 'a_ratio', 'a_tset', 'a_tok_cos']].mean(axis=1, skipna=True).fillna(0).to_numpy()
    _, BASE['B4 mean of fuzzy similarities (tuned t)'] = eval_scores(_b4[ROWS['tune']], _b4[ROWS['hold']], 'B4 fuzzy similarity average')
BASE_TABLE = pd.DataFrame(BASE).T[['F05', 'P_micro', 'R_micro', 'singleton_acc', 'false_merge_rate', 'avg_pred_per_S1']]
display(BASE_TABLE.round(4))
add_experiment('E0', 'Baseline: exact normalised name (no country)', BASE['B1 exact normalised (basic) name'], 'reference point')
add_experiment('E1', 'Name normalisation: core name (legal forms, translit, learned map) + country block',
               BASE['B2+ exact core name + country'],
               'keep' if BASE['B2+ exact core name + country']['F05'] >= BASE['B1 exact normalised (basic) name']['F05'] else 'keep as feature only')

# %% [markdown]
# ## 14. ML matching models
# All models train on **fit-fold blocking candidates**, whose negatives are hard by construction. Early stopping uses the tune fold. Model comparison uses the
# F0.5-oriented protocol above (tuned threshold, macro F0.5 on hold), not ROC-AUC.
# Following the ensembling rules in the knowledge base, a **linear** model is included for structural diversity next to the GBDTs, and an inter-model correlation check is reported.

# %%
EMB_FEATS = [c for c in FE.columns if c.startswith('emb_cos')] + (['blk_G'] if 'blk_G' in FE else [])
FEATS_ALL = [c for c in FE.columns if c not in ('country_eq',)]
FEATS_BASE = [c for c in FEATS_ALL if c not in EMB_FEATS]
FEATS_NOCTX = [c for c in FEATS_BASE if not (c.startswith(CTX_PREFIX) or c.endswith(CTX_SUFFIX) or c == 'q_ncand')]
FEATS_NAME_ONLY = [c for c in FEATS_BASE if not (c.startswith('a_') or c.startswith('q_a') or c.startswith('p_a') or
                                                 c.startswith('af') or c in ('comb_cos', 'name_x_addr', 'both_addr_missing') or
                                                 c.startswith('comb_cos') or c.startswith('name_x_addr') or c.startswith('a_tset'))]
print(f'features: all={len(FEATS_ALL)} base(classical)={len(FEATS_BASE)} no-context={len(FEATS_NOCTX)} name-only={len(FEATS_NAME_ONLY)}')
_COLIDX = {c: i for i, c in enumerate(FE.columns)}


def X_of(rows, feats):
    return FE.iloc[rows, [_COLIDX[c] for c in feats]].to_numpy(np.float32)


def cap_rows(rows, max_rows, seed=SEED):
    if len(rows) <= max_rows:
        return rows
    qs = np.unique(C_QI[rows])
    keep_q = np.random.RandomState(seed).choice(qs, int(len(qs) * max_rows / len(rows)), replace=False)
    return rows[np.isin(C_QI[rows], keep_q)]


LGB_PARAMS = dict(objective='binary', learning_rate=CFG['LGB_LR'], num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255,
                  num_threads=os.cpu_count(), seed=SEED, verbose=-1)


def train_lgb(feats, tr_rows, va_rows, rounds=None, X_tr=None, y_tr=None):
    X_tr = X_of(tr_rows, feats) if X_tr is None else X_tr
    y_tr = C_Y[tr_rows] if y_tr is None else y_tr
    dtr = lgb.Dataset(X_tr, y_tr, feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(X_of(va_rows, feats), C_Y[va_rows], reference=dtr)
    m = lgb.train(LGB_PARAMS, dtr, num_boost_round=rounds or CFG['LGB_ROUNDS'], valid_sets=[dva],
                  callbacks=[lgb.early_stopping(CFG['EARLY_STOP'], verbose=False), lgb.log_evaluation(250)])
    return m


MODELS, VAL_PRED = {}, {}


def predict_model(name, getX):
    m = MODELS[name]
    if m['kind'] == 'ens':
        return np.mean([predict_model(n, getX) for n in m['members']], axis=0)
    X = getX(m['feats'])
    if m['kind'] == 'lgb':
        return m['obj'].predict(X, num_iteration=m['obj'].best_iteration)
    if m['kind'] == 'xgb':
        return m['obj'].predict(xgb.DMatrix(X, missing=np.nan), iteration_range=(0, m['obj'].best_iteration + 1))
    if m['kind'] in ('cat', 'lr'):
        return m['obj'].predict_proba(X)[:, 1]
    raise ValueError(m['kind'])


def register(name, entry):
    MODELS[name] = entry
    VAL_PRED[name] = {f: predict_model(name, lambda feats, f=f: X_of(ROWS[f], feats)).astype(np.float32) for f in ('tune', 'hold')}
    prm, m = eval_scores(VAL_PRED[name]['tune'], VAL_PRED[name]['hold'], name)
    tune_f = tune_global(RuleEval(ROWS['tune'], VAL_PRED[name]['tune'], 'tune'))[1]
    MODELS[name].update(tune_F05=tune_f, hold=m, prm=prm)
    return m


TR_ROWS = cap_rows(ROWS['fit'], CFG['MAX_TRAIN_ROWS'])
print(f'training rows: {len(TR_ROWS):,} (positives {C_Y[TR_ROWS].mean():.2%})')

with Timer('LightGBM (classical features)'):
    _m = train_lgb(FEATS_BASE, TR_ROWS, ROWS['tune'])
    register('lgb', dict(kind='lgb', obj=_m, feats=FEATS_BASE))
    _m.save_model(os.path.join(ART_DIR, 'lgb.txt'))

if CFG['RUN_XGB'] and xgb is not None:
    with Timer('XGBoost'):
        try:
            _Xt, _Xv = X_of(TR_ROWS, FEATS_BASE), X_of(ROWS['tune'], FEATS_BASE)
            _p = dict(objective='binary:logistic', eval_metric='logloss', eta=CFG['LGB_LR'], max_depth=8, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=5, seed=SEED, tree_method='hist')
            if USE_GPU:
                _p.update(device='cuda') if int(xgb.__version__.split('.')[0]) >= 2 else _p.update(tree_method='gpu_hist')
            _m = xgb.train(_p, xgb.DMatrix(_Xt, C_Y[TR_ROWS], missing=np.nan), CFG['LGB_ROUNDS'],
                           evals=[(xgb.DMatrix(_Xv, C_Y[ROWS['tune']], missing=np.nan), 'tune')],
                           early_stopping_rounds=CFG['EARLY_STOP'], verbose_eval=250)
            del _Xt, _Xv
            register('xgb', dict(kind='xgb', obj=_m, feats=FEATS_BASE))
            _m.save_model(os.path.join(ART_DIR, 'xgb.json'))
        except Exception as e:
            print('XGBoost failed:', e)

if CFG['RUN_CATBOOST'] and catboost is not None:
    with Timer('CatBoost'):
        try:
            from catboost import CatBoostClassifier
            _m = CatBoostClassifier(iterations=CFG['LGB_ROUNDS'], learning_rate=0.08, depth=8, eval_metric='Logloss',
                                    task_type='GPU' if USE_GPU else 'CPU', random_seed=SEED, od_type='Iter',
                                    od_wait=CFG['EARLY_STOP'], verbose=250, thread_count=os.cpu_count())
            _m.fit(X_of(TR_ROWS, FEATS_BASE), C_Y[TR_ROWS], eval_set=(X_of(ROWS['tune'], FEATS_BASE), C_Y[ROWS['tune']]))
            register('catboost', dict(kind='cat', obj=_m, feats=FEATS_BASE))
            _m.save_model(os.path.join(ART_DIR, 'catboost.cbm'))
        except Exception as e:
            print('CatBoost failed:', e)

if CFG['RUN_LR']:
    with Timer('Logistic regression (linear, diversity)'):
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        _r = cap_rows(TR_ROWS, 1_500_000)
        _m = make_pipeline(SimpleImputer(strategy='constant', fill_value=-1), StandardScaler(),
                           LogisticRegression(max_iter=400, C=1.0))
        _m.fit(X_of(_r, FEATS_BASE), C_Y[_r])
        register('logreg', dict(kind='lr', obj=_m, feats=FEATS_BASE))

_trees = [n for n in ('lgb', 'xgb', 'catboost') if n in MODELS]
if len(_trees) >= 2:
    register('ens_trees', dict(kind='ens', members=_trees))
if 'logreg' in MODELS and len(_trees) >= 1:
    register('ens_trees+lr', dict(kind='ens', members=_trees + ['logreg']))

CORR = pd.DataFrame({n: VAL_PRED[n]['hold'] for n in MODELS if MODELS[n]['kind'] != 'ens'}).corr()
display(CORR.round(4))
MODEL_TABLE = pd.DataFrame({n: dict(tune_F05=MODELS[n]['tune_F05'], hold_F05=MODELS[n]['hold']['F05'], hold_P=MODELS[n]['hold']['P_micro'],
                                    hold_R=MODELS[n]['hold']['R_micro'], t=MODELS[n]['prm']['t_s2'],
                                    hold_AUC=roc_auc_score(C_Y[ROWS['hold']], VAL_PRED[n]['hold']),
                                    hold_logloss=log_loss(C_Y[ROWS['hold']], np.clip(VAL_PRED[n]['hold'], 1e-6, 1 - 1e-6)))
                            for n in MODELS}).T.sort_values('tune_F05', ascending=False)
display(MODEL_TABLE.round(4))
BEST_CLASSICAL = MODEL_TABLE.index[0]

imp = pd.Series(MODELS['lgb']['obj'].feature_importance('gain'), index=FEATS_BASE).sort_values(ascending=False)
imp.head(30)[::-1].plot.barh(figsize=(8, 8), title='LightGBM gain importance (top 30)'); plt.tight_layout(); plt.show()

# reliability of the best classical model on hold
_ph, _yh = VAL_PRED[BEST_CLASSICAL]['hold'], C_Y[ROWS['hold']]
_bins = pd.cut(_ph, np.linspace(0, 1, 11), include_lowest=True)
REL = pd.DataFrame({'pred': _ph, 'y': _yh}).groupby(_bins).agg(mean_pred=('pred', 'mean'), frac_pos=('y', 'mean'), n=('y', 'size'))
display(REL.round(3))
print(f'Brier (hold) = {brier_score_loss(_yh, _ph):.4f}')

# ---- ablations with a quick LightGBM (same protocol) ----
_quick_q = np.sort(np.random.RandomState(SEED).choice(FOLD_Q['fit'], min(CFG['QUICK_S1'], len(FOLD_Q['fit'])), replace=False))
QUICK_ROWS = ROWS['fit'][np.isin(C_QI[ROWS['fit']], _quick_q)]


def quick_eval(feats, label, tr_rows=None, X_tr=None, y_tr=None, rows_tune=None, rows_hold=None):
    m = train_lgb(feats, QUICK_ROWS if tr_rows is None else tr_rows, ROWS['tune'], rounds=CFG['QUICK_ROUNDS'], X_tr=X_tr, y_tr=y_tr)
    pt = m.predict(X_of(ROWS['tune'], feats), num_iteration=m.best_iteration)
    ph = m.predict(X_of(ROWS['hold'], feats), num_iteration=m.best_iteration)
    return eval_scores(pt, ph, label)[1]


with Timer('ablations E2/E3 + country transfer'):
    m_name = quick_eval(FEATS_NAME_ONLY, 'E2a name-only features')
    m_full = quick_eval(FEATS_BASE, 'E2b name + address features')
    add_experiment('E2', f'Address features added to name features (name-only F0.5 {m_name["F05"]:.4f})', m_full,
                   'keep address features' if m_full['F05'] > m_name['F05'] else 'drop address features')
    # E3: random negatives vs hard (blocking) negatives, same S1, same no-context feature set
    _rng = np.random.RandomState(SEED + 7)
    _pos = GT_Q[np.isin(GT_Q.qi.to_numpy(), _quick_q)]
    _neg = pd.DataFrame({'qi': np.repeat(_quick_q, 5).astype(np.int32)})
    _neg['pi'] = random_same_country(Q_CK[_neg.qi.to_numpy()], P_SLICES, _rng).astype(np.int32)
    _rp = pd.concat([_pos[['qi', 'pi']], _neg], ignore_index=True).sort_values('qi').reset_index(drop=True)
    _ry = np.isin(_rp.qi.to_numpy(np.int64) * NP + _rp.pi.to_numpy(np.int64), GT_Q_KEYS).astype(np.int8)
    _RF = pair_features(_rp.qi.to_numpy(), _rp.pi.to_numpy(), Q, P, Qm, Pm, RET, Qe, Pe)
    m_rand = quick_eval(FEATS_NOCTX, 'E3a trained on random negatives', X_tr=_RF[FEATS_NOCTX].to_numpy(np.float32), y_tr=_ry)
    m_hard = quick_eval(FEATS_NOCTX, 'E3b trained on blocking (hard) negatives')
    add_experiment('E3', f'Hard (blocking) negatives instead of random negatives (random-neg F0.5 {m_rand["F05"]:.4f})', m_hard,
                   'train on blocking candidates' if m_hard['F05'] >= m_rand['F05'] else 'check negatives')
    del _RF, _rp
    # country transfer (proxy for the unseen test country): train on one country, evaluate on the other
    TRANSFER = {}
    _cs = [c for c in P_SLICES if (C_CK[ROWS['fit']] == c).sum() > 10000]
    if len(_cs) >= 2:
        c_tr, c_te = _cs[0], _cs[1]
        for label, tr in [(f'train {c_tr} only', QUICK_ROWS[C_CK[QUICK_ROWS] == c_tr]), ('train all countries', QUICK_ROWS)]:
            m = train_lgb(FEATS_BASE, tr, ROWS['tune'], rounds=CFG['QUICK_ROUNDS'])
            rt, rh = ROWS['tune'][C_CK[ROWS['tune']] == c_te], ROWS['hold'][C_CK[ROWS['hold']] == c_te]
            et = RuleEval(rt, m.predict(X_of(rt, FEATS_BASE), num_iteration=m.best_iteration), 'tune')
            et.q_eval = FOLD_Q['tune'][Q_CK[FOLD_Q['tune']] == c_te]
            prm, _, _ = tune_global(et)
            eh = RuleEval(rh, m.predict(X_of(rh, FEATS_BASE), num_iteration=m.best_iteration), 'hold')
            eh.q_eval = FOLD_Q['hold'][Q_CK[FOLD_Q['hold']] == c_te]
            TRANSFER[label] = eh.metrics(prm)['F05']
        print(f'Country transfer — evaluated on {c_te}:', {k: round(v, 4) for k, v in TRANSFER.items()})
        finding('Cross-country generalisation (proxy for France)',
                f"F0.5 on {c_te}: trained on {c_tr} only = {TRANSFER[f'train {c_tr} only']:.4f} vs all countries = {TRANSFER['train all countries']:.4f}",
                'The test contains an unseen country; features must transfer.',
                'Country-agnostic features (no country id, IDF computed per pool, generic legal forms, transliteration) — '
                'a small transfer gap supports using the same model/thresholds for France.')

add_experiment('E5', f'ML model ({BEST_CLASSICAL}, {len(FEATS_BASE)} classical features, global threshold)',
               MODELS[BEST_CLASSICAL]['hold'], f'best classical model = {BEST_CLASSICAL}')
finding('Model family',
        f"tune/hold F0.5: " + ', '.join(f'{n}={MODEL_TABLE.loc[n, "tune_F05"]:.4f}/{MODEL_TABLE.loc[n, "hold_F05"]:.4f}' for n in MODEL_TABLE.index),
        'GBDTs model the non-linear compensation between name/address/number evidence; correlated trees add little when blended.',
        f'Use {BEST_CLASSICAL} as the classical scorer (selected on TUNE F0.5, never on hold).')

# %% [markdown]
# ## 15. GPU embedding experiment (learned encoder)
# Question: does the learned char-n-gram encoder add signal beyond the classical features? The comparison is **classical** versus **classical + `emb_cos` (+ its context + the Block-G flag)**,
# with the same LightGBM, same data and same protocol. Embeddings are kept only if hold-out F0.5 improves by at least `EMB_MIN_GAIN`.

# %%
USE_EMB = False
if EMB_FEATS:
    with Timer('embedding ablation'):
        m_q_base = quick_eval(FEATS_BASE, 'E7a classical (quick)')
        m_q_emb = quick_eval(FEATS_ALL, 'E7b classical + embeddings (quick)')
        _gain = m_q_emb['F05'] - m_q_base['F05']
        USE_EMB = _gain >= CFG['EMB_MIN_GAIN']
        _auc_emb = roc_auc_score(C_Y[ROWS['hold']], FE.emb_cos.to_numpy()[ROWS['hold']])
        finding('Learned embeddings',
                f'emb_cos alone AUC {_auc_emb:.4f}; quick-model F0.5 {m_q_base["F05"]:.4f} → {m_q_emb["F05"]:.4f} (gain {_gain:+.4f})',
                'The encoder captures character-level similarity across typos/transliteration jointly for name+address.',
                'KEEP embeddings (feature + Block G)' if USE_EMB else 'DROP embedding features (no meaningful gain); Block G kept only if selected by blocking recall')
        add_experiment('E7', f'Embeddings added (classical quick F0.5 {m_q_base["F05"]:.4f})', m_q_emb, 'keep' if USE_EMB else 'drop')
        if USE_EMB:
            with Timer('LightGBM (classical + embeddings)'):
                _m = train_lgb(FEATS_ALL, TR_ROWS, ROWS['tune'])
                register('lgb_emb', dict(kind='lgb', obj=_m, feats=FEATS_ALL))
                _m.save_model(os.path.join(ART_DIR, 'lgb_emb.txt'))
                _members = ['lgb_emb'] + [n for n in ('xgb', 'catboost') if n in MODELS]
                if len(_members) >= 2:
                    register('ens_emb', dict(kind='ens', members=_members))
else:
    print('No embeddings available (no GPU / encoder disabled) — E7 skipped.')
    add_experiment('E7', 'Embeddings (not available in this run)', MODELS[BEST_CLASSICAL]['hold'], 'skipped')

SEL_TABLE = pd.DataFrame({n: dict(tune_F05=MODELS[n]['tune_F05'], hold_F05=MODELS[n]['hold']['F05']) for n in MODELS}).T.sort_values('tune_F05', ascending=False)
STAGE1_MODEL = SEL_TABLE.index[0]
print('Stage-1 model selected on TUNE F0.5:', STAGE1_MODEL)
display(SEL_TABLE.round(4))

# %% [markdown]
# ### 15b. Stage-2 reranker: cross-source confirmation
# Over 80% of S1 entities have matches in **both** S2 and S3, and the records of one entity agree with each other.
# Stage 2 therefore scores each candidate with extra information:
# * Its stage-1 probability and its rank and gap within the S1.
# * How similar it is to the **best-scoring candidate of the other source**, and to the best *other* candidate of its own source, together with those candidates' probabilities.
#
# Stage-1 scores for the fit fold are **out-of-fold** (2-fold by S1). This keeps stage 2 from learning on overconfident in-sample probabilities.

# %%
def stage2_chunk(qi, pi, s3, p1, P_, Pm_):
    n = len(qi)
    s3 = s3.astype(np.int64)
    df = pd.DataFrame({'qi': qi, 'p1': p1, 'h': (p1 >= 0.5).astype(np.float32), 'hs': (p1 >= 0.5).astype(np.float32)})
    g = df.groupby('qi')
    F = {'p1': p1.astype(np.float32)}
    F['p1_rank'] = g.p1.rank(ascending=False, method='min').to_numpy(np.float32)
    F['p1_qmax'] = g.p1.transform('max').to_numpy(np.float32)
    F['p1_gap'] = F['p1_qmax'] - F['p1']
    F['p1_q_n50'] = g.h.transform('sum').to_numpy(np.float32)
    F['p1_src_n50'] = df.groupby([qi, s3]).hs.transform('sum').to_numpy(np.float32)
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
            nt[ok] = _rf(fuzz.token_set_ratio, P_.n_core.take(a).tolist(), P_.n_core.take(b).tolist(), 100)
            ac1, ac2 = P_.a_canon.take(a).tolist(), P_.a_canon.take(b).tolist()
            v = _rf(fuzz.token_set_ratio, ac1, ac2, 100)
            v[(_obj(ac1) == '') | (_obj(ac2) == '')] = np.nan
            at[ok] = v
            cc[ok] = rowdot(Pm_['nchar'], Pm_['nchar'], a, b)
        F[tag + '_p1'], F[tag + '_n_tset'], F[tag + '_a_tset'], F[tag + '_char'] = rp, nt, at, cc
        F[tag + '_support'] = (np.nan_to_num(rp) * np.fmax(np.nan_to_num(nt), np.nan_to_num(at))).astype(np.float32)
    return pd.DataFrame(F)


def stage2_frame(qi, pi, s3, p1, P_, Pm_, keep_df):
    parts = [stage2_chunk(qi[s:e], pi[s:e], s3[s:e], p1[s:e], P_, Pm_) for s, e in group_chunks(qi, 2_000_000)]
    S = pd.concat(parts, ignore_index=True)
    return pd.concat([S, keep_df.reset_index(drop=True)], axis=1)


USE_STAGE2 = False
STAGE2 = None
if CFG['RUN_STAGE2'] and lgb is not None:
    with Timer('stage-2 cross-source reranker'):
        s1_name = 'lgb_emb' if (USE_EMB and 'lgb_emb' in MODELS) else 'lgb'
        s1_feats = MODELS[s1_name]['feats']
        _imp = pd.Series(MODELS[s1_name]['obj'].feature_importance('gain'), index=s1_feats).sort_values(ascending=False)
        KEEP_FEATS = list(_imp.index[:CFG['STAGE2_KEEP_FEATS']])
        n_rounds = max(100, int(MODELS[s1_name]['obj'].best_iteration * 1.05))
        # out-of-fold stage-1 predictions on the fit fold (2 folds by S1)
        P1 = np.zeros(len(CAND), np.float32)
        half = (pd.util.hash_array(Q.entity_id.astype(object).to_numpy()) % 2).astype(np.int64)[C_QI]
        for h in (0, 1):
            tr = TR_ROWS[half[TR_ROWS] != h]
            pr = ROWS['fit'][half[ROWS['fit']] == h]
            mh = lgb.train(LGB_PARAMS, lgb.Dataset(X_of(tr, s1_feats), C_Y[tr]), num_boost_round=n_rounds)
            P1[pr] = mh.predict(X_of(pr, s1_feats))
        for f in ('tune', 'hold'):
            P1[ROWS[f]] = VAL_PRED[s1_name][f]
        S2F = stage2_frame(C_QI, C_PI, C_SRC3.astype(np.int64), P1, P, Pm, FE[KEEP_FEATS])
        S2_FEATS = list(S2F.columns)
        _X = lambda rows: S2F.iloc[rows].to_numpy(np.float32)
        m2 = lgb.train(dict(LGB_PARAMS, num_leaves=63), lgb.Dataset(_X(TR_ROWS), C_Y[TR_ROWS], feature_name=S2_FEATS),
                       num_boost_round=CFG['LGB_ROUNDS'], valid_sets=[lgb.Dataset(_X(ROWS['tune']), C_Y[ROWS['tune']])],
                       callbacks=[lgb.early_stopping(CFG['EARLY_STOP'], verbose=False), lgb.log_evaluation(250)])
        p2 = {f: m2.predict(_X(ROWS[f]), num_iteration=m2.best_iteration).astype(np.float32) for f in ('tune', 'hold')}
        prm2, m2_hold = eval_scores(p2['tune'], p2['hold'], 'stage-2 cross-source reranker')
        tune2 = tune_global(RuleEval(ROWS['tune'], p2['tune'], 'tune'))[1]
        gain = tune2 - MODELS[STAGE1_MODEL]['tune_F05']
        USE_STAGE2 = gain >= CFG['STAGE2_MIN_GAIN']
        STAGE2 = dict(stage1=s1_name, keep_feats=KEEP_FEATS, feats=S2_FEATS, model=m2, val=p2, hold=m2_hold)
        m2.save_model(os.path.join(ART_DIR, 'stage2_lgb.txt'))
        imp2 = pd.Series(m2.feature_importance('gain'), index=S2_FEATS).sort_values(ascending=False)
        display(imp2.head(15).round(0).to_frame('gain'))
        finding('Cross-source confirmation (stage 2)',
                f'tune F0.5 {MODELS[STAGE1_MODEL]["tune_F05"]:.4f} → {tune2:.4f} ({gain:+.4f}); hold F0.5 '
                f'{MODELS[STAGE1_MODEL]["hold"]["F05"]:.4f} → {m2_hold["F05"]:.4f}',
                'A candidate that agrees with the confident candidate of the other source is very likely the same entity; '
                'an isolated look-alike is not.',
                'USE stage-2 reranker in the final pipeline' if USE_STAGE2 else 'Stage 2 not kept (gain below threshold)')
        add_experiment('E8', 'Stage-2 cross-source confirmation reranker (OOF stacking)', m2_hold, 'keep' if USE_STAGE2 else 'drop')
        del S2F
        gc.collect()

FINAL_SCORER = 'stage2' if USE_STAGE2 else STAGE1_MODEL
FINAL_P = {f: (STAGE2['val'][f] if USE_STAGE2 else VAL_PRED[STAGE1_MODEL][f]) for f in ('tune', 'hold')}
print('FINAL scorer:', FINAL_SCORER)
