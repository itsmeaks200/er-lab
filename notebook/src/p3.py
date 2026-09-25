# %% [markdown]
# ## 6. Name-normalisation experiments
# Every transformation is judged on two opposing effects:
# * **Useful collisions**: the share of true (S1, S2/S3) pairs whose representations become *equal* (`pos_equal_pct`, a recall proxy).
# * **Dangerous collisions**: S1 is a deduplicated reference, so any two S1 records that become equal are *different entities*
#   (`S1_collision_pct`). `exact_rule_precision_pct` is the precision of the rule "match every S1 in the same country with an equal representation".
#
# A learned synonym map (e.g. `intl→international`) is estimated **from fit-fold positive pairs only** before these stages run.

# %%
def country_slices(Pdf):
    ck = Pdf.country_key.astype(object).to_numpy()
    b = np.flatnonzero(ck[1:] != ck[:-1]) + 1
    starts, ends = np.r_[0, b], np.r_[b, len(ck)]
    return {ck[s]: (int(s), int(e)) for s, e in zip(starts, ends)}


def random_same_country(ck_arr, slices, rng):
    s = np.array([slices[c][0] for c in ck_arr]); e = np.array([slices[c][1] for c in ck_arr])
    return (s + (rng.rand(len(s)) * (e - s)).astype(np.int64)).astype(np.int64)


# named stage functions (picklable for the parallel map)
def rep_raw(s): return s
def rep_lower(s): return ' '.join(s.lower().split())
def rep_translit(s): return ' '.join(translit(s.lower()).split())
def rep_nbasic(s): return name_basic(s)
def rep_ncore(s): return name_core(name_basic(s))
def rep_nsorted(s): return ' '.join(sorted(name_core(name_basic(s)).split()))
def rep_nsquash(s): return name_core(name_basic(s)).replace(' ', '')
def rep_nskel(s): return name_skel(name_core(name_basic(s))).replace(' ', '')


def rep_experiment(stages, s1_text, s1_ck, pos_s1, o_text, o_ck, s1_raw, o_raw):
    rows, prev_key, prev_eq = [], None, None
    for name, fn in stages:
        r_s1 = umap(s1_text, fn).astype(object).to_numpy()
        r_o = np.empty(len(o_text), dtype=object); r_o[:] = [fn(x) for x in o_text]
        k_s1, k_o = s1_ck + '|' + r_s1, o_ck + '|' + r_o
        eq = (r_s1[pos_s1] == r_o) & (r_o != '')
        vc = pd.Series(k_s1).value_counts()
        n_same = pd.Series(k_o).map(vc).fillna(0).to_numpy()
        n_same[r_o == ''] = 0
        dup = pd.Series(k_s1).duplicated(keep=False).to_numpy()
        rows.append(dict(representation=name, unique_values=len(vc), S1_collision_pct=100 * dup.mean(),
                         pos_equal_pct=100 * eq.mean(), exact_rule_precision_pct=100 * eq.sum() / max(n_same.sum(), 1),
                         mean_S1_sharing_value=n_same.mean()))
        if prev_key is not None:
            for i in np.flatnonzero(eq & ~prev_eq)[:2]:
                print(f'   [{name}] useful collision   : {s1_raw[pos_s1[i]]!r}  ==  {o_raw[i]!r}')
            g = pd.DataFrame({'cur': k_s1, 'prev': prev_key}).groupby('cur', sort=False)['prev'].nunique()
            for v in g[g > 1].index[:2]:
                mem = np.flatnonzero(k_s1 == v)[:3]
                print(f'   [{name}] dangerous collision: ' + '  |  '.join(repr(s1_raw[m]) for m in mem))
        prev_key, prev_eq = k_s1, eq
    out = pd.DataFrame(rows).set_index('representation')

    def role(r):
        if r.pos_equal_pct >= 15 and r.exact_rule_precision_pct >= 50:
            return 'blocking key + feature'
        if r.pos_equal_pct >= 15:
            return 'feature only (ambiguous as a rule)'
        return 'feature only (low coverage)'
    out['recommended_role'] = out.apply(role, axis=1)
    return out


def _jac(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A or B) else np.nan


def _c3(s):
    s = f' {s} '
    return [s[i:i + 3] for i in range(len(s) - 2)]


def sim_auc_table(reps, pos_pairs, neg_pairs):
    """AUC of a similarity representation for positives vs random same-country negatives."""
    rows = []
    for name, fn in reps:
        sp_ = np.array([fn(a, b) for a, b in pos_pairs], dtype=float)
        sn_ = np.array([fn(a, b) for a, b in neg_pairs], dtype=float)
        y = np.r_[np.ones(len(sp_)), np.zeros(len(sn_))]
        s = np.nan_to_num(np.r_[sp_, sn_], nan=-1)
        rows.append(dict(similarity=name, AUC=roc_auc_score(y, s), pos_median=np.nanmedian(sp_), neg_median=np.nanmedian(sn_)))
    return pd.DataFrame(rows).set_index('similarity')


def _rf_ratio(a, b):
    return fuzz.ratio(a, b) / 100 if HAS_RF else np.nan


with Timer('name normalisation experiments'):
    P_SLICES = country_slices(P)
    _fitpos = gt_pairs[gt_pairs.fold == 'fit']
    POS = _fitpos.sample(min(CFG['POS_SAMPLE'], len(_fitpos)), random_state=SEED).reset_index(drop=True)
    pos_s1, pos_pi = POS.s1_idx.to_numpy(), POS.pi.to_numpy()
    S1_CK = S1.country_key.astype(object).to_numpy()
    pos_o_ck = P.country_key.take(pos_pi).astype(object).to_numpy()
    pos_o_name, pos_o_addr = P.business_name.take(pos_pi).tolist(), P.business_address.take(pos_pi).tolist()
    S1_RAW_NAME = S1.business_name.astype(object).to_numpy()
    S1_RAW_ADDR = S1.business_address.astype(object).to_numpy()
    _rng = np.random.RandomState(SEED)
    neg_pi = random_same_country(pos_o_ck, P_SLICES, _rng)

    # ---- learned name synonym map (fit-fold positives only) ----
    NAME_MAP.clear()
    _left = [name_core(b) for b in S1.n_basic.take(pos_s1).tolist()]
    _right = [name_core(name_basic(x)) for x in pos_o_name]
    NAME_MAP.update(learn_token_map(_left, _right))
    print(f'learned NAME_MAP ({len(NAME_MAP)} entries):', dict(list(NAME_MAP.items())[:30]))

    NAME_STAGES = [('raw', rep_raw), ('lower + whitespace', rep_lower), ('+ unicode/transliteration', rep_translit),
                   ('+ punct, &→and, TLD, L.L.C. (basic)', rep_nbasic), ('+ legal forms/stopwords/leet/learned map (core)', rep_ncore),
                   ('+ token sort', rep_nsorted), ('squashed (no spaces)', rep_nsquash), ('consonant skeleton (squashed)', rep_nskel)]
    NAME_REP = rep_experiment(NAME_STAGES, S1.business_name, S1_CK, pos_s1, pos_o_name, pos_o_ck, S1_RAW_NAME, pos_o_name)
display(NAME_REP.round(2))

_n = min(30_000, len(POS))
_pos_pairs = list(zip(S1.n_basic.take(pos_s1[:_n]).tolist(), [name_basic(x) for x in pos_o_name[:_n]]))
_neg_pairs = list(zip(S1.n_basic.take(pos_s1[:_n]).tolist(), [name_basic(x) for x in P.business_name.take(neg_pi[:_n]).tolist()]))
NAME_SIM = sim_auc_table([
    ('token Jaccard (core)', lambda a, b: _jac(name_core(a).split(), name_core(b).split())),
    ('char-3gram Jaccard (core)', lambda a, b: _jac(_c3(name_core(a)), _c3(name_core(b)))),
    ('skeleton-token Jaccard', lambda a, b: _jac(name_skel(name_core(a)).split(), name_skel(name_core(b)).split())),
    ('squashed ratio', lambda a, b: _rf_ratio(name_core(a).replace(' ', ''), name_core(b).replace(' ', ''))),
], _pos_pairs, _neg_pairs)
display(NAME_SIM.round(3))

_nr = NAME_REP
finding('Name normalisation trade-off',
        f"exact-equal positives: raw {_nr.pos_equal_pct.iloc[0]:.1f}% → basic {_nr.pos_equal_pct.iloc[3]:.1f}% → core {_nr.pos_equal_pct.iloc[4]:.1f}% "
        f"→ skeleton {_nr.pos_equal_pct.iloc[-1]:.1f}%; S1 collisions raw {_nr.S1_collision_pct.iloc[0]:.1f}% → skeleton {_nr.S1_collision_pct.iloc[-1]:.1f}%",
        'Each aggressive step recovers matches but also merges distinct S1 entities (dangerous for F0.5).',
        'Keep ALL representations: basic/core/squash/skeleton each become separate features; only high-precision ones '
        '(see recommended_role) are used as exact blocking keys; nothing is deleted from the stored raw text.')
finding('Order-insensitive / character-level name similarity',
        '; '.join(f'{i}: AUC {r.AUC:.3f}' for i, r in NAME_SIM.iterrows()),
        'Word re-ordering, typos and concatenated domains defeat exact tokens.',
        'Use token-set/sort ratios, char-3gram TF-IDF cosine and skeleton/squash similarities as pair features; '
        'char/skeleton evidence also drives the dense encoder (Block G).')

# %% [markdown]
# ## 7. Address normalisation
# Address noise includes abbreviations (`Rd/Road`), component re-ordering (`OR, Eugene, 3900 River Road`), native-script state names,
# `NULL/N/A` placeholders, landmarks (`near SBI ATM`) and missing components. The same useful/dangerous analysis is repeated, plus numeric-anchor,
# postal-code and landmark statistics. A learned address token map (e.g. `ohio→oh`, `telangana→tg`) comes from fit-fold positives only.

# %%
def rep_abasic(s): return addr_basic(s)
def rep_ahand(s): return ' '.join(ADDR_HAND_MAP.get(t, t) for t in addr_basic(s).split())
def rep_acanon(s): return addr_canon(addr_basic(s))
def rep_aset(s): return ' '.join(sorted(set(addr_canon(addr_basic(s)).split())))
def rep_anums(s): return ' '.join(sorted({x.lstrip('0') or '0' for x in _NUM.findall(addr_basic(s))}))


_LANDMARK = {'near', 'opp', 'behind', 'beside', 'besides', 'adjacent', 'nr', 'bh', 'infront', 'next'}


def postal_codes(canon):
    """postal-like numbers: 6-digit tokens (PIN) or 5-digit tokens not followed by a street word (ZIP / code postal)."""
    t = canon.split()
    out = []
    for i, x in enumerate(t):
        if x.isdigit() and (len(x) == 6 or (len(x) == 5 and not (i + 1 < len(t) and t[i + 1].isalpha() and len(t[i + 1]) > 2))):
            out.append(x)
    return out


def _nums(c):
    return {x.lstrip('0') or '0' for x in _NUM.findall(c)}


with Timer('address normalisation experiments'):
    ADDR_MAP.clear()
    _left = [rep_ahand(x) for x in S1.business_address.take(pos_s1).tolist()]
    _right = [rep_ahand(x) for x in pos_o_addr]
    ADDR_MAP.update(learn_token_map(_left, _right))
    print(f'learned ADDR_MAP ({len(ADDR_MAP)} entries):', dict(list(ADDR_MAP.items())[:40]))
    ADDR_STAGES = [('raw', rep_raw), ('lower + whitespace', rep_lower), ('+ unicode/transliteration', rep_translit),
                   ('+ punct, ordinals, placeholders (basic)', rep_abasic), ('+ multilingual street-type map', rep_ahand),
                   ('+ learned token map (canon)', rep_acanon), ('sorted token set', rep_aset), ('numbers only', rep_anums)]
    ADDR_REP = rep_experiment(ADDR_STAGES, S1.business_address, S1_CK, pos_s1, pos_o_addr, pos_o_ck, S1_RAW_ADDR, pos_o_addr)
display(ADDR_REP.round(2))

_neg_addr = P.business_address.take(neg_pi[:_n]).tolist()
_s1c = [addr_canon(addr_basic(x)) for x in S1.business_address.take(pos_s1[:_n]).tolist()]
_poc = [addr_canon(addr_basic(x)) for x in pos_o_addr[:_n]]
_nec = [addr_canon(addr_basic(x)) for x in _neg_addr]


def _anchor_stats(A, B):
    r = Counter()
    for a, b in zip(A, B):
        na, nb = _nums(a), _nums(b)
        pa, pb = set(postal_codes(a)), set(postal_codes(b))
        ka, kb = set(addr_keys(a).split()), set(addr_keys(b).split())
        if na and nb:
            r['both_numbers'] += 1; r['num_shared'] += bool(na & nb); r['num_conflict'] += not (na & nb)
        if pa and pb:
            r['both_postal'] += 1; r['postal_equal'] += bool(pa & pb)
        r['housenum_street_key_shared'] += bool(ka & kb)
        r['set_equal_but_reordered'] += (set(a.split()) == set(b.split()) and a != b and a != '')
        r['n'] += 1
    n = r['n']
    return {'both have numbers %': 100 * r['both_numbers'] / n,
            'numbers overlap | both %': 100 * r['num_shared'] / max(r['both_numbers'], 1),
            'numbers conflict | both %': 100 * r['num_conflict'] / max(r['both_numbers'], 1),
            'both have postal %': 100 * r['both_postal'] / n,
            'postal equal | both %': 100 * r['postal_equal'] / max(r['both_postal'], 1),
            'house#+street key shared %': 100 * r['housenum_street_key_shared'] / n,
            'same tokens, different order %': 100 * r['set_equal_but_reordered'] / n}


ANCHORS = pd.DataFrame({'true pairs': _anchor_stats(_s1c, _poc), 'random same-country pairs': _anchor_stats(_s1c, _nec)})
display(ANCHORS.round(2))
_lm = {k: 100 * np.mean([bool(_LANDMARK & set(addr_basic(x).split())) for x in v]) for k, v in
       [('S1', S1.business_address.take(pos_s1[:_n]).tolist()), ('S2/S3', pos_o_addr[:_n])]}
print('addresses with landmark words (near/opp/behind...) %:', {k: round(v, 2) for k, v in _lm.items()})

ADDR_SIM = sim_auc_table([
    ('token Jaccard (canon)', lambda a, b: _jac(a.split(), b.split())),
    ('char-3gram Jaccard (canon)', lambda a, b: _jac(_c3(a), _c3(b))),
    ('number-set Jaccard', lambda a, b: _jac(_nums(a), _nums(b))),
    ('adjacent-bigram Jaccard', lambda a, b: _jac(list(zip(a.split(), a.split()[1:])), list(zip(b.split(), b.split()[1:])))),
], list(zip(_s1c, _poc)), list(zip(_s1c, _nec)))
display(ADDR_SIM.round(3))

_ar, _an = ADDR_REP, ANCHORS
finding('Address exact matching is weak, token/numeric evidence is strong',
        f"canon exact-equal positives {_ar.pos_equal_pct.loc['+ learned token map (canon)']:.1f}% vs sorted-set {_ar.pos_equal_pct.loc['sorted token set']:.1f}%; "
        f"reordered-only pairs {_an.loc['same tokens, different order %', 'true pairs']:.1f}%; numbers overlap given both have numbers: "
        f"true {_an.loc['numbers overlap | both %', 'true pairs']:.1f}% vs random {_an.loc['numbers overlap | both %', 'random same-country pairs']:.1f}%",
        'Component re-ordering and abbreviations make string equality useless; numbers (house/door/PIN) are stable anchors.',
        '(1) Blocking: TF-IDF address tokens + adjacent bigrams ("3900_river") and house-number+street keys (Block E). '
        '(2) Features: order-free token set/sort ratios, number-set Jaccard/conflict/subset, first-number equality. '
        '(3) Decision: numeric conflict is a strong veto that the GBDT learns.')
finding('Postal codes as a veto',
        f"postal equal given both have one: true {_an.loc['postal equal | both %', 'true pairs']:.1f}% vs random "
        f"{_an.loc['postal equal | both %', 'random same-country pairs']:.1f}% (both have postal in {_an.loc['both have postal %', 'true pairs']:.1f}% of true pairs)",
        'Same-name franchises / homonyms in different towns differ in postal code.',
        'Dedicated postal-code match / conflict / missing features; postal + name-skeleton blocking pass (Block H).')

# %% [markdown]
# ### 7b. Model-ready representations for the training phase
# The frozen normalisation is applied to all train S1 (2.2M) and the pool (10.3M). Next come record-level attributes (lengths, noise flags, name/address
# **commonness** in S1 and in the pool) and the hashed sparse matrices that blocking and features share. The entity-level experiment sample `Q`
# holds `N_S1_SAMPLE` S1 entities with their fit/tune/hold folds. The whole pool stays in play, so distractor density is realistic.

# %%
REC_COLS = ['n_len', 'n_ntok', 'a_len', 'a_ntok', 'f_domain', 'f_nonascii', 'f_upper', 'f_junk', 'a_missing',
            'nf_s1', 'nf_pool', 'af_s1', 'af_pool']


def _ntok(ser):
    c = ser.str.count(' ').to_numpy(np.float32) + 1
    c[(ser == '').to_numpy(bool)] = 0
    return c


def prepare_frame(df):
    """Frozen normalisation used identically for train and test frames (adds columns in place)."""
    if 'country_key' not in df:
        df['country_key'] = df.country.str.strip().str.lower()
    if 'n_basic' not in df:
        df['n_basic'] = umap(df.business_name, name_basic)
    df['n_core'] = umap(df.n_basic, name_core)
    df['n_squash'] = df.n_core.str.replace(' ', '', regex=False)
    df['n_skel'] = umap(df.n_core, name_skel)
    df['n_alt'] = umap(df.n_basic, name_alt)
    if 'a_basic' not in df:
        df['a_basic'] = umap(df.business_address, addr_basic)
    df['a_canon'] = umap(df.a_basic, addr_canon)
    df['n_len'] = df.n_basic.str.len().to_numpy(np.float32)
    df['n_ntok'] = _ntok(df.n_core)
    df['a_len'] = df.a_canon.str.len().to_numpy(np.float32)
    df['a_ntok'] = _ntok(df.a_canon)
    fl = umap(df.business_name, name_flags, as_str=False).to_numpy(np.int64)
    df['f_domain'] = ((fl & 1) > 0).astype(np.float32)
    df['f_nonascii'] = ((fl & 2) > 0).astype(np.float32)
    df['f_upper'] = ((fl & 4) > 0).astype(np.float32)
    df['f_junk'] = ((fl & 8) > 0).astype(np.float32)
    df['a_missing'] = (df.a_canon == '').to_numpy(bool).astype(np.float32)
    df['num0'] = umap(df.a_canon, first_number, as_str=False).to_numpy(np.int64)
    return df


def add_freq_features(s1df, pdf):
    """Commonness of the name/address among S1 entities and in the pool (log counts)."""
    for col, tag in [('n_core', 'nf'), ('a_canon', 'af')]:
        vs, vp = s1df[col].value_counts(), pdf[col].value_counts()
        for df in (s1df, pdf):
            empty = (df[col] == '').to_numpy(bool)
            a = np.log1p(df[col].map(vs).fillna(0).to_numpy(np.float32)); a[empty] = 0
            b = np.log1p(df[col].map(vp).fillna(0).to_numpy(np.float32)); b[empty] = 0
            df[tag + '_s1'], df[tag + '_pool'] = a, b


# ---- hashed sparse representations (shared by blocking and features) ----
def an_split(s): return s.split()
def an_atok(s): return [t for t in s.split() if not t.isdigit()]
def an_anum(s): return list({x.lstrip('0') or '0' for x in _NUM.findall(s)})
def an_abig(s):
    t = s.split()
    return [a + '_' + b for a, b in zip(t, t[1:])]
def an_apost(s): return postal_codes(s)


_HB, _CB = 2 ** CFG['HASH_BITS'], 2 ** CFG['CHAR_BITS']


def _hv(analyzer):
    return HashingVectorizer(analyzer=analyzer, n_features=_HB, alternate_sign=False, norm=None, binary=True, dtype=np.float32)


HV = {'ntok': _hv(an_split), 'nskel': _hv(an_split), 'atok': _hv(an_atok), 'anum': _hv(an_anum),
      'abig': _hv(an_abig), 'apost': _hv(an_apost),
      'nchar': HashingVectorizer(analyzer='char_wb', ngram_range=(3, 3), n_features=_CB, alternate_sign=False,
                                 norm='l2', dtype=np.float32, lowercase=False)}
MAT_SRC = {'ntok': 'n_core', 'nskel': 'n_skel', 'nchar': 'n_core', 'atok': 'a_canon', 'anum': 'a_canon',
           'abig': 'a_canon', 'apost': 'a_canon'}
TOKEN_SPACES = ['ntok', 'nskel', 'atok', 'anum', 'abig', 'apost']


def _hv_transform(args):
    hv, docs = args
    return hv.transform(docs)


def vectorize(ser, hv, chunk=250_000):
    docs = ser.astype(object).tolist()
    parts = run_parallel(_hv_transform, [(hv, docs[i:i + chunk]) for i in range(0, len(docs), chunk)])
    X = sp.vstack(parts, format='csr') if len(parts) > 1 else parts[0].tocsr()
    X.sort_indices()
    return X


def build_matrices(df, tag):
    with Timer(f'vectorise {tag} ({len(df):,} records)'):
        M = {k: vectorize(df[MAT_SRC[k]], HV[k]) for k in HV}
        log('   nnz/record: ' + ', '.join(f'{k}={M[k].nnz / max(len(df), 1):.1f}' for k in M))
    return M


def fit_idf(Pm):
    """IDF statistics of the pool (label-free; recomputed on the test pool at inference)."""
    N = Pm['ntok'].shape[0]
    R = dict(N=N, DF={}, IDF={}, IDF2={}, RARE={}, W={})
    R['cap'] = cap = max(CFG['DF_CAP_MIN'], CFG['DF_CAP_FRAC'] * N)
    mult = {'ntok': 1.0, 'nskel': CFG['SKEL_WEIGHT'], 'atok': 1.0, 'anum': CFG['NUM_WEIGHT'],
            'abig': CFG['BIGRAM_WEIGHT'], 'apost': 1.0}
    for k in TOKEN_SPACES:
        d = np.bincount(Pm[k].indices, minlength=Pm[k].shape[1]).astype(np.float32)
        R['DF'][k] = d
        R['IDF'][k] = (np.log((N + 1) / (d + 1)) + 1).astype(np.float32)
        R['IDF2'][k] = R['IDF'][k] ** 2
        R['RARE'][k] = (d <= CFG['RARE_DF']).astype(np.float32)
        w = R['IDF'][k] * mult[k]
        w[d > cap] = 0
        R['W'][k] = w.astype(np.float32)
        tot = Pm[k].nnz
        dropped = float(d[d > cap].sum())
        log(f'   {k:6s}: {int((d > 0).sum()):>10,} hashed tokens, {int((d > cap).sum()):>6,} above df-cap {cap:,.0f} '
            f'(= {100 * dropped / max(tot, 1):5.1f}% of token occurrences ignored by retrieval)')
    return R


with Timer('prepare train representations'):
    prepare_frame(S1)
    prepare_frame(P)
    add_freq_features(S1, P)
    Q = S1.iloc[SAMPLE_IDX].reset_index(drop=True)
    Q_FOLD = S1_FOLD[SAMPLE_IDX]
    NQ = len(Q)
    s1_to_q = np.full(NS1, -1, np.int64); s1_to_q[SAMPLE_IDX] = np.arange(NQ)
    _g = gt_pairs[s1_to_q[gt_pairs.s1_idx.to_numpy()] >= 0]
    GT_Q = pd.DataFrame({'qi': s1_to_q[_g.s1_idx.to_numpy()].astype(np.int32), 'pi': _g.pi.to_numpy(np.int32)})
    GT_Q_KEYS = GT_Q.qi.to_numpy(np.int64) * NP + GT_Q.pi.to_numpy(np.int64)      # aligned with GT_Q rows
    GT_Q_FOLD = Q_FOLD[GT_Q.qi.to_numpy()]
    N_TRUE_Q = np.bincount(GT_Q.qi.to_numpy(), minlength=NQ)
    FOLD_Q = {f: np.flatnonzero(Q_FOLD == f) for f in ['fit', 'tune', 'hold']}
    Q_CK = Q.country_key.astype(object).to_numpy()
    del S1_RAW_NAME, S1_RAW_ADDR, S1, gt_pairs, POS   # not needed after this point (commonness stats already computed)
    gc.collect()
    print(f'Q (experiment S1) = {NQ:,}; GT pairs in Q = {len(GT_Q):,}; folds =', {k: len(v) for k, v in FOLD_Q.items()})

Qm = build_matrices(Q, 'Q (train sample)')
Pm = build_matrices(P, 'pool (train)')
with Timer('pool IDF'):
    RET = fit_idf(Pm)
gc.collect()

# %% [markdown]
# ### Blocking engine (label-free)
# The engine offers several independent candidate generators. **None of them receives labels.** Ground truth is only used later, to *measure* recall.
#
# | pass | idea | targets |
# |---|---|---|
# | A | exact squashed core name (+country) | exact / punctuation / domain variants |
# | B | first 8 chars of the name consonant skeleton (+country) | vowel typos, transliteration |
# | C | TF-IDF cosine on name tokens + skeleton tokens (top-K, within country) | re-ordering, extra words |
# | D | TF-IDF cosine on address tokens + numbers + adjacent bigrams (top-K) | DBA / trade names, renamed businesses |
# | E | house-number + street-token key | re-ordered addresses |
# | F | combined name/address TF-IDF score (top-K) | typical noisy records |
# | G | learned char-n-gram encoder, GPU dense top-K | heavy typos, domains, transliteration |
# | H | postal code + first name-skeleton token | same area, noisy name |
#
# Tokens more frequent than the document-frequency cap (`llc`, `road`, `delhi`, …) are ignored by retrieval, following the inverted-index stop-word rule.
# Scores are sparse matrix products computed chunk by chunk, one country at a time. The full Cartesian product is never materialised.

# %%
KEY_PASSES = ['A', 'B', 'E', 'H']
PASS_NAMES = {'A': 'A exact squashed name', 'B': 'B name-skeleton prefix', 'C': 'C TF-IDF name', 'D': 'D TF-IDF address',
              'E': 'E house#+street key', 'F': 'F TF-IDF name+address', 'G': 'G learned encoder (GPU)',
              'H': 'H postal + name skeleton'}
PASS_BITS = {p: 1 << i for i, p in enumerate('ABCDEFGH')}
RETR_FIELDS = {'name': ['ntok', 'nskel'], 'addr': ['atok', 'anum', 'abig']}

if HAS_NUMBA:
    @numba.njit
    def _topk_rows(indptr, indices, data, k):
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
else:
    def _topk_rows(indptr, indices, data, k):
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


def topk_csr(S, k):
    S = S.tocsr()
    return _topk_rows(S.indptr, S.indices, S.data.astype(np.float32, copy=False), k)


def scale_cols(X, w):
    X = sp.csr_matrix((X.data * w[X.indices], X.indices.copy(), X.indptr.copy()), shape=X.shape)
    X.eliminate_zeros()
    return X


def retr_mat(M, rows, fields, W):
    X = sp.hstack([scale_cols(M[f][rows], W[f]) for f in fields], format='csr')
    return sk_normalize(X, norm='l2', copy=False)


def _empty_pass():
    return pd.DataFrame({'qi': np.empty(0, np.int32), 'pi': np.empty(0, np.int32),
                         'score': np.empty(0, np.float32), 'rank': np.empty(0, np.int16)})


def _country_groups(q_rows, q_ck, p_slices, n_pool):
    ck = q_ck[q_rows]
    groups = []
    for c, (s, e) in p_slices.items():
        r = q_rows[ck == c]
        if len(r):
            groups.append((c, r, s, e))
    unknown = ~np.isin(ck, np.array(list(p_slices.keys()), dtype=object))
    if unknown.any():          # a country present in S1 but absent from the pool: search the whole pool
        groups.append(('<country-not-in-pool>', q_rows[unknown], 0, n_pool))
    return groups


def sparse_passes(Qm_, Pm_, q_rows, q_ck, p_slices, R, k, which=('C', 'D', 'F'), chunk=None):
    """Label-free TF-IDF retrieval (passes C/D/F), country by country, chunked sparse products."""
    chunk = chunk or CFG['RETR_CHUNK']
    alpha = CFG['ALPHA_NAME']
    out = {p: [] for p in which}
    need_n, need_a = any(p in which for p in 'CF'), any(p in which for p in 'DF')
    for c, rows_c, ps, pe in _country_groups(np.asarray(q_rows), q_ck, p_slices, Pm_['ntok'].shape[0]):
        t = time.time()
        PN = retr_mat(Pm_, slice(ps, pe), RETR_FIELDS['name'], R['W']).T.tocsr() if need_n else None
        PA = retr_mat(Pm_, slice(ps, pe), RETR_FIELDS['addr'], R['W']).T.tocsr() if need_a else None
        for s in range(0, len(rows_c), chunk):
            qi = rows_c[s:s + chunk]
            SN = (retr_mat(Qm_, qi, RETR_FIELDS['name'], R['W']) @ PN).tocsr() if need_n else None
            SA = (retr_mat(Qm_, qi, RETR_FIELDS['addr'], R['W']) @ PA).tocsr() if need_a else None
            mats = {}
            if 'C' in which:
                mats['C'] = SN
            if 'D' in which:
                mats['D'] = SA
            if 'F' in which:
                mats['F'] = (SN * alpha + SA * (1 - alpha)).tocsr()
            for p, S in mats.items():
                r, cix, sc, rk = topk_csr(S, k)
                out[p].append(pd.DataFrame({'qi': qi[r].astype(np.int32), 'pi': (cix.astype(np.int64) + ps).astype(np.int32),
                                            'score': sc, 'rank': rk}))
        log(f'   sparse retrieval [{c}] {len(rows_c):,} queries x {pe - ps:,} pool: {time.time() - t:.0f}s')
        del PN, PA
        gc.collect()
    return {p: (pd.concat(v, ignore_index=True) if v else _empty_pass()) for p, v in out.items()}


def addr_set_key(canon):
    return ' '.join(sorted(set(canon.split())))


def record_keys(df, kind):
    """(row, uint64 hash) arrays for exact-key blocking. kind: A,B,E,H (+ S,K for hard-negative mining)."""
    ck = df.country_key.astype(object).to_numpy()
    if kind in ('A', 'B', 'K'):
        if kind == 'A':
            k = df.n_squash
            minlen = 3
        else:
            k = df.n_skel.str.replace(' ', '', regex=False)
            minlen = 4
            if kind == 'B':
                k = k.str[:8]
        ln = k.str.len().to_numpy(np.int64)
        rows = np.flatnonzero(ln >= minlen)
        vals = ck[rows] + '|' + k.astype(object).to_numpy()[rows]
    elif kind == 'S':
        k = umap(df.a_canon, addr_set_key).astype(object).to_numpy()
        rows = np.flatnonzero(df.a_ntok.to_numpy() >= 3)
        vals = ck[rows] + '|' + k[rows]
    elif kind in ('E', 'H'):
        if kind == 'E':
            ks = umap(df.a_canon, addr_keys).astype(object).str.split().explode()
        else:
            ks = umap(df.a_canon, lambda_postal_str).astype(object).str.split().explode()
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


def lambda_postal_str(canon):
    return ' '.join(postal_codes(canon))


KEY_CACHE = {}


def key_pass(kind, Qdf, Pdf, q_rows, tag, max_block=None):
    """Exact-key blocking: join query keys to pool keys; blocks with > max_block pool records are skipped."""
    max_block = max_block or CFG['MAX_BLOCK_POOL']
    if (tag, kind) not in KEY_CACHE:
        KEY_CACHE[(tag, kind)] = (record_keys(Qdf, kind), record_keys(Pdf, kind))
    (qr, qh), (pr, ph) = KEY_CACHE[(tag, kind)]
    sel = np.isin(qr, q_rows)
    qdf = pd.DataFrame({'h': qh[sel], 'qi': qr[sel]})
    pdf_ = pd.DataFrame({'h': ph, 'pi': pr})
    pdf_ = pdf_[pdf_.h.isin(qdf.h.unique())]
    vc = pdf_.h.value_counts()
    ok = vc.index[vc.to_numpy() <= max_block]
    m = qdf[qdf.h.isin(ok)].merge(pdf_, on='h')[['qi', 'pi']].drop_duplicates()
    return pd.DataFrame({'qi': m.qi.to_numpy(np.int32), 'pi': m.pi.to_numpy(np.int32),
                         'score': np.ones(len(m), np.float32), 'rank': np.zeros(len(m), np.int16)})


def dense_pass(Qe, Pe, q_rows, q_ck, p_slices, k, chunk=None):
    """Pass G: exact dense top-K on the GPU (fp16), within country."""
    chunk = chunk or CFG['DENSE_CHUNK']
    R_, C_, S_, K_ = [], [], [], []
    for c, rows_c, ps, pe in _country_groups(np.asarray(q_rows), q_ck, p_slices, len(Pe)):
        t = time.time()
        Pt = torch.from_numpy(Pe[ps:pe]).cuda()
        kk = min(k, pe - ps)
        with torch.no_grad():
            for s in range(0, len(rows_c), chunk):
                qi = rows_c[s:s + chunk]
                v, ix = torch.topk(torch.from_numpy(Qe[qi]).cuda() @ Pt.T, kk, dim=1)
                R_.append(np.repeat(qi, kk).astype(np.int32)); C_.append((ix.cpu().numpy().ravel() + ps).astype(np.int32))
                S_.append(v.float().cpu().numpy().ravel()); K_.append(np.tile(np.arange(kk, dtype=np.int16), len(qi)))
        del Pt
        torch.cuda.empty_cache()
        log(f'   dense retrieval [{c}] {len(rows_c):,} queries x {pe - ps:,} pool: {time.time() - t:.0f}s')
    if not R_:
        return _empty_pass()
    return pd.DataFrame({'qi': np.concatenate(R_), 'pi': np.concatenate(C_), 'score': np.concatenate(S_), 'rank': np.concatenate(K_)})


def union_candidates(pass_dfs, K_sel, n_pool):
    """Union of the selected passes (top-K of ranked passes) → unique (qi, pi) with a bitmask of passes."""
    keys, bits = [], []
    for p, K in K_sel.items():
        d = pass_dfs[p]
        if p not in KEY_PASSES:
            d = d[d['rank'] < K]
        keys.append(d.qi.to_numpy(np.int64) * n_pool + d.pi.to_numpy(np.int64))
        bits.append(np.full(len(d), PASS_BITS[p], np.uint8))
    keys, bits = np.concatenate(keys), np.concatenate(bits)
    uk, inv = np.unique(keys, return_inverse=True)
    ob = np.zeros(len(uk), np.uint8)
    np.bitwise_or.at(ob, inv.ravel(), bits)
    return pd.DataFrame({'qi': (uk // n_pool).astype(np.int32), 'pi': (uk % n_pool).astype(np.int32), 'bits': ob})

# %% [markdown]
# ## 8. Pair features, and what true matches look like
# Features are computed **only for candidate pairs**, in chunks of about 1M pairs. They fall into these groups:
# * **Name**: raw/basic/core/squash/skeleton equality; Levenshtein, Jaro-Winkler, ratio, token-sort, token-set and partial ratios (rapidfuzz, C++, multi-threaded);
#   squash and skeleton ratios; trade-name (`t/a`, `dba`) ratio; acronym match; token Jaccard/overlap/IDF-weighted Jaccard/cosine; shared **rare** tokens; char-3gram cosine.
# * **Address**: raw/basic/canon equality; ratio, token-set, token-sort and partial ratios; token and bigram TF-IDF cosine; number-set Jaccard, conflict and subset; first-number equality;
#   **postal** match and conflict; lengths; missingness.
# * **Cross-field/context**: source (S2/S3), combined score, name/address *commonness* on both sides, noise flags, which blocking passes found the pair,
#   and **within-S1 context** (rank, gap to the best and z-score of key similarities among the S1's candidates; number of candidates).
#
# Country is deliberately **not** a feature, so the model transfers to France.

# %%
def _rf(scorer, a, b, scale):
    if scorer is None:
        return np.full(len(a), np.nan, np.float32)
    if cpdist is not None:
        return (cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / scale).astype(np.float32)
    return (np.fromiter((scorer(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a)) / scale).astype(np.float32)


def _obj(lst):
    a = np.empty(len(lst), dtype=object)
    a[:] = lst
    return a


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


def pair_features(qi, pi, Q_, P_, Qm_, Pm_, R, Qe=None, Pe=None):
    """Pairwise features for (Q_[qi], P_[pi]). Identical code path for validation and test."""
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
    Ac1, Ac2 = _obj(qac), _obj(pac)
    a_empty = (Ac1 == '') | (Ac2 == '')

    def eq(a, b, mask=None):
        v = (_obj(a) == _obj(b)).astype(np.float32)
        if mask is not None:
            v[mask] = np.nan
        return v
    f['n_raw_eq'], f['n_basic_eq'], f['n_core_eq'] = eq(qnr, pnr), eq(qb, pb), eq(qc, pc)
    f['n_squash_eq'], f['n_skel_eq'] = eq(qs, ps), eq(qk, pk)
    f['a_raw_eq'], f['a_basic_eq'], f['a_canon_eq'] = eq(qar, par, a_empty), eq(qab, pab, a_empty), eq(qac, pac, a_empty)

    # ---- fuzzy (rapidfuzz C++) ----
    f['n_ratio'] = _rf(fuzz.ratio, qb, pb, 100)
    f['n_lev'] = _rf(RF_Lev.normalized_similarity, qc, pc, 1)
    f['n_jw'] = _rf(RF_JW.normalized_similarity, qc, pc, 1)
    f['n_tsort'] = _rf(fuzz.token_sort_ratio, qc, pc, 100)
    f['n_tset'] = _rf(fuzz.token_set_ratio, qc, pc, 100)
    f['n_partial'] = _rf(fuzz.partial_ratio, qs, ps, 100)
    f['n_sq_ratio'] = _rf(fuzz.ratio, qs, ps, 100)
    f['n_skel_ratio'] = _rf(fuzz.ratio, qk, pk, 100)
    alt1 = _rf(fuzz.ratio, qc, palt, 100); alt1[_obj(palt) == ''] = np.nan
    alt2 = _rf(fuzz.ratio, qalt, pc, 100); alt2[_obj(qalt) == ''] = np.nan
    f['n_alt_ratio'] = np.fmax(alt1, alt2)
    f['n_acronym'] = np.fromiter(((len(s2) >= 2 and acronym(c1) == s2) or (len(s1) >= 2 and acronym(c2) == s1)
                                  for c1, c2, s1, s2 in zip(qc, pc, qs, ps)), dtype=np.float32, count=len(qc))
    for name, sc, scale in [('a_ratio', fuzz.ratio, 100), ('a_tset', fuzz.token_set_ratio, 100),
                            ('a_tsort', fuzz.token_sort_ratio, 100), ('a_partial', fuzz.partial_ratio, 100)]:
        v = _rf(sc, qac, pac, scale); v[a_empty] = np.nan
        f[name] = v

    # ---- sparse token statistics ----
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
    f['a_post_eq'] = ((psh > 0)).astype(np.float32)
    f['a_post_conflict'] = ((pa_ > 0) & (pb_ > 0) & (psh == 0)).astype(np.float32)
    f['a_post_missing'] = ((pa_ == 0) | (pb_ == 0)).astype(np.float32)
    q0, p0 = Q_.num0.to_numpy()[qi], P_.num0.to_numpy()[pi]
    f['a_num0_eq'] = np.where((q0 < 0) | (p0 < 0), np.nan, (q0 == p0)).astype(np.float32)

    # ---- cross-field / record level ----
    al = CFG['ALPHA_NAME']
    f['comb_cos'] = (al * np.nan_to_num(f['n_tok_cos']) + (1 - al) * np.nan_to_num(f['a_tok_cos'])).astype(np.float32)
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
    f['country_eq'] = (Q_.country_key.take(qi).to_numpy(object) == P_.country_key.take(pi).to_numpy(object)).astype(np.float32)
    if Qe is not None and Pe is not None:
        f['emb_cos'] = (Qe[qi].astype(np.float32) * Pe[pi].astype(np.float32)).sum(1)
    return pd.DataFrame(f)


GROUP_BASE = ['n_tset', 'n_char_cos', 'n_tok_wjacc', 'a_tset', 'a_tok_cos', 'comb_cos', 'name_x_addr', 'emb_cos']


def add_context_features(F, qi, bits, passes):
    """Within-S1 context: which passes found the pair, #candidates, rank / gap-to-best / z-score of key similarities."""
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


def safe_zone(F):
    """'Safe pruning zone': the name AND the address are both clearly dissimilar."""
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


def compute_features(cand, Q_, P_, Qm_, Pm_, R, passes, Qe=None, Pe=None, prune=False, scorer=None, keep_cols=None):
    """Chunked feature computation over qi-sorted candidates.
    scorer=None → returns (features DataFrame, keep mask); otherwise streams: returns (scores, keep mask, kept columns)."""
    qi_all, pi_all, bits_all = cand.qi.to_numpy(), cand.pi.to_numpy(), cand.bits.to_numpy()
    keep = np.ones(len(cand), bool)
    outs, kept = [], []
    bounds = group_chunks(qi_all, CFG['FEAT_CHUNK'])
    for j, (s, e) in enumerate(bounds):
        F = pair_features(qi_all[s:e], pi_all[s:e], Q_, P_, Qm_, Pm_, R, Qe, Pe)
        if prune:
            k = ~safe_zone(F)
            keep[s:e] = k
            F = F[k].reset_index(drop=True)
        add_context_features(F, qi_all[s:e][keep[s:e]], bits_all[s:e][keep[s:e]], passes)
        if scorer is None:
            outs.append(F)
        else:
            outs.append(scorer(F).astype(np.float32))
            if keep_cols:
                kept.append(F[keep_cols].astype(np.float32))
        if j % 10 == 0 or j == len(bounds) - 1:
            log(f'   features chunk {j + 1}/{len(bounds)} ({e:,}/{len(cand):,} pairs)')
    if scorer is None:
        return pd.concat(outs, ignore_index=True), keep
    return np.concatenate(outs), keep, (pd.concat(kept, ignore_index=True) if kept else None)


# ---- what do true matches look like? ----
with Timer('positive-pair features'):
    _fitgt = GT_Q[Q_FOLD[GT_Q.qi.to_numpy()] == 'fit']
    _pp = _fitgt.sample(min(100_000, len(_fitgt)), random_state=SEED).sort_values('qi')
    POSF = pair_features(_pp.qi.to_numpy(), _pp.pi.to_numpy(), Q, P, Qm, Pm, RET)
show_cols = ['n_core_eq', 'n_tset', 'n_jw', 'n_char_cos', 'n_skl_jacc', 'n_sq_ratio', 'a_tset', 'a_tok_cos', 'a_num_jacc',
             'a_num_conflict', 'a_post_eq', 'a_post_conflict', 'p_a_missing', 'p_f_nonascii', 'p_f_domain']
display(POSF[show_cols].describe(percentiles=[.05, .25, .5, .75]).T.round(3))

fig, axs = plt.subplots(2, 4, figsize=(20, 6.5))
for ax, c in zip(axs.ravel(), ['n_tset', 'n_char_cos', 'n_skel_ratio', 'n_jw', 'a_tset', 'a_tok_cos', 'a_num_jacc', 'comb_cos']):
    ax.hist(POSF[c].dropna(), bins=50, color='seagreen'); ax.set_title(f'true pairs: {c}')
plt.tight_layout(); plt.show()

_nj = pd.cut(POSF.n_tok_jacc.fillna(0), [-0.01, .2, .4, .6, .8, 1.0], labels=['0-.2', '.2-.4', '.4-.6', '.6-.8', '.8-1'])
_aj = pd.cut(POSF.a_tok_jacc.fillna(0), [-0.01, .2, .4, .6, .8, 1.0], labels=['0-.2', '.2-.4', '.4-.6', '.6-.8', '.8-1'])
GRID = pd.crosstab(_nj, _aj, normalize='all') * 100
print('true pairs: name-token Jaccard (rows) x address-token Jaccard (cols), % of pairs')
display(GRID.round(2))
_low_name = (POSF.n_tset < 0.5).mean()
finding('Compensation between name and address evidence',
        f'{_low_name:.1%} of true pairs have name token-set ratio < 0.5, of which {(POSF.loc[POSF.n_tset < 0.5, "a_tset"] > 0.8).mean():.1%} '
        f'have address token-set ratio > 0.8; the both-low corner (Jaccard < 0.2 on both) holds {GRID.iloc[0, 0]:.2f}% of true pairs',
        'Trade names/renamed businesses are only recoverable through the address, and address-less records only through the name.',
        'Blocking must include an address-driven pass (D/E) and a name-driven pass (A/B/C); the model gets both families plus '
        'their product; the both-dissimilar corner is a candidate for safe pruning (validated in §12).')
finding('Transliterated pool names',
        f"true pairs with non-ASCII pool name: n_tset median {POSF.loc[POSF.p_f_nonascii == 1, 'n_tset'].median():.2f}, "
        f"skeleton ratio median {POSF.loc[POSF.p_f_nonascii == 1, 'n_skel_ratio'].median():.2f}",
        'Without transliteration these pairs would have zero name similarity.',
        'Keep the Unicode-name transliteration + consonant skeleton; expose the non-ASCII flag so the model can trust skeleton features more for them.')

# %% [markdown]
# ## 9. Hard-negative construction
# Random negatives are trivially separable. Here negatives are mined for a subset of fit-fold S1 entities, using seven label-free strategies.
# The question is **which features still separate true matches from each kind of hard negative**.

# %%
with Timer('hard-negative mining'):
    _rng = np.random.RandomState(SEED + 1)
    HQ = np.sort(_rng.choice(FOLD_Q['fit'], min(30_000, len(FOLD_Q['fit'])), replace=False))
    _hp = sparse_passes(Qm, Pm, HQ, Q_CK, P_SLICES, RET, k=10, which=('C', 'D', 'F'))
    _rand = pd.DataFrame({'qi': np.repeat(HQ, 3).astype(np.int32)})
    _rand['pi'] = random_same_country(Q_CK[_rand.qi.to_numpy()], P_SLICES, _rng).astype(np.int32)
    NEG_SOURCES = {
        '1 same-country random': _rand,
        '2 similar name (TF-IDF name top-10)': _hp['C'],
        '3 similar address (TF-IDF addr top-10)': _hp['D'],
        '4 same normalised name': key_pass('A', Q, P, HQ, 'train', max_block=2000),
        '5 same address token-set': key_pass('S', Q, P, HQ, 'train', max_block=2000),
        '6 similar name + similar address': _hp['F'],
        '7 aggressive-normalisation collision': key_pass('K', Q, P, HQ, 'train', max_block=2000),
    }
    _parts = []
    for t, d in NEG_SOURCES.items():
        d = d[['qi', 'pi']].copy()
        d['type'] = t
        _parts.append(d)
    HN = pd.concat(_parts, ignore_index=True)
    HN['key'] = HN.qi.to_numpy(np.int64) * NP + HN.pi.to_numpy(np.int64)
    HN = HN[~np.isin(HN.key.to_numpy(), GT_Q_KEYS)]
    _pos = GT_Q[np.isin(GT_Q.qi.to_numpy(), HQ)].assign(type='0 TRUE MATCH')
    _pos['key'] = _pos.qi.to_numpy(np.int64) * NP + _pos.pi.to_numpy(np.int64)
    HN = pd.concat([HN, _pos], ignore_index=True)
    _u = HN.drop_duplicates('key').sort_values('qi').reset_index(drop=True)
    HF = pair_features(_u.qi.to_numpy(), _u.pi.to_numpy(), Q, P, Qm, Pm, RET)
    HF['key'] = _u.key.to_numpy()
    HNF = HN[['key', 'type']].merge(HF, on='key', how='left')
    _t6 = HNF.type.str.startswith('6')
    HNF = HNF[~_t6 | ((HNF.n_tset >= 0.8) & (HNF.a_tset >= 0.8))]
    _t7 = HNF.type.str.startswith('7')
    HNF = HNF[~_t7 | (HNF.n_basic_eq == 0)]

cnt = HNF.type.value_counts().sort_index()
print(cnt.to_string())
feat_cols = [c for c in HF.columns if c not in ('key', 'country_eq')]
posmask = HNF.type.str.startswith('0')
auc_rows = {}
for t in sorted(HNF.type.unique()):
    if t.startswith('0'):
        continue
    sub = HNF[posmask | (HNF.type == t)]
    if (sub.type == t).sum() < 50:
        continue
    y = sub.type.str.startswith('0').to_numpy()
    auc_rows[t] = {c: roc_auc_score(y, sub[c].fillna(-1).to_numpy()) for c in feat_cols}
AUC_HN = pd.DataFrame(auc_rows)
AUC_HN = AUC_HN.apply(lambda s: np.maximum(s, 1 - s))            # direction-free separability
AUC_HN['worst_case'] = AUC_HN.min(axis=1)
AUC_HN = AUC_HN.sort_values('worst_case', ascending=False)
display(AUC_HN.head(25).round(3))
med = HNF.groupby('type')[['n_tset', 'n_char_cos', 'a_tset', 'a_num_jacc', 'a_num_conflict', 'a_post_conflict', 'p_nf_s1', 'q_nf_pool']].median()
display(med.round(3))

fig, ax = plt.subplots(figsize=(12, 7))
_top = AUC_HN.head(20).drop(columns='worst_case')
im = ax.imshow(_top.to_numpy(), aspect='auto', cmap='viridis', vmin=0.5, vmax=1)
ax.set_yticks(range(len(_top))); ax.set_yticklabels(_top.index)
ax.set_xticks(range(_top.shape[1])); ax.set_xticklabels([c[:28] for c in _top.columns], rotation=35, ha='right')
plt.colorbar(im, label='AUC (true vs negative type)'); ax.set_title('Which features separate true matches from each hard-negative type')
plt.tight_layout(); plt.show()

_hard = AUC_HN.drop(columns='worst_case').drop(columns=[c for c in AUC_HN.columns if c.startswith('1')], errors='ignore')
finding('Hard negatives need different evidence than random ones',
        f"median AUC over features: random negatives {AUC_HN.filter(like='1 same').median().iloc[0]:.3f} vs "
        f"same-normalised-name negatives {AUC_HN.filter(like='4 same').median().iloc[0] if AUC_HN.filter(like='4 same').shape[1] else float('nan'):.3f}; "
        f"best worst-case separators: {', '.join(AUC_HN.index[:6])}",
        'A model trained on random negatives learns "names differ → no match" and has nothing to say about same-name/same-address traps.',
        'Train on blocking candidates (hard negatives by construction); keep the address/number/postal/commonness features that '
        'separate the same-name and same-address traps. E3 quantifies random vs hard negatives.')
del HN, HF, _u, _hp
gc.collect()
