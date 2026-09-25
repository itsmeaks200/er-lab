# %% [markdown]
# ## 3. Deep EDA — source-specific noise
# Each source is profiled on its full data (counts, uniqueness, duplicates) and on a sample (string shape, scripts, noise signatures).
# Test sources are profiled too, to measure the train→test shift. Every statistic below feeds a normalisation, feature or blocking decision.

# %%
_DOMAIN_RE = re.compile(r'\.(?:com|net|org|in|co|biz|info|io|us|fr)\b|^[@#][a-z0-9]|www\.')
_NULL_TOK_RE = r'\b(?:null|none|nan|n/a)\b'


def script_of(s):
    for ch in s:
        if ord(ch) > 0x24F and ch.isalpha():
            return unicodedata.name(ch, 'OTHER X').split(' ')[0]
    return 'LATIN'


def source_profile(name, df, n_sample):
    r = dict(source=name, records=len(df))
    r['uniq_names_%'] = 100 * df.business_name.nunique() / len(df)
    r['uniq_addr_%'] = 100 * df.business_address.nunique() / len(df)
    r['n_countries'] = df.country.nunique()
    r['empty_addr_%'] = 100 * (df.business_address.str.strip() == '').mean()
    r['dup_record_%'] = 100 * df.duplicated(['business_name', 'business_address', 'country']).mean()
    smp = df.sample(min(n_sample, len(df)), random_state=SEED)
    nm, ad = smp.business_name.astype(object), smp.business_address.astype(object)
    nlen, alen = nm.str.len(), ad.str.len()
    r['name_len_med'], r['name_len_p95'] = nlen.median(), nlen.quantile(.95)
    r['name_words_mean'] = nm.str.split().str.len().mean()
    r['name_digit_%'] = 100 * nm.str.contains(r'\d').mean()
    r['name_punct_%'] = 100 * nm.str.contains(r'[^\w\s]').mean()
    r['name_nonascii_%'] = 100 * (~nm.map(str.isascii)).mean()
    r['name_accented_latin_%'] = 100 * nm.str.contains('[À-ɏ]').mean()
    r['name_upper_%'] = 100 * nm.map(str.isupper).mean()
    r['name_junk_prefix_%'] = 100 * nm.map(lambda s: bool(s) and not s[0].isalnum()).mean()
    r['name_domain_handle_%'] = 100 * nm.str.lower().map(lambda s: bool(_DOMAIN_RE.search(s))).mean()
    r['addr_len_med'] = alen.median()
    r['addr_words_mean'] = ad.str.split().str.len().mean()
    r['addr_digits_mean'] = ad.str.count(r'\d').mean()
    r['addr_null_token_%'] = 100 * ad.str.lower().str.contains(_NULL_TOK_RE).mean()
    r['addr_nonascii_%'] = 100 * (~ad.map(str.isascii)).mean()
    scripts = nm.map(script_of).value_counts(normalize=True)
    return r, scripts, nlen.clip(upper=100), alen.clip(upper=200)


with Timer('source profiling'):
    PROF, SCRIPTS, NLEN, ALEN = {}, {}, {}, {}
    for k in ['train_s1', 'train_s2', 'train_s3', 'test_s1', 'test_s2', 'test_s3']:
        PROF[k], SCRIPTS[k], NLEN[k], ALEN[k] = source_profile(k, RAW[k], CFG['EDA_SAMPLE'])
    prof = pd.DataFrame(PROF).T.drop(columns='source')
    display(prof.astype(float).round(2).T)

fig, ax = plt.subplots(1, 4, figsize=(22, 4.2))
for k in ['train_s1', 'train_s2', 'train_s3']:
    ax[0].hist(NLEN[k], bins=60, histtype='step', density=True, label=k)
    ax[1].hist(ALEN[k], bins=60, histtype='step', density=True, label=k)
ax[0].set_title('name length (chars)'); ax[1].set_title('address length (chars)')
ax[0].legend(); ax[1].legend()
noise_cols = ['name_upper_%', 'name_junk_prefix_%', 'name_domain_handle_%', 'name_nonascii_%',
              'name_accented_latin_%', 'empty_addr_%', 'addr_null_token_%']
prof.loc[['train_s1', 'train_s2', 'train_s3'], noise_cols].astype(float).T.plot.bar(ax=ax[2], rot=60)
ax[2].set_title('noise signatures by source (% records)')
sc = pd.DataFrame({k: SCRIPTS[k] for k in ['train_s1', 'train_s2', 'train_s3', 'test_s2']}).fillna(0)
sc = sc.loc[sc.max(axis=1) > 0.001]
(1 - sc.loc[['LATIN']]).T.rename(columns={'LATIN': 'non-Latin script share'}).plot.bar(ax=ax[3], rot=0, legend=False)
ax[3].set_title('share of names in non-Latin scripts')
plt.tight_layout(); plt.show()
display((sc * 100).round(2))

cc = pd.DataFrame({k: RAW[k].country.str.strip().value_counts(normalize=True) for k in
                   ['train_s1', 'train_s2', 'train_s3', 'test_s1', 'test_s2', 'test_s3']}).fillna(0)
display((cc * 100).round(2))

P_ = prof.astype(float)
finding('Source-specific casing noise',
        f"ALL-CAPS names: S1 {P_.loc['train_s1', 'name_upper_%']:.1f}% vs S2 {P_.loc['train_s2', 'name_upper_%']:.1f}% vs S3 {P_.loc['train_s3', 'name_upper_%']:.1f}%",
        'Case differences break exact matching and inflate edit distance.',
        'Case-fold everything; keep one raw-exact feature so the model can still reward identical raw strings.')
nonlat = {k: 100 * (1 - SCRIPTS[k].get('LATIN', 0)) for k in SCRIPTS}
finding('Non-Latin scripts (transliteration noise)',
        'non-Latin names: ' + ', '.join(f'{k}={v:.1f}%' for k, v in nonlat.items()),
        'S1 is Latin-only; a Devanagari/Telugu/... S2/S3 name shares no characters with its S1 entity.',
        'Rule-based, dependency-free transliteration (Unicode character names → Latin) + consonant-skeleton keys; '
        'the model receives a non-ASCII flag.')
finding('Junk prefixes, domains and handles',
        f"junk prefix S2/S3 = {P_.loc['train_s2', 'name_junk_prefix_%']:.1f}%/{P_.loc['train_s3', 'name_junk_prefix_%']:.1f}%, "
        f"domain/handle S2/S3 = {P_.loc['train_s2', 'name_domain_handle_%']:.1f}%/{P_.loc['train_s3', 'name_domain_handle_%']:.1f}%",
        '"victorylaboratories.com" / "@memorialproject" have no token overlap with "Victory Laboratories Limited".',
        'Strip punctuation/TLDs; add a *squashed* (space-free) name representation used as a blocking key and fuzzy feature.')
finding('Common (non-unique) business names in the reference source',
        f"only {P_.loc['train_s1', 'uniq_names_%']:.1f}% of S1 names are unique strings",
        'Many distinct entities share a name → name-only matching produces false merges (precision is weighted 2x in F0.5).',
        'Name-frequency (commonness) features on both sides; address agreement required by the model for common names.')
finding('Train→test shift',
        'max |Δ| (train vs test S2) over profile stats: ' +
        ', '.join(f"{c}={abs(P_.loc['train_s2', c] - P_.loc['test_s2', c]):.2f}" for c in noise_cols[:5]),
        'Large shifts would invalidate thresholds tuned on train.',
        'Noise signatures are similar; the only structural shift is the new country (handled by country-agnostic design).')
TEST_COUNTS = {k: len(RAW[k]) for k in ['test_s1', 'test_s2', 'test_s3']}
for k in ['test_s1', 'test_s2', 'test_s3']:
    del RAW[k]                       # reloaded in §20 — frees memory for the training phase
gc.collect()

# %% [markdown]
# ## 4. Ground-truth reconstruction
# `matched_entity_ids` is exploded into a pair table `(S1 entity, S2/S3 entity, source pair, target=1)`.
# The pool of S2 and S3 records is built and sorted by country so that every country is a contiguous slice (used by blocking).
# The **entity-level split** used everywhere is defined here as well (fit/tune/hold by a hash of the S1 id; see §12).

# %%
def build_ground_truth(gt):
    ex = gt[GT_COLS].astype(object).copy()
    ex['o_id'] = ex.matched_entity_ids.str.split(',')
    ex = ex.explode('o_id')
    ex['o_id'] = ex.o_id.str.strip()
    ex = ex[ex.o_id.notna() & (ex.o_id != '')]
    pairs = pd.DataFrame({'s1_id': ex.source1_entity_id.to_numpy(), 'o_id': ex.o_id.to_numpy()}).drop_duplicates()
    pairs['src'] = pairs.o_id.str[:2]
    pairs['target'] = np.int8(1)
    return pairs.reset_index(drop=True)


def build_pool(s2, s3):
    s2 = s2.copy(); s2['src'] = np.int8(2)
    s3 = s3.copy(); s3['src'] = np.int8(3)
    P = pd.concat([s2, s3], ignore_index=True)
    P['country_key'] = P.country.str.strip().str.lower()
    return P.sort_values('country_key', kind='stable').reset_index(drop=True)


with Timer('ground truth + pool'):
    S1 = RAW.pop('train_s1')
    S1['country_key'] = S1.country.str.strip().str.lower()
    P = build_pool(RAW.pop('train_s2'), RAW.pop('train_s3'))
    gt_pairs = build_ground_truth(RAW.pop('train_gt'))
    gc.collect()
    gt_pairs['s1_idx'] = pd.Index(S1.entity_id.astype(object)).get_indexer(gt_pairs.s1_id).astype(np.int32)
    _pidx = pd.Index(P.entity_id.astype(object))
    gt_pairs['pi'] = _pidx.get_indexer(gt_pairs.o_id).astype(np.int32)
    del _pidx
    bad_ref = int(((gt_pairs.s1_idx < 0) | (gt_pairs.pi < 0)).sum())
    if bad_ref:
        print(f'WARNING: dropping {bad_ref} GT pairs with unknown ids')
        gt_pairs = gt_pairs[(gt_pairs.s1_idx >= 0) & (gt_pairs.pi >= 0)].reset_index(drop=True)
    NS1, NP = len(S1), len(P)

N_TRUE_ALL = np.bincount(gt_pairs.s1_idx, minlength=NS1)
N_S2 = np.bincount(gt_pairs.s1_idx[gt_pairs.src == 'S2'], minlength=NS1)
N_S3 = np.bincount(gt_pairs.s1_idx[gt_pairs.src == 'S3'], minlength=NS1)
print(f'S1={NS1:,}  pool(S2+S3)={NP:,}  GT pairs={len(gt_pairs):,}')
display(gt_pairs.head())

mult = pd.Series(N_TRUE_ALL).value_counts().sort_index()
share0, share1, share_many = (N_TRUE_ALL == 0).mean(), (N_TRUE_ALL == 1).mean(), (N_TRUE_ALL > 1).mean()
p_src = P.src.to_numpy()
matched_mask = np.zeros(NP, bool); matched_mask[gt_pairs.pi.to_numpy()] = True
tab = pd.DataFrame({
    'S1 with >=1 match %': [100 * (N_S2 > 0).mean(), 100 * (N_S3 > 0).mean()],
    'mean matches / S1': [N_S2.mean(), N_S3.mean()],
    'max matches / S1': [N_S2.max(), N_S3.max()],
    'pool records matched to some S1 %': [100 * matched_mask[p_src == 2].mean(), 100 * matched_mask[p_src == 3].mean()],
}, index=['S1-S2', 'S1-S3'])
display(tab.round(2))
display(pd.crosstab(pd.Series(N_S2, name='#S2 matches'), pd.Series(N_S3, name='#S3 matches')))

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
mult.plot.bar(ax=ax[0], color='steelblue'); ax[0].set_title('matches per S1 entity'); ax[0].set_xlabel('# true matches')
pd.DataFrame({'S2': pd.Series(N_S2).value_counts(normalize=True), 'S3': pd.Series(N_S3).value_counts(normalize=True)}).sort_index().plot.bar(ax=ax[1])
ax[1].set_title('per-source multiplicity (share of S1)')
plt.tight_layout(); plt.show()

kind = 'one-to-many' if share_many > 0.5 else ('one-to-one' if share1 > 0.8 else 'mixed')
finding('Matching cardinality',
        f'singletons {share0:.2%}, exactly one match {share1:.2%}, >1 match {share_many:.2%}; mean {N_TRUE_ALL.mean():.2f}, max {N_TRUE_ALL.max()}; '
        f'S1 with both S2 & S3 matches {((N_S2 > 0) & (N_S3 > 0)).mean():.2%}',
        'Top-1 assignment would cap recall far below 1 for most entities.',
        f'Problem is {kind.upper()}: threshold-based multi-match decisions (never top-1 only); S2 and S3 are *not* internally deduplicated '
        f'(up to {N_S2.max()} S2 records per S1) → the decision rule must accept several records from the same source.')
finding('Pool contains many distractors',
        f"{100 * (1 - matched_mask.mean()):.1f}% of S2/S3 records are not matched to any S1",
        'Candidate generation will surface unmatched look-alikes → need hard-negative training.',
        'Train on blocking candidates (natural hard negatives), not on random pairs (tested in E3).')
finding('Exclusivity of pool records',
        f"{GT_INTEG['ids_mapped_to_multiple_s1']} S2/S3 ids linked to more than one S1",
        'If a record belongs to at most one S1, competing S1 claims can be resolved globally.',
        'Decision rule option "exclusive": each S2/S3 record is kept only for its best-scoring S1 (tuned in §17).')

# ---- entity-level split (fit / tune / hold) + experimental sample ----
_h = pd.util.hash_pandas_object(S1.entity_id.astype(object), index=False).to_numpy() % 100
f1, f2, _ = CFG['FOLD_PCTS']
S1_FOLD = np.where(_h < f1, 'fit', np.where(_h < f1 + f2, 'tune', 'hold'))
gt_pairs['fold'] = S1_FOLD[gt_pairs.s1_idx.to_numpy()]
SAMPLE_IDX = np.sort(np.random.RandomState(SEED).choice(NS1, min(CFG['N_S1_SAMPLE'], NS1), replace=False))
print('Fold sizes (all S1):', pd.Series(S1_FOLD).value_counts().to_dict(),
      '| sample fold sizes:', pd.Series(S1_FOLD[SAMPLE_IDX]).value_counts().to_dict())

# %% [markdown]
# ### Text-normalisation engine
# These functions are defined once and used unchanged by validation and test inference. §6 and §7 quantify each transformation.
# * **Transliteration**: the Python standard library `unicodedata` turns Indic, Greek and Cyrillic characters into Latin letters using their Unicode *names*
#   (e.g. `DEVANAGARI LETTER TTA → t`, `VOWEL SIGN AA → a`), and strips accents with NFKD. There are no external tables or downloads.
# * **Names**: `basic` (case, translit, punctuation, `&→and`, TLD removal, `L.L.C.→llc`) → `core` (legal forms and stop words removed,
#   leetspeak repaired, learned synonym map) → `squash` (no spaces) → `skeleton` (consonant skeleton, robust to vowels and typos).
# * **Addresses**: `basic` (placeholders removed, ordinals `3rd→3`) → `canon` (multilingual street-type map + learned map from train pairs).

# %%
_SPECIAL = {'œ': 'oe', 'æ': 'ae', 'ß': 'ss', 'ø': 'o', 'ł': 'l', 'đ': 'd', 'ð': 'd', 'þ': 'th', 'ı': 'i', 'ĳ': 'ij',
            '’': "'", '‘': "'", '´': "'", '`': "'", '–': '-', '—': '-', '“': '"', '”': '"', '«': '"', '»': '"',
            '।': ' ', '॥': ' '}
_VOWELS = {'a': 'a', 'aa': 'a', 'i': 'i', 'ii': 'i', 'u': 'u', 'uu': 'u', 'e': 'e', 'ee': 'e',
           'ai': 'ai', 'o': 'o', 'oo': 'o', 'au': 'au'}
_REP = re.compile(r'(.)\1+')


def _indic_letter(rest):
    w = rest.lower().replace('-', ' ').split()
    if not w:
        return ''
    if w[0] == 'vocalic':
        return 'ri' if len(w) > 1 and w[1].startswith('r') else 'li'
    if w[0] in ('candra', 'short', 'independent', 'small', 'capital', 'final', 'medial', 'initial') and len(w) > 1:
        w = w[1:]
    tok = w[-1]
    if tok in _VOWELS:
        return _VOWELS[tok]
    if len(tok) > 1 and tok.endswith('a'):
        tok = tok[:-1]                       # consonant: drop inherent vowel (KA -> k, TTA -> t)
    return _REP.sub(r'\1', tok)


def _char_map(ch):
    if ch in _SPECIAL:
        return _SPECIAL[ch]
    cat = unicodedata.category(ch)
    name = unicodedata.name(ch, '')
    if cat in ('Mn', 'Mc', 'Me', 'Cf'):          # combining marks, Indic vowel signs, ZWJ...
        if 'VOWEL SIGN' in name:
            v = name.split('VOWEL SIGN ', 1)[1].lower().split()
            if v and v[0] == 'vocalic':
                return 'ri'
            return _VOWELS.get(v[-1], '') if v else ''
        if name.endswith('ANUSVARA') or name.endswith('CANDRABINDU'):
            return 'n'
        if name.endswith('VISARGA'):
            return 'h'
        return ''
    if cat == 'Nd':
        d = unicodedata.decimal(ch, None)
        return str(d) if d is not None else ' '
    base = ''.join(c for c in unicodedata.normalize('NFKD', ch) if not unicodedata.combining(c))
    if base and base.isascii():
        return base.lower()
    if cat.startswith('L') and ' LETTER ' in name:
        return _indic_letter(name.split(' LETTER ', 1)[1])
    if cat[0] in 'PSZ':
        return ' '
    return ch


class _TranslitTable(dict):
    def __missing__(self, key):
        v = _char_map(chr(key))
        self[key] = v
        return v


TRANSLIT = _TranslitTable()


def translit(s):
    return s if s.isascii() else s.translate(TRANSLIT)


LEGAL_FORMS = {
    'inc', 'incorporated', 'llc', 'llp', 'pllc', 'lp', 'ltd', 'limited', 'pvt', 'private', 'privet', 'praivet', 'prayvet',
    'corp', 'corporation', 'co', 'cos', 'company', 'plc', 'pte', 'opc', 'gmbh', 'ag', 'sa', 'sas', 'sasu', 'sarl', 'eurl',
    'sci', 'snc', 'scs', 'scop', 'selarl', 'sprl', 'bv', 'nv', 'srl', 'spa',
    'the', 'and', 'of', 'et', 'de', 'du', 'des', 'la', 'le', 'les', 'l', 'd',
}
ADDR_HAND_MAP = {
    'street': 'st', 'str': 'st', 'strt': 'st', 'avenue': 'ave', 'av': 'ave', 'avn': 'ave', 'road': 'rd', 'drive': 'dr',
    'drv': 'dr', 'lane': 'ln', 'court': 'ct', 'crt': 'ct', 'place': 'pl', 'boulevard': 'blvd', 'boul': 'blvd',
    'bd': 'blvd', 'blv': 'blvd', 'highway': 'hwy', 'parkway': 'pkwy', 'pky': 'pkwy', 'circle': 'cir', 'terrace': 'ter',
    'terr': 'ter', 'trail': 'trl', 'square': 'sq', 'suite': 'ste', 'apartment': 'apt', 'appt': 'apt', 'building': 'bldg',
    'floor': 'fl', 'flr': 'fl', 'north': 'n', 'south': 's', 'east': 'e', 'west': 'w', 'northeast': 'ne',
    'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw', 'mount': 'mt', 'fort': 'ft', 'route': 'rte',
    'expressway': 'expy', 'freeway': 'fwy', 'junction': 'jct', 'heights': 'hts', 'center': 'ctr', 'centre': 'ctr',
    'point': 'pt', 'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5', 'sixth': '6',
    'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10', 'number': 'no', 'nr': 'near', 'opposite': 'opp',
    'sector': 'sec', 'sect': 'sec', 'district': 'dist', 'r': 'rue', 'chemin': 'ch', 'allee': 'all', 'impasse': 'imp',
    'faubourg': 'fbg', 'saint': 'st', 'sainte': 'ste', 'residence': 'res', 'batiment': 'bat',
}
ADDR_PLACEHOLDER_TOKENS = {'null', 'none', 'nan', 'na', 'nil', 'unknown', 'undefined', 'notavailable'}
NAME_MAP, ADDR_MAP = {}, {}          # learned from TRAIN fit-fold positive pairs in §6/§7 (empty until then)

_WWW = re.compile(r'\bwww\.')
_TLD = re.compile(r'\.(?:com|net|org|in|co|biz|info|io|us|fr|uk|online|site|store|shop)\b')
_APOS = re.compile(r"['’`´]")
_NON_ALNUM = re.compile(r'[^0-9a-z]+')
_SINGLE_RUN = re.compile(r'\b(?:[a-z] )+[a-z]\b')
_ORDINAL = re.compile(r'\b(\d+)(?:st|nd|rd|th)\b')
_NUM = re.compile(r'\d+')
_VOW = re.compile(r'[aeiouy]')
_TRADE = re.compile(r'\b(?:ta|dba|aka|fka|trading as|doing business as)\b')
_LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't'})


def _merge_singles(m):
    return m.group(0).replace(' ', '')


def name_basic(raw):
    s = translit(raw.lower())
    s = _TLD.sub(' ', _WWW.sub(' ', s))
    s = _APOS.sub('', s.replace('&', ' and '))
    s = _NON_ALNUM.sub(' ', s).strip()
    return _SINGLE_RUN.sub(_merge_singles, s)


def _fix_leet(t):
    if t.isalpha() or t.isdigit():
        return t
    if sum(c.isdigit() for c in t) == 1 and len(t) >= 2:
        return t.translate(_LEET)
    return t


def name_core(basic):
    toks = [NAME_MAP.get(t, t) for t in (_fix_leet(t) for t in basic.split())]
    core = [t for t in toks if t not in LEGAL_FORMS]
    return ' '.join(core if core else toks)


def skel_tok(t):
    if not t or t.isdigit():
        return t
    t = t.replace('ph', 'f').replace('ck', 'k').replace('q', 'k').replace('w', 'v').replace('z', 's').replace('sh', 's')
    return _REP.sub(r'\1', t[0] + _VOW.sub('', t[1:]))


def name_skel(core):
    return ' '.join(skel_tok(t) for t in core.split())


def name_alt(basic):
    """Trade-name part after 't/a', 'dba', 'aka' ... (empty if none)."""
    parts = _TRADE.split(basic)
    return name_core(parts[-1].strip()) if len(parts) > 1 and parts[-1].strip() else ''


def acronym(core):
    t = [x for x in core.split() if not x.isdigit()]
    return ''.join(x[0] for x in t) if len(t) >= 2 else ''


def addr_basic(raw):
    s = translit(raw.lower())
    s = _APOS.sub('', s.replace('&', ' and '))
    s = _ORDINAL.sub(r'\1', _NON_ALNUM.sub(' ', s))
    s = _SINGLE_RUN.sub(_merge_singles, s.strip())
    return ' '.join(t for t in s.split() if t not in ADDR_PLACEHOLDER_TOKENS)


def addr_canon(basic):
    return ' '.join(ADDR_MAP.get(t2, t2) for t2 in (ADDR_HAND_MAP.get(t, t) for t in basic.split()))


def addr_keys(canon):
    """house-number + street-token keys, robust to component re-ordering ('OR, Eugene, 3900 River Rd')."""
    t = canon.split()
    keys = []
    for i, x in enumerate(t[:-1]):
        if x.isdigit() and not t[i + 1].isdigit() and len(t[i + 1]) >= 3:
            keys.append((x.lstrip('0') or '0') + '|' + skel_tok(t[i + 1])[:4])
            if len(keys) >= 3:
                break
    return ' '.join(keys)


def name_flags(raw):
    low = raw.lower()
    b = 0
    if _DOMAIN_RE.search(low):
        b |= 1
    if not raw.isascii():
        b |= 2
    if raw.isupper():
        b |= 4
    if raw[:1] and not raw[:1].isalnum():
        b |= 8
    return b


def first_number(canon):
    m = _NUM.search(canon)
    return int(m.group(0)[:9]) if m else -1


def learn_token_map(left, right, min_count=40, min_prob=0.5, max_diff=2):
    """Data-driven synonym map from positive pairs: token r (noisy side) → token l (S1 side) when r systematically
    replaces l. Keys and targets are kept disjoint (no chains)."""
    pair_c, r_c = Counter(), Counter()
    for a, b in zip(left, right):
        A, B = set(a.split()), set(b.split())
        L, R = A - B, B - A
        if not L or not R or len(L) > max_diff or len(R) > max_diff:
            continue
        w = 1.0 / (len(L) * len(R))
        for r in R:
            r_c[r] += 1
            for l in L:
                pair_c[(r, l)] += w
    mp, targets = {}, set()
    for (r, l), c in sorted(pair_c.items(), key=lambda x: -x[1]):
        if c < min_count:
            break
        if r in mp or l in mp or r in targets or r == l or r.isdigit() or l.isdigit():
            continue
        if c / r_c[r] >= min_prob:
            mp[r] = l
            targets.add(l)
    return mp


for s in ['राम मार्केटिंग प्राइवेट लिमिटेड', 'Smt SMB (INDIA) ÉDUCATION PRIVATE LIMITED', 'victorylaboratories.com',
          'Jordan & Hartley Business L.L.C.', 'Community League 0f New Philadelphia Corp', 'Iridova t/a Etti\'s Audio Installation']:
    b = name_basic(s)
    print(f'{s!r:50} basic={b!r:45} core={name_core(b)!r:35} skel={name_skel(name_core(b))!r} alt={name_alt(b)!r}')
for s in ['OR, Eugene, 3900 River Road', '520 3TH STREET, NEW PHILADELPHIA, OH', 'B-4/18 Tyagi Bhavanashok Vihar Ph-ii, North Delhi, दिल्ली',
          '5403 BENNINTGON AVE, NULL, KANSAS CITY, MO']:
    b = addr_basic(s)
    print(f'{s!r:62} basic={b!r:50} canon={addr_canon(b)!r:45} keys={addr_keys(addr_canon(b))!r}')

# %% [markdown]
# ## 5. Singleton analysis
# About 5% of S1 entities have **no** match. Under F0.5 each of them scores 1 when nothing is predicted and 0 otherwise.
# This section asks whether singletons can be recognised from the S1 record alone (length, missingness, country, commonness, token rarity).
# Candidate-based evidence (candidate counts, best retrieval score) is added in §11, once candidates exist.

# %%
with Timer('singleton analysis'):
    S1['n_basic'] = umap(S1.business_name, name_basic)
    S1['a_basic'] = umap(S1.business_address, addr_basic)
    sg = pd.DataFrame({
        'singleton': N_TRUE_ALL == 0,
        'country': S1.country.astype(object).to_numpy(),
        'name_len': S1.business_name.str.len().to_numpy(np.float32),
        'name_tokens': S1.n_basic.str.count(' ').to_numpy(np.float32) + 1,
        'addr_len': S1.business_address.str.len().to_numpy(np.float32),
        'addr_tokens': S1.a_basic.str.count(' ').to_numpy(np.float32) + 1,
        'addr_missing': (S1.a_basic == '').to_numpy(bool),
        'addr_has_number': S1.a_basic.str.contains(r'\d').to_numpy(bool),
        'name_dup_in_s1': S1.n_basic.map(S1.n_basic.value_counts()).to_numpy(np.float32),
        'addr_dup_in_s1': S1.a_basic.map(S1.a_basic.value_counts()).to_numpy(np.float32),
    })
    _tok = S1.n_basic.astype(object).str.split().explode()
    _tf = _tok.map(_tok.value_counts())
    sg['name_min_token_freq'] = _tf.groupby(level=0).min().reindex(range(NS1)).fillna(0).to_numpy(np.float32)
    sg["has_legal_form"] = _tok.isin(LEGAL_FORMS).groupby(level=0).any().reindex(range(NS1)).fillna(False).astype(bool).to_numpy()
    del _tok, _tf

num_cols = [c for c in sg.columns if c not in ('singleton', 'country')]
cmp = sg.groupby('singleton')[num_cols].agg(['mean', 'median']).T.unstack()
cmp.columns = [f'{"singleton" if a else "matched"}_{b}' for a, b in cmp.columns]
display(cmp.round(3))
display(sg.groupby('country').singleton.agg(['mean', 'size']).rename(columns={'mean': 'singleton_rate'}))

fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
b1 = pd.cut(sg.name_dup_in_s1, [0, 1, 2, 5, 20, 1e9], labels=['1', '2', '3-5', '6-20', '>20'])
sg.groupby(b1).singleton.mean().plot.bar(ax=ax[0], rot=0, color='indianred')
ax[0].set_title('singleton rate vs how many S1 share the name'); ax[0].axhline(sg.singleton.mean(), ls='--', c='k')
b2 = pd.qcut(sg.name_min_token_freq.rank(method='first'), 10, labels=False)
sg.groupby(b2).singleton.mean().plot.bar(ax=ax[1], rot=0, color='indianred')
ax[1].set_title('singleton rate by rarest-name-token frequency decile (0 = rarest)'); ax[1].axhline(sg.singleton.mean(), ls='--', c='k')
plt.tight_layout(); plt.show()

# How predictable is "singleton" from S1 attributes alone?
_smp = sg.sample(min(300_000, NS1), random_state=SEED)
_X = _smp[num_cols].astype(np.float32).assign(country=pd.factorize(_smp.country)[0])
_y = _smp.singleton.to_numpy()
_cut = int(len(_X) * 0.7)
if lgb is not None:
    _m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=SEED)
    _m.fit(_X.iloc[:_cut], _y[:_cut])
    SINGLETON_AUC = roc_auc_score(_y[_cut:], _m.predict_proba(_X.iloc[_cut:])[:, 1])
else:
    from sklearn.linear_model import LogisticRegression
    _m = LogisticRegression(max_iter=500).fit(_X.iloc[:_cut].fillna(0), _y[:_cut])
    SINGLETON_AUC = roc_auc_score(_y[_cut:], _m.predict_proba(_X.iloc[_cut:].fillna(0))[:, 1])
print(f'S1-attribute-only singleton classifier AUC = {SINGLETON_AUC:.3f}')
finding('Singletons from S1 attributes alone',
        f'singleton rate {sg.singleton.mean():.2%}; attribute-only AUC {SINGLETON_AUC:.3f}; '
        f'rate for names shared by >20 S1 = {sg.loc[sg.name_dup_in_s1 > 20, "singleton"].mean():.2%}',
        'If S1 attributes alone separated singletons we could gate them before matching; a weak AUC means the '
        'no-match decision must come from the (lack of) candidate evidence.',
        'No separate singleton gate. S1-side attributes (name/address commonness, lengths, missingness) are fed to the pair '
        'model, and singletons are handled by the anchor threshold (an S1 gets matches only if its best candidate is '
        'confident) — tuned for macro F0.5 in §16–§17.')
del _smp, _X, _y, _m
gc.collect()
