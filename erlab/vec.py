"""Frame preparation (normalised columns + record attributes) and hashed sparse representations."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from . import text as T
from .utils import umap, run_parallel, Timer, log

REC_COLS = ['n_len', 'n_ntok', 'a_len', 'a_ntok', 'f_domain', 'f_nonascii', 'f_upper', 'f_junk', 'a_missing',
            'nf_s1', 'nf_pool', 'af_s1', 'af_pool']
TEXT_COLS = ['entity_id', 'business_name', 'business_address', 'country', 'country_key', 'n_basic', 'n_core',
             'n_squash', 'n_skel', 'n_alt', 'a_basic', 'a_canon']


def _ntok(ser):
    c = ser.str.count(' ').to_numpy(np.float32) + 1
    c[(ser == '').to_numpy(bool)] = 0
    return c


def prepare_basic(df):
    if 'country_key' not in df:
        df['country_key'] = df.country.str.strip().str.lower()
    if 'n_basic' not in df:
        df['n_basic'] = umap(df.business_name, T.name_basic)
    if 'a_basic' not in df:
        df['a_basic'] = umap(df.business_address, T.addr_basic)
    return df


def prepare_frame(df):
    """Frozen normalisation used identically for train and test (requires NAME_MAP/ADDR_MAP already set)."""
    prepare_basic(df)
    df['n_core'] = umap(df.n_basic, T.name_core)
    df['n_squash'] = df.n_core.str.replace(' ', '', regex=False)
    df['n_skel'] = umap(df.n_core, T.name_skel)
    df['n_alt'] = umap(df.n_basic, T.name_alt)
    df['a_canon'] = umap(df.a_basic, T.addr_canon)
    df['n_len'] = df.n_basic.str.len().to_numpy(np.float32)
    df['n_ntok'] = _ntok(df.n_core)
    df['a_len'] = df.a_canon.str.len().to_numpy(np.float32)
    df['a_ntok'] = _ntok(df.a_canon)
    fl = umap(df.business_name, T.name_flags, as_str=False).to_numpy(np.int64)
    df['f_domain'] = ((fl & 1) > 0).astype(np.float32)
    df['f_nonascii'] = ((fl & 2) > 0).astype(np.float32)
    df['f_upper'] = ((fl & 4) > 0).astype(np.float32)
    df['f_junk'] = ((fl & 8) > 0).astype(np.float32)
    df['a_missing'] = (df.a_canon == '').to_numpy(bool).astype(np.float32)
    df['num0'] = umap(df.a_canon, T.first_number, as_str=False).to_numpy(np.int64)
    return df


def add_freq_features(s1df, pdf):
    """Commonness of the core name / canonical address among ALL S1 entities and in the pool (log counts)."""
    for col, tag in [('n_core', 'nf'), ('a_canon', 'af')]:
        vs, vp = s1df[col].value_counts(), pdf[col].value_counts()
        for df in (s1df, pdf):
            empty = (df[col] == '').to_numpy(bool)
            a = np.log1p(df[col].map(vs).fillna(0).to_numpy(np.float32)); a[empty] = 0
            b = np.log1p(df[col].map(vp).fillna(0).to_numpy(np.float32)); b[empty] = 0
            df[tag + '_s1'], df[tag + '_pool'] = a, b


# ---------------------------------------------------------------- hashed sparse matrices
def an_split(s): return s.split()
def an_atok(s): return [t for t in s.split() if not t.isdigit()]
def an_anum(s): return list({x.lstrip('0') or '0' for x in T._NUM.findall(s)})
def an_apost(s): return T.postal_codes(s)


def an_abig(s):
    t = s.split()
    return [a + '_' + b for a, b in zip(t, t[1:])]


MAT_SRC = {'ntok': 'n_core', 'nskel': 'n_skel', 'nchar': 'n_core', 'atok': 'a_canon', 'anum': 'a_canon',
           'abig': 'a_canon', 'apost': 'a_canon'}
TOKEN_SPACES = ['ntok', 'nskel', 'atok', 'anum', 'abig', 'apost', 'nchar']


def make_hv(cfg):
    hb, cb = 2 ** cfg['vec']['hash_bits'], 2 ** cfg['vec']['char_bits']

    def hv(an):
        return HashingVectorizer(analyzer=an, n_features=hb, alternate_sign=False, norm=None, binary=True, dtype=np.float32)
    return {'ntok': hv(an_split), 'nskel': hv(an_split), 'atok': hv(an_atok), 'anum': hv(an_anum),
            'abig': hv(an_abig), 'apost': hv(an_apost),
            'nchar': HashingVectorizer(analyzer='char_wb', ngram_range=(3, 3), n_features=cb, alternate_sign=False,
                                       norm='l2', dtype=np.float32, lowercase=False)}


def _hv_transform(args):
    hv, docs = args
    return hv.transform(docs)


def vectorize(ser, hv, chunk=250_000):
    docs = ser.astype(object).tolist()
    parts = run_parallel(_hv_transform, [(hv, docs[i:i + chunk]) for i in range(0, len(docs), chunk)])
    X = sp.vstack(parts, format='csr') if len(parts) > 1 else parts[0].tocsr()
    X.sort_indices()
    return X


def build_matrices(df, cfg, tag):
    hv = make_hv(cfg)
    with Timer(f'vectorise {tag} ({len(df):,} records)'):
        M = {k: vectorize(df[MAT_SRC[k]], hv[k]) for k in hv}
        log('   nnz/record: ' + ', '.join(f'{k}={M[k].nnz / max(len(df), 1):.1f}' for k in M))
    return M


def fit_idf(Pm, cfg):
    """Pool IDF statistics (label-free; recomputed on the test pool at inference)."""
    rc = cfg['retr']
    N = Pm['ntok'].shape[0]
    R = dict(N=N, DF={}, IDF={}, IDF2={}, RARE={}, W={})
    cap = max(rc['df_cap_min'], rc['df_cap_frac'] * N)
    ccap = max(rc['df_cap_min'], rc['char_df_cap_frac'] * N)
    R['cap'] = cap
    mult = {'ntok': 1.0, 'nskel': rc['skel_weight'], 'atok': 1.0, 'anum': rc['num_weight'],
            'abig': rc['bigram_weight'], 'apost': 1.0, 'nchar': 1.0}
    for k in TOKEN_SPACES:
        d = np.bincount(Pm[k].indices, minlength=Pm[k].shape[1]).astype(np.float32)
        R['DF'][k] = d
        R['IDF'][k] = (np.log((N + 1) / (d + 1)) + 1).astype(np.float32)
        R['IDF2'][k] = R['IDF'][k] ** 2
        R['RARE'][k] = (d <= 50).astype(np.float32)
        w = R['IDF'][k] * mult[k]
        w[d > (ccap if k == 'nchar' else cap)] = 0
        R['W'][k] = w.astype(np.float32)
    return R
