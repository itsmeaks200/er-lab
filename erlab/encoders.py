"""Dense encoders (GPU):
* HashEncoder — hashed char-3gram/token EmbeddingBag + MLP trained from scratch with in-batch InfoNCE on fit positives
  (SC-Block-style supervised contrastive blocking; no pretrained weights).
* Sentence-transformers (pretrained MIT/Apache models, optional contrastive fine-tuning with MultipleNegativesRankingLoss).
"""
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize as sk_normalize

from .utils import log

ENC_FIELDS = ['nchar', 'ntok', 'nskel', 'atok', 'anum', 'abig']


def _torch():
    import torch
    return torch


def enc_inputs(M, rows, R, bucket_bits):
    torch = _torch()
    B, nf = 2 ** bucket_bits, len(ENC_FIELDS)
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


def build_hash_encoder(n_buckets, dim):
    torch = _torch()

    class HashEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.EmbeddingBag(n_buckets, dim, mode='sum', sparse=True)
            torch.nn.init.normal_(self.emb.weight, std=0.05)
            self.mlp = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, 2 * dim), torch.nn.GELU(),
                                           torch.nn.Linear(2 * dim, dim))

        def forward(self, idx, off, w):
            x = self.emb(idx, off, per_sample_weights=w)
            return torch.nn.functional.normalize(x + self.mlp(x), dim=-1)
    return HashEncoder()


def _dev():
    return 'cuda' if _torch().cuda.is_available() else 'cpu'


def train_hash_encoder(Qm, Pm, R, pos_qi, pos_pi, ec, seed=42):
    torch = _torch()
    dev = _dev()
    torch.manual_seed(seed)
    model = build_hash_encoder(len(ENC_FIELDS) * 2 ** ec['bucket_bits'], ec['dim']).to(dev)
    opt_s = torch.optim.SparseAdam(list(model.emb.parameters()), lr=ec['lr'])
    opt_d = torch.optim.AdamW(model.mlp.parameters(), lr=1e-3)
    rng = np.random.RandomState(seed)
    order = np.argsort(pos_qi, kind='stable')
    sq, spi = pos_qi[order], pos_pi[order]
    uq, start, counts = np.unique(sq, return_index=True, return_counts=True)
    bs, tau = min(ec['batch'], max(len(uq) // 2, 2)), ec['tau']
    for ep in range(ec['epochs']):
        pick = start + (rng.rand(len(uq)) * counts).astype(np.int64)   # one positive per S1 per epoch
        a_rows, b_rows = sq[pick], spi[pick]
        perm = rng.permutation(len(a_rows))
        tot, nb = 0.0, 0
        model.train()
        for s in range(0, len(perm) - bs + 1, bs):
            bi = perm[s:s + bs]
            ea = model(*[t.to(dev) for t in enc_inputs(Qm, a_rows[bi], R, ec['bucket_bits'])])
            eb = model(*[t.to(dev) for t in enc_inputs(Pm, b_rows[bi], R, ec['bucket_bits'])])
            logits = ea @ eb.T / tau
            lab = torch.arange(len(bi), device=dev)
            loss = (torch.nn.functional.cross_entropy(logits, lab) + torch.nn.functional.cross_entropy(logits.T, lab)) / 2
            opt_s.zero_grad(); opt_d.zero_grad()
            loss.backward()
            opt_s.step(); opt_d.step()
            tot += loss.item(); nb += 1
        log(f'   hash-encoder epoch {ep + 1}/{ec["epochs"]}: loss {tot / max(nb, 1):.4f}')
    model.eval()
    return model


def embed_hash(model, M, R, n, bucket_bits, dim, bs=16384):
    torch = _torch()
    dev = _dev()
    out = np.empty((n, dim), np.float16)
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(n, s + bs)
            out[s:e] = model(*[t.to(dev) for t in enc_inputs(M, slice(s, e), R, bucket_bits)]).half().cpu().numpy()
    return out


# ---------------------------------------------------------------- sentence-transformers
def st_texts(df, prefix=''):
    n = df.business_name.astype(object).to_numpy()
    a = df.business_address.astype(object).to_numpy()
    return [f'{prefix}{x} | {y}' for x, y in zip(n, a)]


def load_st(name, max_len):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name, device=_dev())
    m.max_seq_length = max_len
    if _dev() == 'cuda':
        m = m.half()
    return m


def embed_st(model, texts, batch, slice_size=200_000):
    from .utils import Progress
    out = np.empty((len(texts), model.get_sentence_embedding_dimension()), np.float16)
    pg = Progress('sentence-transformer embeddings', len(texts), 'texts')
    for s in range(0, len(texts), slice_size):
        e = model.encode(texts[s:s + slice_size], batch_size=batch, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
        out[s:s + len(e)] = e.astype(np.float16)
        pg.update(s + len(e))
    pg.done()
    return out


def finetune_st(model, q_texts, p_texts, sc, seed=42):
    """Contrastive fine-tuning (MultipleNegativesRankingLoss, in-batch negatives) on (S1 text, matched record text)."""
    torch = _torch()
    dev = _dev()
    torch.manual_seed(seed)
    model = model.float().to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=sc['ft_lr'], weight_decay=0.01)
    n, bs = len(q_texts), sc['ft_batch']
    steps = max(1, sc['ft_epochs'] * (n // bs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * steps))) * max(0.0, 1 - s / steps))
    scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda')
    rng = np.random.RandomState(seed)
    step = 0
    for ep in range(sc['ft_epochs']):
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n - bs + 1, bs):
            bi = perm[s:s + bs]
            fa = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in model.tokenize([q_texts[i] for i in bi]).items()}
            fb = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in model.tokenize([p_texts[i] for i in bi]).items()}
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dev == 'cuda'):
                ea = torch.nn.functional.normalize(model(fa)['sentence_embedding'].float(), dim=-1)
                eb = torch.nn.functional.normalize(model(fb)['sentence_embedding'].float(), dim=-1)
                logits = ea @ eb.T * 20.0                     # MultipleNegativesRankingLoss scale
                lab = torch.arange(len(bi), device=dev)
                loss = (torch.nn.functional.cross_entropy(logits, lab) + torch.nn.functional.cross_entropy(logits.T, lab)) / 2
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); step += 1
            if step % 500 == 0:
                log(f'   st fine-tune step {step}/{steps} loss {tot / ((s // bs) + 1):.4f}')
    model.eval()
    if dev == 'cuda':
        model = model.half()
    return model
