"""Ditto-style cross-encoder reranker: serialise both records, fine-tune a small multilingual transformer
(MIT/Apache licence) on fit-fold candidate pairs (positives + hard negatives), score the uncertain band only."""
import math

import numpy as np

from .utils import log


def serialise(df, idx):
    n = df.business_name.take(idx).tolist()
    a = df.business_address.take(idx).tolist()
    return [f'name: {x} | address: {y}' for x, y in zip(n, a)]


def _dev():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def train_crossenc(cc, a_txt, b_txt, y, seed=42):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    dev = _dev()
    tok = AutoTokenizer.from_pretrained(cc['model'])
    model = AutoModelForSequenceClassification.from_pretrained(cc['model'], num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cc['lr'], weight_decay=0.01)
    n, bs = len(y), cc['batch']
    steps = cc['epochs'] * math.ceil(n / bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda')
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(seed)
    yt = torch.from_numpy(np.asarray(y, np.float32))
    for ep in range(cc['epochs']):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for j, s in enumerate(range(0, n, bs)):
            bi = perm[s:s + bs]
            enc = tok([a_txt[i] for i in bi], [b_txt[i] for i in bi], truncation=True, max_length=cc['max_len'],
                      padding=True, return_tensors='pt').to(dev)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dev == 'cuda'):
                logit = model(**enc).logits.squeeze(-1)
                loss = lossf(logit.float(), yt[bi].to(dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item()
            if j % 2000 == 0:
                log(f'   cross-encoder ep{ep + 1} step {j}/{math.ceil(n / bs)} loss {tot / (j + 1):.4f}')
    model.eval()
    return model, tok


def predict_crossenc(model, tok, a_txt, b_txt, cc, bs=1024):
    import torch
    dev = _dev()
    out = np.empty(len(a_txt), np.float32)
    with torch.no_grad():
        for s in range(0, len(a_txt), bs):
            enc = tok(a_txt[s:s + bs], b_txt[s:s + bs], truncation=True, max_length=cc['max_len'], padding=True,
                      return_tensors='pt').to(dev)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dev == 'cuda'):
                out[s:s + bs] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    return out


def select_rows(qi, p1, band, top_n):
    """rows worth re-scoring: p1 inside the uncertain band AND within the S1's top-n."""
    import pandas as pd
    rk = pd.Series(p1).groupby(qi).rank(ascending=False, method='first').to_numpy()
    return np.flatnonzero((p1 >= band[0]) & (p1 <= band[1]) & (rk <= top_n))
