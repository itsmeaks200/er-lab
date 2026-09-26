"""Ditto-style cross-encoder reranker: serialise both records, fine-tune a multilingual transformer or a decoder LLM with
LoRA (MIT/Apache licence, <=8B) on candidate pairs of the 'enc' fold (positives + hard negatives; never the rows the stage-2 model
trains on), score the uncertain band only. Batches are length-sorted; training and scoring have time/size caps."""
import math
import time

import numpy as np

from .utils import log


def serialise(df, idx):
    n = df.business_name.take(idx).tolist()
    a = df.business_address.take(idx).tolist()
    return [f'name: {x} | address: {y}' for x, y in zip(n, a)]


def _dev():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def is_llm(cc):
    return cc.get('arch', 'encoder') == 'llm'


def load_model(cc):
    """encoder: MIT/Apache bi-directional transformer + fresh classification head, full fine-tune (fp16 autocast).
    llm: decoder LLM (e.g. Qwen2.5-1.5B / 7B, Apache-2.0) as a sequence classifier with LoRA adapters (bf16), optionally
    4-bit NF4 quantised (QLoRA; needs bitsandbytes) with gradient checkpointing."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import os
    dev = _dev()
    if is_llm(cc) and os.environ.get('ER_SMOKE_LLM'):       # smoke tests only: tiny stand-in LLM, no 4-bit
        cc = dict(cc, model=os.environ['ER_SMOKE_LLM'], load_4bit=False)
    tok = AutoTokenizer.from_pretrained(cc['model'])
    if not is_llm(cc):
        return AutoModelForSequenceClassification.from_pretrained(cc['model'], num_labels=1).to(dev), tok
    from peft import LoraConfig, get_peft_model
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    kw = dict(num_labels=1, torch_dtype=torch.bfloat16)
    if cc.get('load_4bit'):
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
        kw['device_map'] = {'': 0}
    model = AutoModelForSequenceClassification.from_pretrained(cc['model'], **kw)
    model.config.pad_token_id = tok.pad_token_id
    if cc.get('load_4bit'):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif cc.get('grad_ckpt'):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    r = cc.get('lora_r', 16)
    model = get_peft_model(model, LoraConfig(task_type='SEQ_CLS', r=r, lora_alpha=2 * r, lora_dropout=0.05,
                                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    if not cc.get('load_4bit'):
        model = model.to(dev)
    return model, tok


def encode(tok, a, b, cc, dev):
    if is_llm(cc):
        txt = [f'Record A: {x}\nRecord B: {y}\nDo A and B refer to the same real-world business?' for x, y in zip(a, b)]
        return tok(txt, truncation=True, max_length=cc['max_len'], padding=True, return_tensors='pt').to(dev)
    return tok(a, b, truncation=True, max_length=cc['max_len'], padding=True, return_tensors='pt').to(dev)


def _autocast(cc, dev):
    import torch
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16 if is_llm(cc) else torch.float16, enabled=dev == 'cuda')


def train_crossenc(cc, a_txt, b_txt, y, seed=42):
    import torch
    from transformers import get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    dev = _dev()
    model, tok = load_model(cc)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cc['lr'], weight_decay=0.01)
    n, bs, acc = len(y), cc['batch'], max(1, cc.get('grad_accum', 1))
    steps = cc['epochs'] * math.ceil(n / (bs * acc))
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda' and not is_llm(cc))     # bf16 needs no loss scaling
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(seed)
    yt = torch.from_numpy(np.asarray(y, np.float32))
    t0, budget = time.time(), cc.get('train_minutes', 0) * 60
    every = cc.get('log_every', 2000 if not is_llm(cc) else 100)
    log(f'   cross-encoder {cc["model"]} ({cc.get("arch", "encoder")}{", 4-bit" if cc.get("load_4bit") else ""}): '
        f'{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params ({sum(p.numel() for p in params) / 1e6:.1f}M trainable), '
        f'{n:,} pairs, batch {bs}x{acc}, {steps:,} optimiser steps' + (f', time cap {budget / 60:.0f} min' if budget else ''))
    stop = False
    for ep in range(cc['epochs']):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        opt.zero_grad()
        for j, s in enumerate(range(0, n, bs)):
            bi = perm[s:s + bs]
            enc = encode(tok, [a_txt[i] for i in bi], [b_txt[i] for i in bi], cc, dev)
            with _autocast(cc, dev):
                logit = model(**enc).logits.squeeze(-1)
                loss = lossf(logit.float(), yt[bi].to(dev))
            scaler.scale(loss / acc).backward()
            if (j + 1) % acc == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
            tot += loss.item()
            if j % every == 0:
                el = time.time() - t0
                log(f'   cross-encoder ep{ep + 1} batch {j}/{math.ceil(n / bs)} loss {tot / (j + 1):.4f} '
                    f'({(j + 1) * bs / max(el, 1e-9):,.0f} pairs/s)')
            if budget and time.time() - t0 > budget:
                log(f'   cross-encoder: time cap reached at ep{ep + 1} batch {j} (loss {tot / (j + 1):.4f})')
                stop = True
                break
        if stop:
            break
    model.eval()
    return model, tok


def predict_crossenc(model, tok, a_txt, b_txt, cc, bs=None):
    """Length-sorted batches (much less padding); returns probabilities in the original order."""
    import torch
    dev = _dev()
    bs = bs or cc.get('pred_batch', 1024)
    order = np.argsort([len(a) + len(b) for a, b in zip(a_txt, b_txt)], kind='stable')
    out = np.empty(len(a_txt), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for j, s in enumerate(range(0, len(order), bs)):
            ix = order[s:s + bs]
            enc = encode(tok, [a_txt[i] for i in ix], [b_txt[i] for i in ix], cc, dev)
            with _autocast(cc, dev):
                out[ix] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
            if j % 500 == 0 and j:
                log(f'   cross-encoder predict {s:,}/{len(order):,} ({s / max(time.time() - t0, 1e-9):,.0f} pairs/s)')
    return out


def select_rows(qi, p1, band, top_n, max_rows=0):
    """rows worth re-scoring: p1 inside the uncertain band AND within the S1's top-n; if more than max_rows, the ones
    closest to p1 = 0.5 (same rule for every fold, so the feature is distributed identically)."""
    import pandas as pd
    rk = pd.Series(p1).groupby(qi).rank(ascending=False, method='first').to_numpy()
    sel = np.flatnonzero((p1 >= band[0]) & (p1 <= band[1]) & (rk <= top_n))
    if max_rows and len(sel) > max_rows:
        sel = np.sort(sel[np.argsort(np.abs(p1[sel] - 0.5), kind='stable')[:max_rows]])
    return sel
