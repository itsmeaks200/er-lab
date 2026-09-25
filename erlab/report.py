"""Compact, paste-friendly reporting.
results/runs/<EXP>.md        full report of one experiment (tables)
results/LEADERBOARD.csv      one line per model/rule evaluation (append-only)
results/PASTE_BACK.md        the text to copy back to the chat (latest invocation only, kept short)
results/summary.json         machine-readable latest metrics (small; downloadable)"""
import os
import time
import json
import platform

import numpy as np
import pandas as pd

from .utils import TIMINGS, rss_gb, save_json

_BUF = []


def _fmt(df, max_rows=40):
    if df is None or len(df) == 0:
        return '(empty)'
    df = df.head(max_rows).copy()
    for c in df.columns:
        if df[c].dtype.kind == 'f':
            df[c] = df[c].round(4)
    try:
        return df.to_markdown()
    except Exception:
        return '```\n' + df.to_string() + '\n```'


class Report:
    def __init__(self, cfg, exp_id, desc):
        self.cfg, self.exp_id, self.desc = cfg, exp_id, desc
        self.lines = [f'## {exp_id} — {desc}', f'_{time.strftime("%Y-%m-%d %H:%M")} on {platform.node()}_']
        self.short = [f'### {exp_id}: {desc}']
        self.metrics = {}

    def h(self, t):
        self.lines.append(f'\n### {t}')

    def text(self, t, short=False):
        self.lines.append(t)
        if short:
            self.short.append(t)

    def table(self, name, df, short=False, max_rows=40, short_rows=15):
        self.lines += [f'\n**{name}**', _fmt(df, max_rows)]
        if short:
            self.short += [f'**{name}**', _fmt(df, short_rows)]

    def metric(self, **kw):
        self.metrics.update(kw)

    def leaderboard(self, row):
        rd = self.cfg['paths']['results_dir']
        os.makedirs(rd, exist_ok=True)
        path = os.path.join(rd, 'LEADERBOARD.csv')
        row = dict(time=time.strftime('%m-%d %H:%M'), exp=self.exp_id, **row)
        df = pd.DataFrame([row])
        df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)

    def close(self, status='OK', error=None):
        rd = self.cfg['paths']['results_dir']
        os.makedirs(os.path.join(rd, 'runs'), exist_ok=True)
        tm = pd.DataFrame(TIMINGS[-12:])
        self.lines += ['\n**timings (last steps)**', _fmt(tm), f'peak-ish RSS now {rss_gb():.1f} GB']
        if error:
            self.lines.append('\n**ERROR**\n```\n' + error[-6000:] + '\n```')
            self.short.append('**ERROR**\n```\n' + error[-2500:] + '\n```')
        self.short.append(f'status={status}; metrics={json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.metrics.items()})}')
        with open(os.path.join(rd, 'runs', f'{self.exp_id}.md'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(self.lines) + '\n')
        _BUF.append('\n'.join(self.short))
        with open(os.path.join(rd, 'PASTE_BACK.md'), 'w', encoding='utf-8') as fh:
            fh.write('# PASTE BACK (copy everything below into the chat)\n\n' + '\n\n'.join(_BUF) + '\n')
        s = os.path.join(rd, 'summary.json')
        old = json.load(open(s)) if os.path.exists(s) else {}
        old[self.exp_id] = dict(status=status, desc=self.desc, **self.metrics)
        save_json(old, s)
        print('\n'.join(self.short), flush=True)


def reset_paste_back(cfg):
    _BUF.clear()
