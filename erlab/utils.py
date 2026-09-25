"""Logging, timing, memory, parallel helpers (cross-platform: fork on Linux, spawn on Windows)."""
import os
import sys
import time
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import pyarrow  # noqa: F401
    STR = 'string[pyarrow]'
except ImportError:  # pragma: no cover
    STR = object

T0 = time.time()
TIMINGS = []


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9 if psutil else float('nan')


def log(*msg):
    print(f'[{time.time() - T0:7.0f}s | RSS {rss_gb():6.1f} GB]', *msg, flush=True)


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        log(f'>> {self.name}')
        return self

    def __exit__(self, *a):
        dt = time.time() - self.t
        TIMINGS.append(dict(step=self.name, seconds=round(dt, 1), rss_gb=round(rss_gb(), 2)))
        log(f'<< {self.name}: {dt:.1f}s')


class Progress:
    """Throttled progress lines for long loops: at most one line per `every` seconds with done/total, rate and ETA."""
    def __init__(self, tag, total, unit='items', every=60.0):
        self.tag, self.total, self.unit, self.every = tag, max(int(total), 1), unit, every
        self.t0 = self.last = time.time()
        log(f'   {tag}: started ({total:,} {unit})')

    def update(self, done, force=False):
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        el = now - self.t0
        rate = done / max(el, 1e-9)
        eta = (self.total - done) / max(rate, 1e-9)
        log(f'   {self.tag}: {done:,}/{self.total:,} {self.unit} ({100 * done / self.total:5.1f}%) | {rate:,.0f}/s | '
            f'elapsed {el / 60:.1f} min | ETA {eta / 60:.1f} min')

    def done(self):
        log(f'   {self.tag}: done {self.total:,} {self.unit} in {(time.time() - self.t0) / 60:.1f} min')


def cfg_hash(obj, n=10):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


# ---------------------------------------------------------------- parallel map
N_JOBS = max(1, min(os.cpu_count() or 1, int(os.environ.get('ER_JOBS', '48'))))
_WORKER_STATE = {}


def _init_worker(state):
    """Installs learned maps (and anything else) into worker processes (needed for spawn on Windows)."""
    from . import text
    text.NAME_MAP.clear(); text.NAME_MAP.update(state.get('NAME_MAP', {}))
    text.ADDR_MAP.clear(); text.ADDR_MAP.update(state.get('ADDR_MAP', {}))


def _apply_chunk(args):
    func, items = args
    return [func(x) for x in items]


def _worker_state():
    from . import text
    return dict(NAME_MAP=dict(text.NAME_MAP), ADDR_MAP=dict(text.ADDR_MAP))


def run_parallel(fn, args_list, n_jobs=None):
    n_jobs = n_jobs or N_JOBS
    if n_jobs <= 1 or len(args_list) < 2:
        return [fn(a) for a in args_list]
    kw = {}
    if sys.platform.startswith('linux'):
        import multiprocessing as mp
        kw['mp_context'] = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(args_list)), initializer=_init_worker,
                             initargs=(_worker_state(),), **kw) as ex:
        return list(ex.map(fn, args_list))


def pmap(func, items, chunk=100_000):
    items = list(items)
    if len(items) < 2 * chunk or N_JOBS <= 1:
        return [func(x) for x in items]
    parts = run_parallel(_apply_chunk, [(func, items[i:i + chunk]) for i in range(0, len(items), chunk)])
    return [y for p in parts for y in p]


def umap(ser, func, as_str=True):
    """Apply func to the unique values of a Series (parallel) and broadcast back."""
    codes, uniq = pd.factorize(ser, sort=False)
    res = pmap(func, uniq.tolist())
    arr = np.empty(len(res), dtype=object)
    arr[:] = res
    out = arr[codes]
    return pd.Series(out, index=ser.index, dtype=STR if as_str else None)


def obj(lst):
    a = np.empty(len(lst), dtype=object)
    a[:] = lst
    return a


def save_json(obj_, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj_, fh, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def load_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)
