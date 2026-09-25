#!/usr/bin/env python
"""ER-lab command line.

  python run.py list                              # show experiments
  python run.py make-mini --out ./mini_data       # small faithful dataset for smoke tests
  python run.py exp ENV                           # environment check
  python run.py exp SMOKE --data ./mini_data      # end-to-end in minutes
  python run.py wave 1                            # run every experiment of a wave (in order)
  python run.py overnight                         # full-data plan: stage-1 frontier, model zoo, HPO, cross-encoder, submission
  python run.py exp M01 M02 --set model.rounds=3000 world.n_s1_sample=400000
Afterwards: cat results/PASTE_BACK.md  → paste into the chat.
Global options: --data DIR (or env ER_DATA), --cache DIR (ER_CACHE), --results DIR (ER_RESULTS), --jobs N (ER_JOBS)
"""
import argparse
import os
import sys


def main():
    for stream in (sys.stdout, sys.stderr):      # Windows consoles default to cp1252; never crash on '→' etc.
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['list', 'exp', 'wave', 'make-mini', 'overnight'])
    ap.add_argument('ids', nargs='*')
    ap.add_argument('--data'); ap.add_argument('--cache'); ap.add_argument('--results'); ap.add_argument('--jobs', type=int)
    ap.add_argument('--out', default='./mini_data')
    ap.add_argument('--set', nargs='*', default=[], help='config overrides, e.g. model.rounds=3000')
    ap.add_argument('--keep-going', action='store_true', help='continue the wave after a failed experiment')
    ap.add_argument('--smoke', action='store_true', help='tiny config (use with the mini dataset) to test code paths fast')
    a = ap.parse_args()
    if a.data: os.environ['ER_DATA'] = a.data
    if a.cache: os.environ['ER_CACHE'] = a.cache
    if a.results: os.environ['ER_RESULTS'] = a.results
    if a.jobs: os.environ['ER_JOBS'] = str(a.jobs)

    from erlab.registry import EXPERIMENTS, SMOKE_CFG, OVERNIGHT
    if a.cmd == 'list':
        for k, v in EXPERIMENTS.items():
            print(f"wave {v['wave']}  {k:6s} [{v['kind']:8s}] {v['desc']}")
        return 0
    if a.cmd == 'make-mini':
        from erlab.data import make_mini_dataset
        make_mini_dataset(os.environ.get('ER_DATA', './dataset'), a.out)
        return 0
    from erlab.experiments import run_experiment
    if a.cmd == 'overnight':
        ids, a.set, a.keep_going = OVERNIGHT['ids'], OVERNIGHT['sets'] + a.set, True
        print(f"overnight plan: {' '.join(ids)} | settings: {' '.join(a.set)}", flush=True)
    elif a.cmd == 'exp':
        ids = a.ids
    else:
        waves = {int(x) for x in a.ids}
        ids = [k for k, v in EXPERIMENTS.items() if v['wave'] in waves and k != 'SMOKE']
    unknown = [i for i in ids if i not in EXPERIMENTS]
    if unknown:
        print('unknown experiments:', unknown); return 2
    ok_all = True
    for i in ids:
        print(f'\n==================== {i}: {EXPERIMENTS[i]["desc"]}', flush=True)
        ok = run_experiment(i, EXPERIMENTS[i], a.set, SMOKE_CFG if a.smoke else None)
        ok_all &= ok
        if not ok and not a.keep_going and a.cmd in ('wave', 'overnight'):
            print('stopping wave after failure (use --keep-going to continue)')
            break
    res = os.environ.get('ER_RESULTS', './results')
    print(f'\nDONE. Paste back: {os.path.join(res, "PASTE_BACK.md")}')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
