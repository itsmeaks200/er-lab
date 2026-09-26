# Prompt for the Copilot agent on the remote PC (paste this as-is)

You are operating the experiment runner in this repo (`er-lab`, the repo root) on this machine. You do NOT design experiments or change the algorithms.
Someone else writes the code, and I relay your results. Your job: set up, run the exact commands below, keep them running,
fix only *environment* problems (missing packages, paths, CUDA), and report outputs verbatim.

Rules:
1. Never edit files inside `erlab/` or `run.py`, except to fix an obvious environment-only issue (like an import path).
   If code fails, do NOT rewrite it. Report the traceback, which is in `results/PASTE_BACK.md`.
2. Never commit or push. Never delete `cache/` unless I ask. It holds hours of computation.
3. Long commands: run them in a persistent terminal, e.g. `nohup python run.py wave 1 > wave1.log 2>&1 &` on Linux or
   `Start-Process` / a separate terminal on Windows. Poll the log only every 15-20 minutes (`tail -n 30 wave1.log`), and do nothing in between, until it prints `DONE.`
4. When a command finishes, print the full content of `results/PASTE_BACK.md` and nothing else. That is what I copy.
5. If memory runs out (MemoryError / killed), rerun the same command with `--set world.n_s1_sample=150000` and say so.

## Step 0: setup (once)
```bash
git pull                     # run everything from the repo root (the folder with run.py)
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import numba, numpy; print('numba', numba.__version__, 'numpy', numpy.__version__)"   # must NOT fail (numba is 10-100x faster)
```
If `torch.cuda.is_available()` is False, install the CUDA build of torch that matches the driver (`nvidia-smi` shows the CUDA version):
`pip install torch --index-url https://download.pytorch.org/whl/cu124` (or cu121/cu126 as appropriate).

Data: the default is `./dataset/{train,test}/` in the repo root (gitignored), so nothing to set if it is there.
Otherwise set the folder that contains `train/` and `test/` with the 7 TSV files:
Linux: `export ER_DATA=/abs/path/to/dataset`   Windows: `$env:ER_DATA="C:\abs\path\to\dataset"`
Optionally put caches on a big disk: `ER_CACHE=/big/disk/er_cache` (needs ~100-150 GB).

## Step 1: environment and smoke test
```bash
python run.py exp ENV
python run.py make-mini --out ./mini_data
python run.py wave 0 1 2 3 --smoke --data ./mini_data --results ./results_smoke --keep-going
```
Report `results/PASTE_BACK.md` (ENV) and `results_smoke/PASTE_BACK.md` (smoke). The smoke scores are meaningless; only OK/FAILED matters.

## Step 2: stage-1 learned blocking on the FULL data (run this first)
```bash
python run.py exp P01 P02 --keep-going > p.log 2>&1        # Windows: Start-Process / separate terminal, same command
```
P01 uses a 300k-S1 world; P02 uses ALL ~2.2M train S1 (needed for its per-pool-record features), so it is the slow and
memory-hungry one. Do NOT shrink `world.n_s1_sample` for P02 if it runs out of memory; report the error instead.
Afterwards print `results/PASTE_BACK.md`.

## Overnight run (one job, ~10-12 h)
```powershell
Start-Process python -ArgumentList 'run.py','overnight' -RedirectStandardOutput overnight.log -RedirectStandardError overnight.err -WindowStyle Hidden
```
Runs the approach pipelines SUB1 → SUB3 → SUB4 → SUB2 → M12 → SUB6 → SUMMARY → P02 → SUB5 → SUMMARY on the full data,
continuing after failures. Each approach: holdout score → refit on all labelled rows → test submission, written to
`results/submissions/<ID>/` (holdout.json, matching_results.tsv, candidate_pairs.tsv, submission_meta.json) and one row in
`results/submissions/SUMMARY.csv`. `results/PASTE_BACK.md` collects everything. Check with `Get-Content overnight.log -Tail 5`.

## Day-2 run (one job, ~12 h; after the overnight run has finished; reuses its caches)
```powershell
git pull
python -m pip install -r requirements.txt          # adds peft, accelerate, bitsandbytes (LLM judges)
python -c "import peft, bitsandbytes; print('peft', peft.__version__, 'bnb', bitsandbytes.__version__)"
Start-Process python -ArgumentList 'run.py','day2' -RedirectStandardOutput day2.log -RedirectStandardError day2.err -WindowStyle Hidden
```
SUB7 → SUB8 → SUB9 → M10F → SUB10 → SUB12 → SUMMARY → SUB11 → SUB13 → SUMMARY: FE2 feature block, Qwen2.5-1.5B LoRA judge,
Qwen2.5-7B 4-bit QLoRA judge (downloads ~15 GB once), stacks of all text models. Same outputs as the overnight run
(`results/submissions/<ID>/`, SUMMARY.csv, PASTE_BACK.md). If bitsandbytes fails to import, SUB11/SUB13 fail and the rest runs.

## Step 3+: the real waves (only when I say which one)
```bash
python run.py wave 1 --keep-going          # blocking      (first run builds caches: ~1-2 h)
python run.py wave 2 --keep-going          # models         (~2-4 h)
python run.py wave 3 --keep-going          # advanced       (~3-6 h)
python run.py exp S01                      # submission     (~1-2 h) → results/submission/*.tsv
```
After each: print `results/PASTE_BACK.md`. For new experiments I will say "git pull, then run: python run.py exp X Y Z".

## Files I may ask you to download
`results/PASTE_BACK.md`, `results/LEADERBOARD.csv`, `results/summary.json`, `results/runs/<EXP>.md`,
and finally `results/submission/matching_results.tsv` + `candidate_pairs.tsv`.
