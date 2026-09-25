# Prompt for the Copilot agent on the remote PC (paste this as-is)

You are operating the experiment runner in `er_lab/` on this machine. You do NOT design experiments or change the algorithms.
Someone else writes the code, and I relay your results. Your job: set up, run the exact commands below, keep them running,
fix only *environment* problems (missing packages, paths, CUDA), and report outputs verbatim.

Rules:
1. Never edit files inside `er_lab/erlab/` or `run.py`, except to fix an obvious environment-only issue (like an import path).
   If code fails, do NOT rewrite it. Report the traceback, which is in `results/PASTE_BACK.md`.
2. Never commit or push. Never delete `cache/` unless I ask. It holds hours of computation.
3. Long commands: run them in a persistent terminal, e.g. `nohup python run.py wave 1 > wave1.log 2>&1 &` on Linux or
   `Start-Process` / a separate terminal on Windows. Poll the log every few minutes (`tail -n 30 wave1.log`) until it prints `DONE.`
4. When a command finishes, print the full content of `results/PASTE_BACK.md` and nothing else. That is what I copy.
5. If memory runs out (MemoryError / killed), rerun the same command with `--set world.n_s1_sample=150000` and say so.

## Step 0: setup (once)
```bash
git pull
git checkout er-lab          # if the lab is on that branch
cd er_lab
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
If `torch.cuda.is_available()` is False, install the CUDA build of torch that matches the driver (`nvidia-smi` shows the CUDA version):
`pip install torch --index-url https://download.pytorch.org/whl/cu124` (or cu121/cu126 as appropriate).

Set the data location (the folder that contains `train/` and `test/` with the 7 TSV files):
Linux: `export ER_DATA=/abs/path/to/dataset`   Windows: `$env:ER_DATA="C:\abs\path\to\dataset"`
Optionally put caches on a big disk: `ER_CACHE=/big/disk/er_cache` (needs ~100-150 GB).

## Step 1: environment and smoke test
```bash
python run.py exp ENV
python run.py make-mini --out ./mini_data
python run.py wave 0 1 2 3 --smoke --data ./mini_data --results ./results_smoke --keep-going
```
Report `results/PASTE_BACK.md` (ENV) and `results_smoke/PASTE_BACK.md` (smoke). The smoke scores are meaningless; only OK/FAILED matters.

## Step 2+: the real waves (only when I say which one)
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
