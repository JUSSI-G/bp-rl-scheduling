# Reproducing the results

Every command runs from the repository root with the virtual environment active
(Python 3.14.3, packages from `requirements.txt`). Scripts are run as modules:

```
python -m experiments.<script> [options]
```

All defaults are the settings that produced the files in `results/`. Runtimes are
approximate, single core, taken from the `seconds` columns and the training logs in
`results/`. The `log_*.txt` files are the console output of the runs, redirected from
PowerShell (UTF-16 encoded).

Every script accepts `--shots` (or `--episodes`), so any of them can be smoke-tested in
seconds, for example `python -m experiments.run_task_a --shots 100 --output smoke.csv`.

## 0. Sanity checks

| Command | Output | Time |
|---|---|---|
| `python -m pytest` | 10 tests | 2 s |
| `python -m experiments.ldpc_baseline` | `figures/ldpc_baseline_72.png` | 2 s |
| `python -m experiments.compare_bp` | console only: own flooding BP vs ldpc | 15 s |

## 1. Task A: flooding vs serial vs random serial

| Command | Output | Time |
|---|---|---|
| `python -m experiments.run_task_a` | `results/task_a.csv` | 5 min |
| `python -m experiments.plot_task_a` | `figures/task_a.png` | 2 s |

## 2. Task B: does the serial order matter?

| Command | Output | Time |
|---|---|---|
| `python -m experiments.run_task_b` | `results/task_b_72_12_6.npz` | 1 min |
| `python -m experiments.plot_task_b` | `figures/task_b_72_12_6.png` | 2 s |

## 3. Task C: learned schedule on [[72,12,6]]

The headline agent. `--policy-seed 0 --no-residual` reproduces the original training run
(4 features) bit for bit.

| Command | Output | Time |
|---|---|---|
| `python -m experiments.train_agent --code 72 --shots 15000 --seed 123 --policy-seed 0 --no-residual --output results/learned_weights_long.json` | `results/learned_weights_long.json` | 10 min |
| `python -m experiments.evaluate_task_c` | `results/task_c_long.csv` | 4 min |
| `python -m experiments.train_rls` | `results/rls_q_table.npy`, `results/rls_settings.json` | 3 min |
| `python -m experiments.evaluate_rls` | `results/task_c_rls.csv` | 4 min |
| `python -m experiments.plot_task_c` | `figures/task_c.png` | 2 s |

The JSON written by `train_agent` today also contains a zero `residual` weight and the keys
`no_residual` and `policy_seed`; the four original weights are identical.

## 4. Overnight run 1: transfer and reward ablations

Training (night-1 weight files have four features, hence `--no-residual`):

| Command | Output | Time |
|---|---|---|
| `python -m experiments.train_agent --code 72 --shots 15000 --iteration-penalty 0.0 --no-residual --output results/weights_72_success_only.json > results/log_train_72_success_only.txt` | `weights_72_success_only.json` | 6 min |
| `python -m experiments.train_agent --code 144 --error-rate 0.06 --shots 30000 --no-residual --output results/weights_144_penalty.json > results/log_train_144_penalty.txt` | `weights_144_penalty.json` | 41 min |
| `python -m experiments.train_agent --code 144 --error-rate 0.06 --shots 30000 --iteration-penalty 0.0 --no-residual --output results/weights_144_success_only.json > results/log_train_144_success_only.txt` | `weights_144_success_only.json` | 43 min |

Evaluation:

| Command | Output | Time |
|---|---|---|
| `python -m experiments.evaluate_agent --code 72 --error-rates 0.05 --shots 10000 --weights results/learned_weights_long.json results/weights_72_success_only.json results/weights_144_penalty.json --output results/overnight_72.csv > results/log_eval_72.txt` | `overnight_72.csv` | 4 min |
| `python -m experiments.evaluate_agent --code 144 --error-rates 0.04 0.05 0.06 --shots 20000 --weights results/learned_weights_long.json results/weights_144_penalty.json results/weights_144_success_only.json results/weights_72_success_only.json --output results/overnight_144_iter50.csv > results/log_eval_144_iter50.txt` | `overnight_144_iter50.csv` | 2.6 h |
| `python -m experiments.evaluate_agent --code 144 --error-rates 0.05 --shots 20000 --max-iterations 200 --weights results/learned_weights_long.json results/weights_144_penalty.json results/weights_144_success_only.json results/weights_72_success_only.json --output results/overnight_144_iter200.csv > results/log_eval_144_iter200.txt` | `overnight_144_iter200.csv` | 47 min |

## 5. Overnight run 2: training seeds, residual feature, [[288,12,18]]

Training (all on [[72,12,6]], p = 0.05):

| Command | Output | Time |
|---|---|---|
| `python -m experiments.train_agent --code 72 --shots 15000 --seed 1 --no-residual --output results/weights_72_original_s1.json > results/log_train_72_original_s1.txt` | `weights_72_original_s1.json` | 8 min |
| same with `--seed 2` / `--seed 3` (`_s2`, `_s3`) | `weights_72_original_s2.json`, `_s3.json` | 8 min each |
| `python -m experiments.train_agent --code 72 --shots 60000 --no-residual --output results/weights_72_original_60k.json > results/log_train_72_original_60k.txt` | `weights_72_original_60k.json` | 31 min |
| `python -m experiments.train_agent --code 72 --shots 15000 --output results/weights_72_residual_s123.json > results/log_train_72_residual_s123.txt` | `weights_72_residual_s123.json` | 9 min |
| same with `--seed 1` / `--seed 2` (`_s1`, `_s2`) | `weights_72_residual_s1.json`, `_s2.json` | 8 min each |

`results/rule_residual_first.json` (`{"weights": [0, 0, 0, 0, 5]}`) was written by hand.

Evaluation:

| Command | Output | Time |
|---|---|---|
| `python -m experiments.evaluate_agent --code 144 --error-rates 0.05 --shots 20000 --max-iterations 200 --weights results/learned_weights_long.json results/weights_72_original_s1.json results/weights_72_original_s2.json results/weights_72_original_s3.json results/weights_72_original_60k.json results/rule_residual_first.json results/weights_72_residual_s123.json results/weights_72_residual_s1.json results/weights_72_residual_s2.json --output results/night2_144.csv > results/log_night2_144.txt` | `night2_144.csv` | 2.7 h |
| `python -m experiments.evaluate_agent --code 288 --error-rates 0.05 0.06 --shots 10000 --weights results/learned_weights_long.json results/weights_72_residual_s123.json --output results/night2_288.csv > results/log_night2_288.txt` | `night2_288.csv` | 73 min |
| `python -m experiments.evaluate_agent --code 288 --error-rates 0.06 --shots 10000 --max-iterations 200 --weights results/learned_weights_long.json --output results/night2_288_iter200.csv > results/log_night2_288_iter200.txt` | `night2_288_iter200.csv` | 46 min |

Figures:

| Command | Output | Time |
|---|---|---|
| `python -m experiments.plot_transfer` | `figures/transfer.png` | 2 s |
| `python -m experiments.plot_robustness` | `figures/robustness.png` | 2 s |

`evaluate_agent` stores each agent under its weight-file path in the `schedule` column, and
the plot scripts look rows up by that exact string. Pass the paths exactly as above
(`results/<file>.json`, forward slashes).

## 6. Task D: mixture of Tanner graphs

Training (3 000 shots, half [[72,12,6]] and half [[144,12,12]], p = 0.05). The code choice,
errors and policy use streams spawned from `SeedSequence(seed)`, so no training stream
equals the evaluation error stream `default_rng(1)`.

| Command | Output | Time |
|---|---|---|
| `python -m experiments.train_rl_mixture --seed 123 > results/log_train_mixture_seed123.txt` | `learned_weights_mixture_seed123.json` | 3–4.5 min |
| same with `--seed 1` / `--seed 2` | `learned_weights_mixture_seed1.json`, `_seed2.json` | 3–4.5 min each |

Evaluation. The control is the existing 3 000-shot 72-only agent `learned_weights.json`:

| Command | Output | Time |
|---|---|---|
| `python -m experiments.evaluate_agent --code 144 --error-rates 0.05 --shots 20000 --max-iterations 200 --weights results/learned_weights_mixture_seed123.json results/learned_weights_mixture_seed1.json results/learned_weights_mixture_seed2.json results/learned_weights.json --output task_d_144.csv` | `task_d_144.csv` | about 75 min |
| `python -m experiments.evaluate_agent --code 288 --error-rates 0.06 --shots 10000 --max-iterations 200 --weights results/learned_weights_mixture_seed123.json results/learned_weights_mixture_seed1.json results/learned_weights_mixture_seed2.json results/learned_weights.json --output task_d_288.csv` | `task_d_288.csv` | about 2.4 h |
| `{ cat task_d_144.csv; tail -n +2 task_d_288.csv; } > results/task_d_mixture.csv` | `results/task_d_mixture.csv` | |
| `python -m experiments.plot_task_d` | `figures/task_d_mixture.png`, `figures/task_d_mixture_dark.png` | 3 s |

For the stored file, each agent was instead run as its own `evaluate_agent` process, all 8
in parallel (70 min of wall time), and the identical baseline rows were kept once. Every
agent row uses per-shot generators, so it does not depend on which other agents share the
run: the two commands above give the same rows apart from `seconds`. The flooding, serial
and random-serial rows are identical to those in `night2_144.csv` and
`night2_288_iter200.csv`. The analysis is in `results/task_d_summary.md`.

## Superseded runs (kept for completeness, not used in figures)

| File | Command | Superseded by |
|---|---|---|
| `results/learned_weights.json` | `python -m experiments.train_agent --code 72 --shots 3000 --seed 123 --policy-seed 0 --no-residual --output results/learned_weights.json` | `learned_weights_long.json` (15 000 shots) |
| `results/task_c.csv` | `python -m experiments.evaluate_task_c --shots 2000 --weights results/learned_weights.json --output results/task_c.csv` | `task_c_long.csv` (10 000 shots) |
| `results/task_e_transfer_144.csv` | `python -m experiments.evaluate_transfer` | `overnight_144_iter50.csv` (20 000 shots, more error rates) |

The first two were re-checked with the current code and reproduce exactly (apart from the
`seconds` column).

## Random-number conventions

All randomness comes from `numpy.random.default_rng`, so results are exactly repeatable,
but the scripts do not all use the same convention:

- **Errors:** a fresh generator with seed 1 for every error rate (`--seed`), so different
  schedules always see the same errors. Exception: `ldpc_baseline` uses one generator for
  all error rates in sequence.
- **Random serial order and agent tie-breaking:**
  - `evaluate_task_c`, `evaluate_rls`, `evaluate_transfer`, `evaluate_agent`: one
    generator per shot, seeded `[99, shot]` (`src.sim.decode_all_shots`). Results for a shot
    do not depend on which other shots or schedules were run.
  - `run_task_b`: one generator per shot, seeded `[seed, shot]` for seeds 0 to 19.
  - `run_task_a`, `compare_bp`: one generator (seed 7) shared by all shots in sequence.
    That is why the random-serial numbers in `task_a.csv` differ slightly from those in
    Task C at the same settings.
- **Training:** `train_agent` uses `--seed` for the training errors and `--policy-seed`
  (default `seed + 1`) for the exploration policy; `train_rls` uses a single generator.
