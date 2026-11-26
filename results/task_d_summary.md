# Task D: training on a mixture of Tanner graphs

**Question (course Task D):** can an RL agent trained on a mixture of Tanner graphs
consistently find better schedules than random ordering?

**Short answer:** the mixture-trained agent beats random serial for 2 of 3 training seeds.
The third seed is barely better than random. At equal training budget, the mixture is
**not better than training on [[72,12,6]] alone**, and it is clearly worse than the headline
72-trained agent, which had 5× more training shots. The mixture adds nothing measurable.

## Setup

- **Training:** `experiments/train_rl_mixture.py`. One weight vector, 3000 training shots in
  total. Each shot comes from [[72,12,6]] or [[144,12,12]] with probability 1/2, with
  code-capacity noise at p = 0.05. Everything else is the same as `train_agent.py`:
  - 4 rollouts and the mean reward of the rollouts as the baseline (the mean includes each
    rollout's own reward, so this is not a leave-one-out baseline);
  - reward = success − 0.05 × iterations;
  - learning rate 0.15, at most 20 iterations during training;
  - original 4 features (the residual weight is held at 0).
- **Training seeds** 123, 1 and 2. The code choice, the errors and the policy use three
  streams spawned from `SeedSequence(seed)`, so no training stream equals the evaluation
  error stream `default_rng(1)`, even for seed 1.
- **Evaluation:** `experiments/evaluate_agent.py`, unchanged, with error seed 1 and
  per-shot policy generators `[99, shot]`. The flooding, serial and random-serial rows are
  identical to `night2_144.csv` and `night2_288_iter200.csv`, so all comparisons below are
  on the same shots.
- **Same-budget control:** `results/learned_weights.json`, trained on [[72,12,6]] only, 3000
  shots, seed 123, otherwise identical. It was evaluated on the same shots.

Note that the headline 72-trained agent, `learned_weights_long.json`, was trained on
**15 000** shots, not 3000. Comparing the mixture with it mixes up "which codes" and "how
much training", which is why the control was added.

## Results

± is one binomial standard error. "+a/−b" means the agent rescues a shots that random
serial fails and loses b that random serial decodes. "Paired diff" is the reduction in LER
on identical shots, with its paired standard error.

### [[144,12,12]], p = 0.05, 200 iterations, 20 000 shots

| Agent | LER | Paired diff vs random | +rescued/−lost | Stuck / wrong | Mean iters |
|---|---|---|---|---|---|
| random serial | 0.0458 ± 0.0015 | n/a | n/a | 721 / 195 | 3.12 |
| 72-trained, 15k shots (headline) | 0.0390 ± 0.0014 | −14.8% | +201/−65 | 479 / 301 | 3.35 |
| 72-trained, 3k shots (control) | 0.0413 ± 0.0014 | 0.0044 ± 0.0007 (−9.7%) | +132/−43 | 601 / 226 | 3.42 |
| 144-trained, 30k shots, reward with iteration penalty | 0.0429 ± 0.0014 | −6.3% | +120/−62 | 605 / 253 | 3.14 |
| 144-trained, 30k shots, success-only reward | 0.04145 ± 0.0014 | −9.5% | +131/−44 | 591 / 238 | 3.39 |
| **mixture, seed 123** | 0.0413 ± 0.0014 | 0.0045 ± 0.0008 (−9.8%) | +167/−77 | 536 / 290 | 3.77 |
| **mixture, seed 1** | 0.0418 ± 0.0014 | 0.0040 ± 0.0007 (−8.7%) | +133/−53 | 604 / 232 | 3.35 |
| **mixture, seed 2** | 0.0447 ± 0.0015 | 0.0011 ± 0.0007 (−2.4%) | +96/−74 | 678 / 216 | 4.29 |

### [[288,12,18]], p = 0.06, 200 iterations, 10 000 shots

| Agent | LER | Paired diff vs random | +rescued/−lost | Stuck / wrong | Mean iters |
|---|---|---|---|---|---|
| random serial | 0.0583 ± 0.0023 | n/a | n/a | 579 / 4 | 5.13 |
| 72-trained, 15k shots (headline) | 0.0429 ± 0.0020 | −26.4% | +165/−11 | 412 / 17 | 5.97 |
| 72-trained, 3k shots (control) | 0.0493 ± 0.0022 | 0.0090 ± 0.0010 (−15.4%) | +99/−9 | 481 / 12 | 5.54 |
| 144-trained | not evaluated in this setting | | | | |
| **mixture, seed 123** | 0.0469 ± 0.0021 | 0.0114 ± 0.0012 (−19.6%) | +129/−15 | 456 / 13 | 6.79 |
| **mixture, seed 1** | 0.0484 ± 0.0021 | 0.0099 ± 0.0011 (−17.0%) | +110/−11 | 473 / 11 | 5.54 |
| **mixture, seed 2** | 0.0562 ± 0.0023 | 0.0021 ± 0.0009 (−3.6%) | +49/−28 | 560 / 2 | 6.88 |

The 144-trained agents were trained at p = 0.06. The headline and 144-trained rows come from `night2_144.csv`, `overnight_144_iter200.csv`
and `night2_288_iter200.csv`; they were not re-run. The control and mixture rows are in
`task_d_mixture.csv`.

### Learned weights (unsatisfied, light_on, uncertainty, wobbling)

| Seed | Weights |
|---|---|
| 123 | −2.381, +0.442, −0.629, −0.597 |
| 1 | −0.822, +2.441, −0.697, +0.216 |
| 2 | −0.596, +0.258, +0.043, −0.015 |

## Interpretation

1. **Consistency:** 2 of 3 seeds give clear gains on both codes, 9–10% on [[144,12,12]]
   and 17–20% on [[288,12,18]], 6–9 paired standard errors. Seed 2 gives 2–4%, about
   1.6σ and 2.3σ, which is not a meaningful improvement. With 3 seeds that is not
   "consistently". Seed 2 fits the earlier pattern: every working run has negative weights
   on both `uncertainty` and `unsatisfied`, and seed 2's uncertainty weight is about 0
   (+0.04).
2. **Mixture vs 72 only, at equal budget:**
   - On [[144,12,12]] the best mixture seed equals the control: 0.0413 vs 0.0413.
   - On [[288,12,18]] it is slightly better: 0.0469 vs 0.0493. That difference is about one
     unpaired standard error, so it is not significant; a paired comparison between the two
     agents was not computed.
   - Seed for seed, the mixture is no better than training on [[72,12,6]] alone.
3. **Mixture vs the headline agent:** the headline 72-trained agent, with 15 000 shots, is
   better than every mixture run: 0.0390 vs at best 0.0413 on 144, and 0.0429 vs at best
   0.0469 on 288. Training length matters more than the mix of codes here. A 15 000-shot
   mixture run was not done.
4. **Your hypothesis is supported by the training logs:**

   | Seed | [[72]] shots with mixed rollout outcomes | [[144]] shots with mixed rollout outcomes |
   |---|---|---|
   | 123 | 149 of 1504 | 21 of 1496 |
   | 1 | 120 of 1521 | 26 of 1479 |
   | 2 | 125 of 1535 | 38 of 1465 |

   A mixed outcome means at least one of the 4 rollouts succeeds and at least one fails.
   Shots from [[144,12,12]] carry 4–7× fewer of these. So most of the success signal comes
   from the [[72,12,6]] shots, and the [[144,12,12]] half mostly adds iteration-count
   signal. This fits the result that the mixture behaves like 72-only training with about
   half the useful shots. It is consistent with the hypothesis, not a proof of it.
5. **Same trade-off as before:** like the 72-trained agent, the mixture agents trade stuck
   shots for wrong answers. On [[144,12,12]], seed 123 has 721 → 536 stuck and 195 → 290
   wrong. They also need more iterations than random serial: 3.4–4.3 vs 3.1 on 144, and
   5.5–6.9 vs 5.1 on 288.

## Commands

```
python -m experiments.train_rl_mixture --seed 123 > results/log_train_mixture_seed123.txt
python -m experiments.train_rl_mixture --seed 1   > results/log_train_mixture_seed1.txt
python -m experiments.train_rl_mixture --seed 2   > results/log_train_mixture_seed2.txt
python -m experiments.evaluate_agent --code 144 --error-rates 0.05 --shots 20000 --max-iterations 200 --weights results/learned_weights_mixture_seed123.json --output <file>
python -m experiments.evaluate_agent --code 288 --error-rates 0.06 --shots 10000 --max-iterations 200 --weights results/learned_weights_mixture_seed123.json --output <file>
```

There was one `evaluate_agent` run per agent and setting: the three mixture seeds and
`results/learned_weights.json`. The runs went in parallel and were merged into
`task_d_mixture.csv`, keeping one copy of the identical baseline rows. Training took about
4.5 min per seed; the eight evaluations took 70 min of wall time in parallel.
