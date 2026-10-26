# M5: the 24 h `full` and `dev` runs (status 2026-10-26 08:50)

**What runs.** Two 24 h jobs on Leonardo, one A100 each, from the same commit (`e6f3893`) and config:
- the paper's settings (TF32, replay 1000);
- the stability fixes as default (D39): a failure head on the rule's two errors and a survival-weighted
  critic loss;
- compiled imagination (D37);
- the reviewed dataset (D40).

| run | job | data | started | ends |
|---|---|---|---|---|
| `full-24h` | 62300423 (SMfull) | `full`: 124,077 clips, 250.9 h | 2026-10-26 01:54 | ~2026-10-27 01:54 |
| `dev-24h` | 62300428 (SMdev) | `dev`: 1,294 clips, 2.55 h | 2026-10-26 01:54 | ~2026-10-27 01:54 |

- Run folders: `lazyhpc/runs/20261026-015332_{SMfull,SMdev}_e6f3893`.
- Logs: `lazyhpc/logs/rsl_rl/supermimic_d39/<time>_{full,dev}-24h`.

## Status at 6.8 h (training logs only; the benchmark comes after the runs)

| | `full` | `dev` |
|---|---|---|
| iteration | 19,710 | 20,780 |
| speed | 1.16 s/it (2,889 it/h), 33.4 GB | 1.15 s/it (3,042 it/h), 19.6 GB |
| expected at 24 h | about 69,000 | about 73,000 |
| early critic blow-up | none (lowest value 0.1, take-off at iteration 270) | none (0.1, iteration 240) |
| failure spikes, critic-loss bursts | 0, 0 | 0, 0 |
| world-model loss (Eq. 1) | 0.092 → 0.014 | 0.091 → 0.011 |
| critic loss (Eq. 3) | about 0.033, flat | 0.053 → 0.0067 |
| imagined reward per step (max 0.1) | 0.085 | 0.073 |
| training episodes ending in failure | 0.9% | 20% |
| imagined vs simulated 32-step return | correlation ≥ 0.997 | ≥ 0.996 |
| sampler: mean normalized return of all bins | 0.590, with 78% of bins visited (≈ 0.75 over visited bins) | 0.824, all bins visited |

- **Both are stable so far:** smooth losses and no spikes. Neither went through the early blow-up that
  destabilized the old baseline (M4 stability study).
- **The failure rates of the two runs are not comparable.** On `dev` the sampler concentrates starts on
  the hardest clips; on `full` it spreads them over about 900,000 bins, a quarter of which are still
  unvisited. Their quality will be compared on the benchmark.

## `full` collapsed at 17.5 h; resumed from 44,000 with a larger constraint buffer (D42)

- **What happened:** at iteration ~44,620 (2026-10-26 ~17:55) the world model, critic and actor went NaN
  within 10 iterations: the logged losses disappear, episodes end after 11 steps, 96% of them fail. The
  checkpoints from 45,000 on are NaN (moved to the run's `nan_quarantine/`); 44,000 is clean.
- **Likely cause:** the simulator's constraint buffer. MuJoCo Warp warned "nefc overflow" about 250 times
  per 100 iterations all run long (up to 636 rows requested, 300 allowed); dropped rows give wrong contact
  physics, and one non-finite state in the replay poisons every network. `dev` warned 263 times in total.
- **Actions (user):** the job was cancelled at 17.4 h (it only produced NaN), `njmax` raised to 1024
  (D42), and `full` resumed from `model_44000` for another 24 h as `SMfull2`.
- **Detection:** the logger skips NaN values, so the losses simply stopped appearing; the collapse showed
  as a jump of training failure to 96%.

## The CUDA-graph attempt (D41)

The first pair of these jobs (commit `8f68e36`) used CUDA graphs on the compiled step. An isolated
benchmark on the A100 had shown them 2.8× faster on the policy update and equal to eager. In the
training loop they logged no iteration in 2 h: 100% CPU, idle GPU, and the warning "pending, uninvoked
backwards". They were cancelled at 01:51 and resubmitted without CUDA graphs (`e6f3893`, the same
training code as `31295b6`). About 2 h of both jobs were lost. Making CUDA graphs work in the loop is
open (D41).

## Next

- **After the runs (~2026-10-27 02:00), benchmark the checkpoints:**
  - `dev`: on the 1,294-clip `dev`;
  - `full`: on `dev` too, for comparison, plus a stratified sample of `full`. All 124,077 clips is about
    96 times the work of a `dev` benchmark, so it needs its own plan.
- **Options:**
  - extend either run by resubmitting its `job.sh` (it resumes from the newest checkpoint);
  - an upper-body goal interface trained on a frozen checkpoint (paper Appendix B; not started).
