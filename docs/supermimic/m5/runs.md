# M5: the 24 h `full` and `dev` runs (status 2026-10-27 07:45)

**What runs.** 24 h jobs on Leonardo, one A100 each, with the same config:
- the paper's settings (TF32, replay 1000);
- the stability fixes as default (D39): a failure head on the rule's two errors and a survival-weighted
  critic loss;
- compiled imagination (D37);
- the reviewed dataset (D40).

| run | job | data | ran | result |
|---|---|---|---|---|
| `dev-24h` | 62300428 (SMdev), `e6f3893` | `dev`: 1,294 clips, 2.55 h | 2026-10-26 01:54 → 10-27 01:58 | finished: 71,910 iterations |
| `full-24h` | 62300423 (SMfull), `e6f3893` | `full`: 124,077 clips, 250.9 h | 2026-10-26 01:54 → 19:18 | went NaN at ~44,620; cancelled (below) |
| `full-24h`, resumed | 62362613 (SMfull2), `bc44e76` (njmax 1024, D44) | `full` | 2026-10-26 19:29 → ~10-27 19:30 | running: from `model_44000` |

- Run folders: `lazyhpc/runs/20261026-015332_{SMfull,SMdev}_e6f3893`, `lazyhpc/runs/20261026-192504_SMfull2_bc44e76`.
- Logs: `lazyhpc/logs/rsl_rl/supermimic_d39/<time>_{full,dev}-24h` (the resumed run writes a new `<time>_full-24h`).

## Benchmark on `dev`

Every checkpoint at a multiple of 2000, on the reviewed 1,294-clip `dev`, every clip once from its first
frame (paper F.1); `full`'s first job up to 44,000 (its clean checkpoints). Results:
`logs/supermimic_eval/m5_{dev,full}-24h/`; figure: `make_figures.py`.

![benchmark on dev](figures/benchmark_dev.png)

| | failed clips | tracking return | body error arms / legs / torso |
|---|---|---|---|
| `dev-24h`, 22,000 (7.4 h) | 0 | 0.887 | 5.9 / 4.0 / 1.7 cm |
| `dev-24h`, 46,000 (15.5 h) | 0 | 0.897 | 5.6 / 4.0 / 1.5 cm |
| `dev-24h`, **70,000** (23.6 h, best) | **0** | **0.906** | **5.4 / 3.6 / 1.4 cm** |
| `full-24h`, 38,000 (best of the first job) | 0 | 0.881 | 5.9 / 4.1 / 1.5 cm |
| `full-24h`, 44,000 (last before the NaN) | 1 | 0.869 | 6.1 / 4.2 / 1.6 cm |
| PPO-Sim, 24 h (cluster, M4) | 15 | 0.770 | 10.7 arms |
| old SuperMimic baseline, best of 24 h (M4) | 2 | 0.872 | 6.2 arms |
| stability runs, best (A2′, 34,000, M4) | 2 | 0.894 | |

The M4 rows were benchmarked on the 1,296-clip `dev`, which also holds the two clips removed in D40; every
policy failed those, so their failure counts are about 2 higher than they would be here.

- **`dev` trained without a collapse and kept improving for the whole 24 h.** It fails no clip at 22,000
  and at every checkpoint from 46,000 on but one (60,000: 1 clip). Its tracking return rises from 0.887 at
  22,000 to 0.906 at 70,000, above everything before: PPO-Sim's 24 h (0.770, 15 failures), the old
  baseline's best (0.872) and the stability runs' best (0.894).
- **`full` tracks `dev` closely up to about 16,000 iterations, then levels off at 0.87–0.88** with 0 to 6
  failures on `dev`. `dev` is 1% of `full`'s motions, and from about 28,000 `full`'s sampler sends half of
  its starts to `full`'s hardest bins (most of them not in `dev`). Its own test is a sample of `full`, after
  the resumed job ends.
- **Caveats:** one seed per run. The benchmark runs with `njmax` 1024 (D44), while these checkpoints trained
  with 300, so their contacts are now simulated more exactly than in training. Each checkpoint took 5.2 min
  on the local GPU, about 4× the M4 benchmarks; the larger constraint buffer is the likely cause (not yet
  measured). The odd thousands are being benchmarked now.

## Training signals at 6.8 h (training logs only)

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

## `full` collapsed at 17.5 h; resumed from 44,000 with a larger constraint buffer (D44)

- **What happened:** at iteration ~44,620 (2026-10-26 ~17:55) the world model, critic and actor went NaN
  within 10 iterations: the logged losses disappear, episodes end after 11 steps, 96% of them fail. The
  checkpoints from 45,000 on are NaN (moved to the run's `nan_quarantine/`); 44,000 is clean.
- **Likely cause:** the simulator's constraint buffer. MuJoCo Warp warned "nefc overflow" about 250 times
  per 100 iterations all run long (up to 636 rows requested, 300 allowed); dropped rows give wrong contact
  physics, and one non-finite state in the replay poisons every network. `dev` warned 263 times in total.
- **Actions (user):** the job was cancelled at 17.4 h (it only produced NaN), `njmax` raised to 1024
  (D44), and `full` resumed from `model_44000` for another 24 h as `SMfull2`.
- **Detection:** the logger skips NaN values, so the losses simply stopped appearing; the collapse showed
  as a jump of training failure to 96%.
- **The resumed run (SMfull2)** logged "Resuming from ... model_44000.pt" and has had no overflow warning
  since. At 79,690 iterations (2026-10-27 06:32, 11 h in) it was healthy: no NaN, critic burst or failure
  spike; 1.14 s per iteration; training failure about 37% and return 9.7. Those two reflect the sampler,
  which since about 28,000 starts half of the episodes on `full`'s hardest bins, not a regression (bin
  returns kept rising).

## The CUDA-graph attempt (D41)

The first pair of these jobs (commit `8f68e36`) used CUDA graphs on the compiled step. An isolated
benchmark on the A100 had shown them 2.8× faster on the policy update and equal to eager. In the
training loop they logged no iteration in 2 h: 100% CPU, idle GPU, and the warning "pending, uninvoked
backwards". They were cancelled at 01:51 and resubmitted without CUDA graphs (`e6f3893`, the same
training code as `31295b6`). About 2 h of both jobs were lost. Making CUDA graphs work in the loop is
open (D41).

## Next

- **`full`:** when the resumed job ends (~2026-10-27 19:30), benchmark its checkpoints on `dev` and on a
  stratified sample of `full` (all 124,077 clips is about 96 times a `dev` benchmark).
- **`dev`:** the odd-thousand checkpoints are being benchmarked; the figure keeps the multiples of 2000.
- **Options:** an upper-body goal interface trained on a frozen checkpoint (paper Appendix B; not started).
