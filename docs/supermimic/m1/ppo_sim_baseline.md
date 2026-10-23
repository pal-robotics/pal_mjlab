# PPO-Sim baseline, 12 h on the local GPU (2026-10-22/23)

**Setup.**
- Task `Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz`: the paper's env, arms and waist
  at 1 Hz (D27).
- PPO-Sim: 8192 envs, BeyondMimic's PPO hyperparameters, the paper's encoder-decoder actor and 256-128
  critic.
- Library `dev`: 1,296 BONES clips, 2.56 h. Seed 42, commit 21d5962, RTX 5060 Ti.
- 8,800 iterations in 11.9 h (4.87 s per iteration, no preemptions), 1.7 B env steps.

**Benchmark.** `eval.py`, paper F.1: every clip once from frame 0, foot friction 1.0, no randomization
or noise, deterministic policy. Body errors are root-relative, in cm. The return is tracking-only (see
`eval.py`).

| iteration | wall-clock | failed / 1296 | tracking return | legs | arms | torso | arm joint err |
|---|---|---|---|---|---|---|---|
| 1000 | 1.3 h | 169 | 0.620 | 8.5 | 15.2 | 3.4 | 0.45 rad |
| 2000 | 2.7 h | 83 | 0.665 | 7.8 | 13.9 | 3.2 | 0.44 |
| 3000 | 4.0 h | 56 | 0.686 | 7.4 | 13.5 | 2.8 | 0.45 |
| 4000 | 5.4 h | 43 | 0.705 | 7.1 | 12.5 | 2.9 | 0.40 |
| 5000 | 6.8 h | 39 | 0.714 | 6.9 | 12.2 | 2.7 | 0.39 |
| 6000 | 8.1 h | 34 | 0.725 | 6.7 | 11.7 | 2.6 | 0.37 |
| 7000 | 9.5 h | 28 | 0.733 | 6.7 | 11.3 | 2.6 | 0.36 |
| 8000 | 10.8 h | 26 | 0.740 | 6.5 | 11.0 | 2.4 | 0.35 |
| 8800 | 11.9 h | **25** | **0.742** | **6.4** | **10.8** | **2.4** | **0.34** |

- **Still improving, slowly.** Over the last 1,000 iterations the return gained 0.002 and failures
  dropped by 1.
- **Clean control.** Torque saturation stays low (arms 0.3%, waist 2.1%, legs 0.03%), and arm torque
  rate is flat at 0.020.
- **Spread.** Tracking return per clip: p10 0.55, median 0.77, p90 0.90.

**Where it fails** (final checkpoint, 25 clips):
- **Jumps:** 16, mostly vertical (`high_jump`, `reach_jump`).
- **Squat transitions:** 3, crouch-to-stand and stand-to-crouch.
- **The rest:** 6 locomotion and dance clips with fast turns or stops, a folk dance (krakowiak), one
  crate lift with a 360° turn.
- **Lowest tracking by category:** Advanced Locomotion 0.57, Unusual Locomotion 0.58, Dancing 0.59.
- **Highest:** Communication 0.86, Gestures 0.84, Object Interaction 0.83.
- **Arms are still the largest error** (10.8 cm on average, 15–17 cm on dancing, sports and stunts).

**Run-to-run noise.** This run is the D27 1 Hz run's configuration, seed included, retrained.

| | D27 1 Hz run, iteration 999 | this run, iteration 1000 |
|---|---|---|
| failures | 144 | 169 |
| tracking return | 0.619 | 0.620 |
| arm error | 15.3 cm | 15.2 cm |
| leg error | 8.6 cm | 8.5 cm |

- **Means are reproducible; failure counts are not.** GPU simulation isn't bit-deterministic, and the
  failure count varies by about ±15% between identical runs early in training.
- **Effect on the gain A/B.** The A/B's failure difference (162 vs 144) is within that noise. Its
  decision metric, arm error 21 vs 15 cm, is far beyond it.

**What this sets for M4.** SuperMimic has to beat this curve in wall-clock time on the same benchmark:

| target | PPO-Sim reached it at |
|---|---|
| failures < 100 | ~2 h |
| tracking return 0.70 | ~5 h |
| return 0.74 / ≤ 26 failures | ~11 h |

**Not comparable to the paper's PPO numbers** (LAFAN1 0.60/0.70/0.72 on 1/4/8 H100s in 18 h):
- their return includes the regularizers;
- they used G1 and 40 long clips;
- this run had ~10× less compute (12 h on this card ≈ 1.5–2.5 H100-hours).

Reproduce:
- training: `pal_mjlab.scripts.supermimic.train <task> --agent.experiment-name supermimic_ppo_sim_1hz`;
- benchmark: `pal_mjlab.scripts.supermimic.eval` on `model_{1000..8800}.pt`.

Results are in `logs/supermimic_eval/ppo_sim_1hz/` (local, not committed).
