# 5-DoF: tight tracking and the action scale (2026-11-04)

> **Superseded in part (D57).** The defaults went back to the paper's reward on the user's decision. This report is
> the record of those runs. The paper reward with the 3.5 Hz gains and 1 Hz action scales was not run here, so B2's
> numbers do not describe it.

**Question.** On the real robot the 5-DoF policy slides its feet and does not stand perfectly still. Does tighter
tracking help? And do the arms and waist need the 1 Hz gains, or only the action scales those gains gave them?

**Answer.**
- Tighter tracking helps: half the leg error, −17 % arm error, −29 % root drift while standing, 0 failures (B vs A).
- The 1 Hz gains are not needed: the robot's own 3.5 Hz gains with the 1 Hz action scales track as well or better
  (B2 vs B). The cost is about 5× the arm torque rate, still to be checked on the robot.
- Foot sliding in simulation does not change in any run: about 2.2–2.5 cm/s, 7 % of planted foot-steps above 5 cm/s.
- The defaults later went back to the paper's reward and kept B2's gains and action scales (D55, D57).

**Not the data.** Where the G1 source has a foot planted, the 5-DoF reference foot moves 0.6 cm/s (G1 0.3), drifts
4 mm per planted stretch (G1 1.7), and is > 1 cm below the floor in 1.1 % of those frames
(`kangaroo_bones_seed/data/variants/feet`). No retarget was needed.

## Runs

| run | reward | arm and waist gains | action scales | iterations | logs |
|---|---|---|---|---:|---|
| A | paper | 1 Hz | 1 Hz | 146,920 | `2026-10-3{0,1}_*_hands-full-v2-24h` |
| B | tight | 1 Hz | 1 Hz | 150,550 | `2026-10-3{0,1}_*_hands-tight-full-24h` |
| B2 | tight | 3.5 Hz (the robot's) | 1 Hz | 154,850 | `2026-11-0{2,3}_*_hands-tightscale-full-24h` |

Each ran 48 h on one Leonardo A100 with the same data (5-DoF set, D51, no flips); the legs keep their gains. No run
has a NaN. A's world-model loss stays below 1; B has one blip, 2.28 at 87,040, gone on the next log line. Logs are in
`$FAST/omartin1/lazyhpc/logs/rsl_rl/supermimic/`, the exact code in `$FAST/omartin1/lazyhpc/runs/` (`*_SMhands*_64e0f03`,
`*_SMhandsTight*_67ce3f1`, `*_912a8ed`).

- **Tight reward:** the six tracking stds are halved, and `motion_feet_pos` (std 0.05 m) and `motion_feet_lin_vel`
  (std 0.1 m/s) track `leg_*_5_link` on their own, weight 1 each. In the paper's reward a sliding foot barely counts,
  because its error is averaged with the other 13 bodies'.
- **Action scale:** α = 0.25·effort/Kp sets the exploration noise and the action-rate cost per radian. B2 has every
  gain at 3.5 Hz and the 1 Hz runs' α (rad per unit: arm 1–2 and waist 0.66, arm 3–5 0.52, Seed wrist 0.42). It tells
  whether the 1 Hz gains (D27) helped through the servo itself or through α.

## Benchmark

`scripts/supermimic/eval.py` on the 1,321-clip dev library. Each run is scored in its own training task, because a
policy only runs right with the gains it was trained with: B2 in a 1 Hz config has 12× softer arm servos and reads
10 cm of arm error, which is not valid. Tracking return depends on the reward, so it only compares within one reward.
Feet are `leg_*_5_link`; "planted" means the reference foot is below 0.15 m and slower than 2 cm/s.

| at 146k | A | B | B2 |
|---|---:|---:|---:|
| failures | 0 | 0 | 0 |
| tracking return, paper's reward | 0.917 | 0.921 | |
| tracking return, tight reward | | 0.785 | 0.811 |
| leg / arm / torso body error, cm | 3.5 / 3.5 / 1.3 | 1.8 / 2.9 / 1.2 | 1.8 / 2.9 / 1.2 |
| arm joint error, rad | 0.082 | 0.070 | 0.059 |
| foot slide, cm/s: robot foot's horizontal speed while planted | 2.21 | 2.46 | 2.45 |
| foot slip: share of those frames above 5 cm/s | 7.1 % | 7.0 % | 6.8 % |
| root drift, cm/s: horizontal speed while the reference root stands still | 3.68 | 2.63 | 2.67 |
| arm torque rate, Δτ/τmax per step | 0.015 | 0.021 | 0.099 |
| arm torque saturation | 0.03 % | 0.03 % | 0.75 % |
| waist torque rate | 0.026 | 0.034 | 0.112 |
| waist torque saturation | 0.50 % | 0.60 % | 2.48 % |

- **Clip by clip:** B tracks better than A on 69 % of clips and its root drifts less on 77 %. B2 tracks better than
  B on 89 %.
- **Standing:** on the 110 clips with "idle" or "stand" in the name, root drift goes 3.02 → 2.37 → 2.17 cm/s (A, B,
  B2) and foot slide 1.75 → 1.61 → 1.46 cm/s.
- **Over training** (`results.csv`, every 20k): every run has 0 failures from 100k on. B scores the same as A or
  higher on the paper's reward at every checkpoint. B2 scores 2–4 % above B at every checkpoint and reaches 0.815 at
  154k. Foot slide stays around 2–2.8 cm/s in A and B; B2 starts higher (3.3 at 20k) and joins them from 140k.
- **B2's training curves look worse than the benchmark:** its episodes were ~20 % shorter and its training return
  ~5 % lower than B's, without worse tracking.

## Next

1. **Check B2's `model_146000.onnx` on the robot for jitter:**
   `$FAST/omartin1/lazyhpc/logs/rsl_rl/supermimic/2026-11-03_08-53-43_hands-tightscale-full-24h/`. If jitter shows, the
   fix is a torque- or action-rate penalty, not softer servos. A's and B's policies need the 1 Hz gains.
2. **Measure the sliding properly.** `foot_slide` is the ankle frame's horizontal speed, so it includes foot roll.
   Measure sole-point slip in simulation first, then compare the simulator's contact and friction with the real floor.

## Evidence

`results.csv` has one row per run, checkpoint and scoring task (`scored_in`: `paper`, or `own` for the run's training
task). The per-clip JSONs (21 MB) and the benchmark scripts are on helium, in `logs/supermimic_eval/tight_comparison/`
of the main checkout. To score a checkpoint with today's defaults (paper reward, B2's gains and action scales):

```bash
uv run -m pal_mjlab.scripts.supermimic.eval --task Mjlab-SuperMimic-Flat-Pal-Kangaroo-Hands \
  --checkpoints <run>/model_146000.pt --library <dataset_5dof>/libraries/dev --out-dir results/
```

A, B and B2's tight scoring need their own code, from the run folders above.
