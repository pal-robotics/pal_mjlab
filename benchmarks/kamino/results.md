# Kamino against mujoco_warp on kangaroo_full

The closed-chain Kangaroo (4-DoF arms, 24 loops, actuators on the ball screws) on the flat velocity task, 2 ms timestep, decimation 10, measured through the full `ManagerBasedRlEnv.step`. Measured on 2026-11-05 on one GPU, shared with a resident inference service. Rollouts have no noise or randomization unless stated. Tables at the end come from `scripts/bench_kamino.py`.

## Result

**Recommendation: stay on mujoco_warp for training.** At settings fast enough for RL, Kamino is about 54 times slower, cuts typical loop error by about half, and blows up under exploration noise, so its 3 h run didn't learn. mujoco_warp's soft loops already stay within about 1.3 mm while walking.

| | mujoco_warp | Kamino, CRF 60 (default) | Kamino, dense LLT |
|---|---|---|---|
| Env-steps/s at 1024 worlds | 18,198 | 340 | 98 |
| Env-steps/s at 4096 worlds | 20,521 | 361 | not run |
| Loop error walking, mean / p95 / max (mm) | 0.111 / 0.652 / 1.26 | 0.054 / 0.554 / 21.0 | not run |
| Loop error zero action, mean / p95 / max (mm) | 0.070 / 0.267 / 0.445 | 0.031 / 0.130 / 17.4 | 0.009 / 0.034 / 0.329 |
| Solves reaching tolerance, zero action | n/a | 5% | 98% |
| Reward after 3 h of training | 80.6 | -0.09 | not run |

The bridge looks sound for normal motion: the baseline policy, trained on mujoco_warp, walks on Kamino for 20 s in all 256 worlds without a fall. Over a separate 10 s run it averaged 0.94 m/s on Kamino and 0.97 m/s on mujoco_warp, against a 1 m/s command.

## Speed

![Speed against loop error](speed_vs_violation.png)

Kamino's bars use its own body poses, described below.

- Kamino's speed barely changes with world count: 320, 340 and 361 env-steps/s at 256, 1024 and 4096 worlds. mujoco_warp goes from 9,873 to 20,521.
- Walking doesn't close the gap: 17,535 against 343 env-steps/s at 1024 worlds, 51 times slower.
- Speed follows the iteration budget: 647, 340 and 176 env-steps/s at 30, 60 and 120 PADMM iterations. Each PADMM iteration runs 30 inner linear solver iterations.
- Fewer iterations isn't an option: at 30, the worst loop reaches 861 mm and only 3% of solves reach tolerance.

## Constraint violation

Loop error is the distance between the two sites of each `<connect>`, read from `site_xpos` with the same code on both simulators. "Kamino bodies" is the same distance on Kamino's own body poses, before they are mapped to `qpos`, because that mapping can hide error.

Walking 20 s with the baseline policy is the fairest comparison, since neither simulator falls:

| Walking, 256 worlds | mujoco_warp | Kamino |
|---|---|---|
| Loop error, mean / p95 / max (mm) | 0.111 / 0.652 / 1.26 | 0.054 / 0.554 / 21.0 |
| Kamino bodies, mean / p95 / max (mm) | n/a | 0.020 / 0.078 / 84.8 |
| Mean by group: hip / ankle / knee (mm) | 0.054 / 0.142 / 0.132 | 0.032 / 0.079 / 0.020 |
| Loop velocity, mean / max (mm/s) | 2.06 / 87.8 | 1.93 / 2,500 |
| Tree joint residual, p95 / max (mm or mrad) | 0 by construction | 0.034 / 927 |
| Solves reaching tolerance in 60 iterations | n/a | 0% |

A separate 10 s walking run located Kamino's large maximums in time. All of them came in the first 0.5 s, as the robot settles from the reset pose:

| Walking 10 s, 256 worlds, 128,000 world-steps | mujoco_warp | Kamino |
|---|---|---|
| Worst loop error in the first 0.5 s (mm) | under 1.3 | 10.0 |
| World-steps over 5 mm after 0.5 s | 0 | 0 |
| World-steps over 1 mm after 0.5 s | 20,194 (16%) | 4,790 (3.7%) |

So once the robot is walking, Kamino holds its loops better. It isn't clean, though: some body exceeded 10 m/s on 673 world-steps (0.5%), peaking at 38 m/s, where mujoco_warp's fastest body reached 4.2 m/s. That run didn't record when those happened.

The zero-action rows aren't quiet standing. The home pose isn't balanced, so the robot settles, tips over near 3 s (timed on mujoco_warp), resets and lands again. Kamino also records one fall per world, and those events probably set its maximums.

Dense LLT is what converged Kamino looks like: about 7.5 times lower mean loop error than mujoco_warp, at 186 times the cost. More CRF iterations don't remove the outliers: 120 iterations still reach 133 mm.

Kamino's RL warm start and the position-matching one are indistinguishable here: same speed, same mean and p95. A single short check at 1024 worlds once showed 230 mm against 0.075 mm, but the longer runs don't confirm it.

## Training

| 3 h run | Baseline | Kamino |
|---|---|---|
| Worlds | 4096 | 1024 |
| Iterations | 2,025 | 151 |
| Samples | 199 M | 3.7 M |
| Final mean reward | 80.6 | -0.09 |
| Final episode length (steps) | 1000 | 16 |
| At 3.64 M samples: reward / episode length | 4.0 / 149 | -0.03 / 17.5 |
| Worst value loss over the first 150 iterations | 0.40 | 650,568 |

The Kamino policy learned to end episodes quickly: length fell from about 100 steps to 16 while reward rose towards zero, because each step was costing reward on average. It does the same on mujoco_warp (59.5 episode ends per world in 20 s, 57.2 on Kamino), so the result is in the policy, not only in the simulator.

The likely cause is blow-ups during exploration. In the logs, `foot_slip` averaged as low as -16 (baseline worst -0.005), and the learning rate sat at its 1e-5 floor on 79 of 151 iterations. A one-off run of the training environment, randomization on, with actions drawn from N(0, 1), shows the blow-ups directly:

| 1024 worlds, 120 env steps | mujoco_warp | Kamino |
|---|---|---|
| Max body speed (m/s) | 7.6 | 80,850 |
| World-steps with a body over 10 m/s | 0 | 1,386 (1.1%) |
| Max loop error (mm) | 2.53 | 679 |
| Worst single-step reward | -0.6 | -270 |

There were no NaNs in training, so the `nan` termination never fired. What triggers the blow-ups isn't known yet. This run can't separate random actions from randomization, and the bridge isn't ruled out.

Even with that fixed, the budget is the limit. The baseline reached reward 40 and 849-step episodes at iteration 300 (29.5 M samples, 27 min). At Kamino's 344 samples/s, the same samples take about 24 h.

## Caveats

- Kamino's contacts, limits and friction are hard where MuJoCo's are soft, so a difference in policy behaviour is expected and isn't by itself a defect.
- The two training runs differ in world count (4096 and 1024). Kamino's speed doesn't grow with worlds, so 1024 gives it more policy updates in 3 h.
- Kamino raises placeholder bodies to 0.01 kg (`min_body_mass`), adding 0.04 kg in total. It also skips joint friction below 0.05 (`min_frictionloss`).
- Metrics are read after resets, so the state a world ends in isn't measured on either simulator.
- The working solver settings came from a sweep on this model: CRF, 30 inner and 60 PADMM iterations, rho 0.05, tolerance 1e-3. Other solvers or tuning could do better. Kamino's own RL setting (9 inner iterations) diverged on foot contact.
- Both simulators use the on-load model fixes in `kangaroo_full_constants.py`. These belong upstream in `kangaroo_simulation`.

## Open question

Is +/-0.02437 m the real ankle screw travel? It stops pure ankle pitch at +0.52 rad, short of the +0.663 rad joint limit, which needs -0.0263 m. It's left unchanged.

## Benchmark tables

`stand_*` is zero action for 5 s and `walk_*` is a 1 m/s command for 20 s, both at 256 worlds. `policy_mjw` and `policy_kam` are the baseline and Kamino policies. "Falls per world" counts every terminated episode.

| Case | Env-steps/s | Physics substeps/s |
|---|---|---|
| speed_mjwarp_256 | 9873 | 170380 |
| speed_kamino_256 | 320 | 3237 |
| speed_mjwarp_1024 | 18198 | 239328 |
| speed_kamino_1024 | 340 | 3438 |
| speed_mjwarp_4096 | 20521 | 240384 |
| speed_kamino_4096 | 361 | 3622 |
| speed_kamino_1024_b30 | 647 | 6664 |
| speed_kamino_1024_b120 | 176 | 1754 |
| speed_kamino_1024_dense | 98 | 972 |
| speed_kamino_1024_rl_warmstart | 341 | 3428 |
| speed_mjwarp_1024_walk | 17535 | 247248 |
| speed_kamino_1024_walk | 343 | 3446 |

| Case | Loop mm mean | p95 | max | Kamino bodies mm max | Loop mm/s mean | Tree mm or mrad max | PADMM iterations | Converged | Falls per world | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| stand_kamino_b30 | 0.734 | 0.458 | 860.665 | 1925.958 | 712.31 | 2158.756 | 29.1 | 0.03 | 1.42 | 0.000 |
| stand_kamino_b120 | 0.068 | 0.145 | 132.529 | 67.925 | 4.58 | 975.539 | 93.4 | 0.61 | 1.00 | 0.000 |
| stand_kamino_dense | 0.009 | 0.034 | 0.329 | 1.064 | 0.32 | 42.650 | 37.2 | 0.98 | 1.00 | 0.000 |
| stand_kamino_rl_warmstart | 0.031 | 0.128 | 10.358 | 51.310 | 0.42 | 608.779 | 58.1 | 0.05 | 1.00 | 0.000 |
| stand_mjwarp | 0.070 | 0.267 | 0.445 | - | 0.22 | - | - | - | 1.00 | 0.000 |
| walk_mjwarp_policy_mjw | 0.111 | 0.652 | 1.256 | - | 2.06 | - | - | - | 0.00 | 0.000 |
| walk_mjwarp_policy_kam | 0.094 | 0.870 | 0.997 | - | 1.06 | - | - | - | 59.51 | 0.000 |
| stand_kamino | 0.031 | 0.130 | 17.440 | 65.403 | 0.43 | 828.148 | 58.1 | 0.05 | 1.00 | 0.000 |
| walk_kamino_policy_mjw | 0.054 | 0.554 | 20.975 | 84.822 | 1.93 | 926.831 | 60.0 | 0.00 | 0.00 | 0.000 |
| walk_kamino_policy_kam | 0.053 | 0.609 | 2.106 | 3.041 | 2.18 | 23.137 | 57.3 | 0.10 | 57.17 | 0.000 |

| Training run | Iterations | Hours | Mean reward | Mean episode length |
|---|---|---|---|---|
| mjwarp | 2025 | 2.99 | 80.65 | 1000 |
| kamino | 151 | 2.97 | -0.09 | 16 |
