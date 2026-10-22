# Arm/waist gains: 3.5 Hz vs 1 Hz (D27), 2026-10-22

**Question.** MAC/WBC lowered the arm and waist PD natural frequency to 1 Hz "to improve tracking".
Should the SuperMimic task keep PR #126's 3.5 Hz on every joint?

**Setup.** Identical runs except the arm and waist gains:
- PPO-Sim on `dev` (1,296 BONES clips, 2.56 h), 1,000 iterations, 8192 envs, seed 42, about 80 min each
  on the RTX 5060 Ti.
- Legs at 3.5 Hz in both.
- Gains from BeyondMimic's rule (Kp = armature·ω², ζ = 2, action scale 0.25·effort/Kp).

| task | arms + waist | arm Kp/effort | arm action scale |
|---|---|---|---|
| `Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers` | 3.5 Hz | 3.9–5.9 /rad | 0.04–0.06 rad/unit |
| `...-Grippers-ArmWaist1Hz` | 1 Hz | 0.3–0.5 /rad | 0.52–0.78 rad/unit |
| (G1 in BeyondMimic, main joints) | 10 Hz | 0.46–0.71 /rad | 0.35–0.55 rad/unit |

**Benchmark** (`eval.py`, paper F.1):
- Every clip once from frame 0, foot friction 1.0, no randomization or noise, deterministic policy.
- Decision metric: root-relative per-body position error, paired per clip.
- Reproduce with `src/pal_mjlab/scripts/supermimic/run_gain_ab.sh <task> <tag>`, then `compare_evals.py`.

## Results (final checkpoints, iteration 999)

| metric (mean over 1,296 clips) | 3.5 Hz | 1 Hz | 1 Hz better on |
|---|---|---|---|
| arm body position error | 21.2 cm | **15.3 cm** | 90.4% of clips |
| leg body position error | 11.3 cm | **8.6 cm** | 94.8% |
| torso body position error | 4.3 cm | **3.5 cm** | 70.1% |
| arm joint error | 0.59 rad | **0.45 rad** | 83.2% |
| tracking-only normalized return | 0.529 | **0.619** | 95.4% |
| failed clips | 162 | **144** | 58 fail only at 3.5 Hz, 40 only at 1 Hz |
| arm torque rate (Δτ/τ_max per step) | 0.024 | **0.020** | |
| arm torque saturation | 0.77% | **0.04%** | |
| waist joint error | 0.18 rad | **0.12 rad** | |

Over training, 3.5 Hz's arm joint error grows (0.45 → 0.59 rad from iteration 200 to 999) while its arm
torque rate falls: the policy buys smoothness by not following the arms. At 1 Hz the arm error is lower
at every checkpoint, and both runs are still improving at iteration 999. Full curves are in the
`compare_evals.py` output.

## Why

The policy's action is a setpoint, q_sp = q₀ + α·a with α = 0.25·τ_max/Kp, so:

a = 4·τ/τ_max + 4·(Kp/τ_max)·(q − q₀) + 4·(Kd/τ_max)·q̇

One unit is always 25% effort, but holding a joint away from the default pose and moving it costs
actions in proportion to Kp/τ_max and Kd/τ_max.
- **Action size.** Kangaroo's arms have about 100× G1's reflected inertia for 1–2× its effort. At
  3.5 Hz their Kp/τ_max is 7–10× BeyondMimic's, so following `dev` arm references needs |a| ≈ 8
  (median) to 41 (p95).
- **Action-rate penalty.** The penalty (‖Δa‖²) on arm motion costs up to 28% of the maximum per-step
  reward (p95), against 0.2% at 1 Hz.
- **What 1 Hz does.** It puts the arms back in BeyondMimic's regime, where the setpoint is a torque
  proxy. It doesn't make a better servo.

## Caveats

- **Short and single-seed:** one seed, 1,000 iterations, both runs still improving. Training clips
  only, as in the paper's benchmark.
- **Two factors at once:** arms and waist changed together, so their effects aren't separated.
- **Simulation only:** 1 Hz makes the real arm and waist servos 12× less stiff (natural frequency 3.5×
  lower, dominant time constant ~0.6 s instead of ~0.17 s). PAL's hardware experience with 3.5 Hz is
  not covered by this test.
- **Untested alternative:** keep 3.5 Hz and anchor the setpoint on the reference, q_sp = q_ref + α·a.
  It removes the same overhead without softening the hardware, but changes the paper's action
  definition.
