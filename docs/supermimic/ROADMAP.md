# SuperMimic on Kangaroo: roadmap

**Target.** One policy that tracks any retargeted clip, BONES-SEED first, on Kangaroo with 7-DoF arms.

**Method.** Trained as in SuperMimic (arXiv 2610.28480):
- a learned differentiable world model (WM), fit online from MuJoCo Warp transitions;
- the policy trained by BPTT over H=32 imagined steps;
- a critic bootstrapping beyond H.

**Not in scope yet:** the velocity encoder, sim-to-real.

Decisions are cited as Dn (see DECISIONS.md). Measured facts are in LEARNINGS.md. Every milestone ends
with a gate: if it fails, we stop and rethink before building on top.

## M0: Evidence and plumbing (done 2026-10-22)

- Branch `omm/supermimic` (D1), and the arm-axis fix (D3).
- Packed motion library: converter, plus the `smoke` (0.25 h) and `dev` (2.56 h) subsets (D11).
- Torch FK with MuJoCo-parity and real-fixture tests (D5).
- Sim benchmark (D15) and learning-update benchmark (D16).
- **Gate:** numbers in LEARNINGS.md.
- **Result:**
  - 5 ms × 4 runs at 201k raw steps/s (4096 envs), with a four-bar error under 3.2 mm.
  - Paper nets in bf16 with compiled FK: 0.74 s of learning per iteration, 7.7 GB.
  - Eager FK is unusable at 3–5 s per update, so compile it (D5).
  - Locally ~8× slower than an H100, so M5 goes to Lambda.

## M1: Paper-faithful env + PPO-Sim baseline

**Status (2026-10-22).**
- Done:
  - env, PPO config and tests (35, including FK parity with the simulator and fall detection);
  - a 3-iteration check at 8192 envs (4.98 s per iteration);
  - an audit against the paper's methodology: tracked bodies, gains, goal content, convex-hull term,
    calibration offsets (D0, D25, D26).
- Gain A/B (D27): 1 Hz arms and waist track better than 3.5 Hz (`m1/gain_ab.md`).
- PPO-Sim baseline done (`m1/ppo_sim_baseline.md`): 25/1296 failures, tracking return 0.742 after 11.9 h.
  **M1 gate met.**
- Still to do: logging rollouts for M2, at the start of M2 (user decides when).

- Task `Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers` (D2, D6, D12–D15):
  - library-backed motion command: goal builder, clip-end termination (D8), paper return-based reset
    sampling;
  - shared `terms.py`: reward, termination, proprio obs;
  - `state.py`: state extraction (D4); `context.py`: ξ readback.
- A `-PPO` registration with BeyondMimic hyperparameters (D17).
- `eval.py`: each clip played once from frame 0, no DR or noise. It reports normalized return and
  failures by category.
- Parity tests: terms on extracted state equal the env's manager terms; obs equal the noise-free obs group.
- **Gate:** PPO learns `dev` stably. Keep the curve as the baseline, and log rollouts with action noise
  0.1 for M2.

## M2: Offline world model (the main risk)

- `replay.py` and `networks.py`. Fit the WM on M1 rollouts with k=8 windows (Eq. 1).
- Report held-out 8- and 32-step open-loop error per component: root, joints, and body positions after FK.
- Ablations:
  - "10 batches" read as 10 steps vs one large batch (D10);
  - delta vs absolute prediction;
  - normalizer.
- M2b: per-step ∂J/∂a norms through the WM at H=32, a quick check for exploding or vanishing gradients.
- **Gate:** 32-step rollouts stay bounded, body-position error at 8 steps is within a few cm, and the
  gradient norms are sane.

## M3: Policy on a frozen world model

- `imagine.py`: Eq. 2 with the survival weight, and the critic with λ-returns (Eq. 3). Paper optimizer and
  clipping.
- Train on the frozen M2 WM, evaluate in sim, and log the imagined-vs-sim return gap.
- Compare H=32 with and without the critic.
- **Gate:** sim return tracks imagined return, and the critic run beats the no-critic run.

## M4: Online SuperMimic loop on `dev` (local GPU)

- `runner.py`, Alg. A.1:
  - collect ≈2.5 steps per iteration;
  - 10 WM steps;
  - 1 BPTT step;
  - critic fit;
  - cosine LR schedule;
  - checkpoint and resume (rebuild the replay on resume; games preempt the GPU).
- Compare with PPO-Sim on wall-clock to a target return. Light H ablation (16 vs 32).
- **Gate:** ≥ PPO-Sim on `dev`. If it misses, run the G1 control (D18) before tuning anything
  Kangaroo-specific.

## M5: Scale

- Local dataset sweep: 1, 10, 40 h, to see the compute-to-target trend (paper Fig. 3b/c).
- Paper config on one Lambda H100 80 GB with the full 250.9 h (`--subset full`, 13.9 GB). Budget about
  10 h, roughly $30.
- Benchmark all 124,079 clips.
- **Gate:** failure counts by category, and the compute-to-target curve.

## Later

- Kangaroo sys-ID DR in ξ (D14).
- A library-free ONNX export: the policy plus a goal-builder interface.
- Deployment.
- Velocity and sparse-WBC encoders (Appendix B).
