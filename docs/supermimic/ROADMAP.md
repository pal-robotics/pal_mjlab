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
- Rollouts for M2 logged at the start of M2 (D29).

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

## M2: Offline world model (done 2026-10-23, `m2/world_model.md`, `m2/report.pdf`)

- **Built:**
  - `rl/replay.py`: windows within episodes, held-out clips.
  - `rl/world_model.py`: the paper's MLP, Eq. 1 loss, termination head.
  - Scripts: collect, train, eval (open loop, by body group after FK), gradient check (model gradient
    vs random, replayed in the sim).
- **Data:** 6.1M transitions from three PPO-Sim checkpoints (D29).
- **Result, main model (D30), held-out clips:**
  - Body error 0.86 cm at 8 steps and 3.8 cm at 32, 4–6× below no-change. Imagined return within 1.1%
    at 32 steps.
  - Gradient steps improve the simulated return from 97–100% of starts, vs 44–47% for random. The model
    is optimistic at H = 32: predicted gain 1.3–1.6× the real one.
- **Ablations:**
  - k = 1 diverges at 32 steps.
  - Absolute targets are 2× worse.
  - Width 1024 is 15–20% worse.
  - An unseen policy costs about 15%.
  - The "10 batches" reading (D10) moves to M4.
- **Termination:** the learned head reaches AUC 0.937, against a gate of 0.95. The env's rule on predicted
  states reaches 0.995.
- **Gate met**, except the learned head. Following the paper, M4 keeps the head and compares it with the
  rule throughout (D31).

## M3: Imagination and the policy update (done 2026-10-23, `m3/imagination.md`, `m3/report.pdf`)

- **Built:**
  - analytic heads (D32);
  - the online replay buffer (D33);
  - imagination with Eq. 2 and Eq. 3 and the paper's optimizers;
  - the Alg. A.1 runner, registered with the mjlab CLI;
  - dataset and subset options with a transferable dataset folder (D35).
  108 tests pass.
- **Check 1 (closed loop, PPO-Sim actor):** imagined vs simulated 32-step return −0.9%, correlation
  0.995, body error 2.1 cm. **Pass.**
- **Check 2 (300 updates on the frozen M2 model):**
  - The machinery works: without the critic the real 32-step return rises 7%, and imagination tracks
    the sim throughout.
  - Whole-clip tracking degrades in every frozen-model run:
    - no critic: 829 of 1296 clips fail (myopic objective);
    - with the critic at the paper's learning rate: 406–415;
    - lr 2e-5: 83.
  - The frozen learned head missed the new policy's falls (0.4% predicted vs 8.3% in the sim).
  - Read as the reason the paper refits online, not a code defect. M4 watches these signals.
- **Smoke run of the M4 task:** 1.57 s per iteration, 11.3 GB with bf16 and replay 500. The paper
  settings don't fit the local 16 GB GPU.

## M4: Online SuperMimic loop on `dev` (next; launch on the user's word)

- `rl/runner.py`, Alg. A.1, from scratch, with the paper's settings:
  - collect ≈2.5 steps per iteration;
  - 10 WM batches (D10);
  - 1 BPTT step with the learned termination head (D31);
  - critic fit;
  - cosine LR schedule;
  - checkpoints every 2000 iterations; resume with `scripts/supermimic/train.py`.
- **Configurations** (`README.md`): local `dev` (bf16, replay 500), cluster `dev` and `full` (TF32,
  replay 1000).
- **Watch:**
  - sim episode failures and length;
  - the imagined-vs-sim return gap;
  - the learned head vs the rule (AUC, and predicted vs simulated failures);
  - the whole-clip benchmark of checkpoints (`eval.py`): M3 showed that 32-step checks miss later falls.
- Compare with PPO-Sim on wall-clock to a target return.
- Ablations, in order of value:
  1. without the critic: the paper's ablation (M3 shows it myopic on a frozen model);
  2. termination by the env's rule (one switch), if the head lags it in the logs (D31);
  3. H = 16, only if the imagined-vs-sim gap grows (the model is optimistic at H = 32, M2).
- **Gate:** ≥ PPO-Sim on `dev`, and the critic run beats the no-critic run. If it misses, run the G1
  control (D18) before tuning anything Kangaroo-specific.

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
