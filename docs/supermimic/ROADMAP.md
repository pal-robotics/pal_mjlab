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

## M4: Online SuperMimic loop on `dev` (local 2026-10-24, cluster 2026-10-25; `m4/online_loop.md`, `m4/cluster_runs.md`)

- **Run:** Alg. A.1 from scratch, 12 h local GPU like PPO-Sim. The only deviations are bf16 and replay
  500 (D16). 27,510 iterations; `logs/rsl_rl/supermimic_wm_1hz/*_dev-local`.
- **Benchmark of all 27 checkpoints** (1296 `dev` clips):
  - SuperMimic matched PPO-Sim's 12 h tracking return at 5.7 h.
  - Best checkpoint (11.3 h): 3 failures and tracking return 0.832, against PPO-Sim's 25 and 0.742.
    Legs, arms and torso errors are 22–32% lower.
  - 10 of the 15 late checkpoints beat PPO-Sim.
- **Training oscillates.** A critic overestimation loop collapses the policy every 1.3–2 h, then it
  recovers to a higher peak. 4 of the 15 late checkpoints are in a collapse, the last one is on a
  downswing.
- **The world model and the learned failure head work online.** Imagination is within a few percent of
  the sim, and the head reaches AUC 0.999 once refit online.
- **Gate:** met at the peaks (about 2× faster than PPO-Sim); not met for stability. The critic ablation
  is still open.
- **Cluster runs, done 2026-10-25** (`m4/cluster_runs.md`, `m4/cluster_report.pdf`). The paper's
  exact settings (TF32, replay 1000) for 24 h on one Leonardo A100, next to PPO-Sim on the same setup:
  - PPO-Sim plateaus at 0.770 and 15 failures.
  - SuperMimic passes it at 9.3 h; best checkpoint at 22.3 h: 2 failures, 0.872, arm error 41% lower.
  - The oscillation remains (collapses at 8.8, 12.2, 14.1 and 21.4 h), so bf16 and replay 500 were not
    the cause.
  - A late failure mode: extreme imagined rewards blow up the critic loss.
  - Gate: met at equal hours and at the best checkpoint; stability not met; critic ablation open.
- **Stability study (running, `m4/stability.md`, D38).** User decision 2026-10-25: fix speed and
  stability before M5.
  - Speed: the imagination step and the world-model loss are compiled (D37).
  - Diagnosis: the main collapses come from the critic going wrong on live states. The late burst is
    post-fall world-model extrapolation, through the joint-limit penalty, poisoning the critic.
  - Runs: a second-seed baseline plus four ablations (no critic; failure head on the rule's features;
    survival-weighted critic loss; no privileged critic inputs), 12 h each on Leonardo.
- **Next:** adopt what fixes stability, then `full` directly on the cluster (M5).

## M5: Speed, stability, scale (options, user decides; `m4/cluster_report.pdf` §5)

User decisions 2026-10-25: (1) compile, (2) the stability ablations are under way in M4; no G1
control, multi-GPU or data sweep for now; `full` on the cluster once stability and speed are fixed.
The options as written in the report: Costs are local GPU time (free) or Leonardo A100 hours: 1 A100-h ≈ 8 local budget
hours, about 4,560 left (≈ 570 A100-h). In the suggested order:
1. **Speed.** Profile one iteration, then compile the whole imagination step and the world-model loss
   step (today only the FK). Tests how much of our 8× per-GPU gap to the paper is ours; the reproduction
   measured 4.5× from compiling. Every later option gets cheaper by that factor. Cost: code plus about
   2 h of local benchmarks.
2. **Stability, one change at a time,** one 12 h local run each (bf16 and replay 500 are not the cause),
   judged on the benchmark (collapse share, failure spikes, best and last):
   - (a) a calmer sampler: rate 0.05, unvisited bins easy;
   - (b) a small failure head on the rule's own features;
   - (c) the clean previous action in obs, noise only when executing (D9);
   - (d) policy learning rate or −J/H; critic without privileged inputs (D7); frozen critic
     normalization (D34);
   - (e) first log which reward term makes the rare extreme imagined rewards of the late cluster run.
   (a)–(c) are what the reproduction (CuboiLeo/SuperMimic-Reproduced) does differently.
3. **G1 control (D18):** our code on mjlab's G1 with BONES-SEED G1 clips (`dev`-sized), with the
   reproduction's curves as reference. Separates Kangaroo from implementation. Cost: G1 task, one local
   run.
4. **Critic ablation and push test** (paper Fig. 4b, §4.3), still open from M4. One local run.
5. **Multi-GPU:** explicit gradient all-reduce, lr × √N (as the reproduction). Needed for the paper's
   2-GPU result. Cost: code, about 2 A100-h to test.
6. **`full` (250.9 h) on the cluster,** SuperMimic and PPO-Sim: 24 A100-h per run (≈ 190 local h). Not
   expected to cure the oscillation; the paper is stable on LAFAN1's 40 clips too.
7. **The paper's two results** (3 h on 1 H100; 9.8 h on 2 H100, zero failures, on G1): 10–50 and
   60–160 A100-h at today's 2,060 it/h, a third to a quarter of that after (1). The ranges reflect D10 and
   the paper's unreported single-GPU speed.
8. **Data sweep** (1, 10, 40 h; paper Fig. 3b/c): local runs, or 24 A100-h each.
- **Gate:** failure counts by category on the full benchmark, and the compute-to-target curve.

## Later

- Kangaroo sys-ID DR in ξ (D14).
- A library-free ONNX export: the policy plus a goal-builder interface.
- Deployment.
- Velocity and sparse-WBC encoders (Appendix B).
