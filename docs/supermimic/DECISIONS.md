# SuperMimic on Kangaroo: decisions

The paper is arXiv 2610.28480. No code was released, so we build from the paper text only. Each entry
records the decision, why we took it, and what would make us revisit it. **(user)** marks decisions the
user made; the rest are proposals, open to change. Entries are numbered so commits and issues can cite
them, e.g. "D7".

**D0. Follow the paper's methodology; the WBC branch is a starting point, not a constraint.** (user)
- Robot config, XML, gains, loader and sampling change whenever that makes the implementation easier.
  Every change gets an entry here.
- Anything inherited from WBC or PAL tasks must either follow the paper or have a written reason.
- Audited 2026-10-22, which led to D25 and D26 and the changes to D6, D12 and D14.
- Kept on purpose, each with its reason:
  - the IMU frame for angular velocity and gravity (the real sensor sits on the torso);
  - the critic's privileged root-pose error and time left (D7, D22);
  - the CoM offset sampled at startup (D14);
  - BeyondMimic's reset noise (the paper keeps BeyondMimic's setup except for pushes).

**D1. Branch and layout.**
- Branch `omm/supermimic`, created from `MAC/WBC`.
- Separate task in `src/pal_mjlab/tasks/supermimic/`, laid out like mjlab's tracking task; the WBC task
  is left alone (user):
  - `supermimic_env_cfg.py`: robot-agnostic base config with the paper's numbers.
  - `mdp/`: the command, sampler, observations, rewards, terminations, and the pure `tracking.py` terms.
  - `kangaroo/`: robot config, PPO config, task registration.
  - `rl/`: networks.
  - `kinematics.py`, `library.py`, `state.py`: core pieces shared with the future world-model code.
- Docs live here. The library converter is `scripts/bones_to_library.py`; benchmarks and the resumable
  launcher are in `scripts/supermimic/`.

**D2. Interface: the basic joint-following tracker.** (user)
- The goal is the reference trajectory over K=10 future frames, each frame being BeyondMimic's command
  (D6).
- The reward is the paper's body-space Table A.2.
- The WBC sparse end-effector interface becomes a later downstream encoder (paper Appendix B).

**D3. Arm 5/7 axis is +z in `kangaroo_grippers.xml`.** (user)
- Why: GMR is authoritative and the robot URDF agrees. The ranges are unchanged.
- Effect: the BONES clips load unchanged, matching GMR body positions exactly, with 0% of actuated frames
  out of range.
- Before: the axis was −z, with arm bodies off by up to 0.18 m and arm_5 out of range on 15–21% of frames.
- Side effects:
  - This also changes the existing WBC grippers task. Any npz converted with the old axis must be
    reconverted.
  - `kangaroo_hands.xml` is untouched (5-DoF arm, symmetric range).

**D4. World-model state is in generalized coordinates, heading-invariant, about 77 dims.**
- Layout: root height, root tilt (6D rotation in the heading frame), root linear and angular velocity in
  the root frame, and all 32 joint q and q̇, including the passive femur/knee (that keeps it Markov).
- The WM predicts normalized deltas, plus planar displacement and Δyaw in the heading frame. The world
  root pose is integrated analytically.
- Inputs: (ŝ, a, ξ). A running mean/std normalizer is updated only in the WM fit and frozen during policy
  updates.
- Why: compact, exact symmetry, cheap.
- Revisit if the per-component WM error in M2 is dominated by the representation.

**D5. Heads.**
- Analytic reward and observation through a torch FK of the robot tree (`kinematics.py`). This gives
  exact rewards, and the sim and imagination use the same functions.
- Learned termination d_ω(ŝ, g), trained with BCE, as in the paper; its calibration is logged.
- M0 cost: the FK must be `torch.compile`d. Eager it is launch-bound (3–5 s per update); compiled it adds
  ~20% over no FK.
- Revisit, with a learned reward head as the fallback, if FK gradients become a problem.

**D6. Goal frame, for each of t+1..t+10.**
- Contents: BeyondMimic's command without state estimation, which is the variant matching the paper's
  proprioception (no base linear velocity):
  - reference q and q̇ of the 28 actuated joints;
  - reference root orientation relative to the current robot root (6D).
  - That is 62 values per frame, 620 for the goal.
- An earlier version also had reference root velocity and height. Those came from WBC's sparse
  interface, so they were removed.

**D7. Critic input** is (ŝ, g, ξ) plus the privileged root-pose error (planar position, yaw). The
heading-invariant ŝ can't see drift, and the reward has a global root-position term.

**D8. Clip end.**
- In sim, the episode ends at the clip end (no teleport to a new clip), so replay windows never mix clips.
- In imagination, the return stops at the last frame, with no bootstrap. Goal frames past the end repeat
  the last frame.

**D9. Replay.**
- Contents: an E × 1000 ring of state, world pose, executed noisy action, ξ, (clip, frame), terminated
  flag and episode-start flag.
- WM windows stay inside one episode.
- Imagination starts carry the previous 9 steps, to rebuild the 10-frame obs history. a_{t−1} in obs is the
  executed action.
- Env: our runner uses `auto_reset=False` so terminal states are stored. Pushes are removed.

**D10. Reading the ambiguities.**
- UTD 0.4 = 2.5 vector env steps per iteration, via a fractional accumulator. This fits the paper's
  320k iterations in ~9.8 h.
- "10 batches of 2048 windows" = 10 WM optimizer steps per iteration; ablate in M2.
- A warm-up fills the replay before the first update.

**D11. Motion library: one packed folder.**
- `qpos.npy` (N, 39), `qvel.npy` (N, 38) in MuJoCo qvel layout, and `clips.json`. About 308 B per frame.
- Body poses come from FK on the fly.
- Full set: 13.9 GB fp32, versus ~44 GB for the WBC loader's per-body layout.
- Local runs use subsets on the GPU; the full set goes on the H100.

**D12. Reward.**
- Paper Table A.2 only: weights and stds over the 14 bodies of D25, action rate −0.1, joint limits −2.
- The joint-limit term uses the hard limits of the actuated joints: references exceed the 0.9× soft
  limits on up to 27% of frames.
- The hip/ankle convex-hull penalties (PAL's coupled actuator limits) were removed: they aren't in the
  paper. They come back with sys-ID DR before hardware work.

**D13. Termination.** The paper's rule only: |e_z| > 0.25 m or tilt > 0.8. No end-effector termination,
no self-collision term.

**D14. Domain randomization.** (user)
- v1 uses the paper's set as ξ: foot friction, joint calibration offsets, pelvis_2 CoM.
- Calibration offsets apply only to the 28 measured (actuated) joints, since the passive four-bar joints
  have no encoders. So ξ is 1 + 28 + 3 = 32 dims.
- Friction and calibration are resampled every episode. The CoM offset is sampled once per env at
  startup.
  - Why: resampling it at reset triggers mujoco_warp `set_const` over all worlds, which took the step
    from 72 to 235 ms at 8192 envs (M1).
  - Coverage: with thousands of envs the CoM distribution is still covered, and the world model gets ξ
    either way.
- Kangaroo sys-ID DR (frictionloss, damping, armature) joins ξ before any hardware work.

**D15. Sim settings: 5 ms × 4 substeps, as in the paper.** (user)
- Solver iterations 10, ls_iterations 20.
- The four-bar's solref 0.004 gets clamped at 5 ms. M0 measured that as fine: loop residual 0.17 mm mean,
  3.2 mm max, no NaNs.
- Gains: see D26.

**D16. Two configs.**
- "paper": the exact hyperparameters and TF32, for a Lambda H100.
- "dev": the local 16 GB GPU.
  - The same nets and batch 4096, but bf16 autocast instead of TF32: 7.7 GB measured vs 12.7 GB.
  - Replay of 500 steps per env, and data subsets.
  - Smaller nets only if memory forces it.
- Every deviation is logged in the run config.

**D17. Baseline: PPO-Sim (paper §4.1).** (user: paper numbers, local GPU)
- Task `Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers`: the same env, rsl_rl PPO with BeyondMimic's
  hyperparameters (mjlab G1 tracking), and 8192 envs (twice SuperMimic's).
- Actor: the paper's encoder-decoder (`rl/models.py`, 2048-1024-512-256 each, z = 32 with tanh), with a
  Gaussian head.
- Critic: 256-128, with the privileged inputs of D7.
- Budget: about 12 h of local GPU on `dev`. `scripts/supermimic/train.py` resumes from the latest
  checkpoint when the queue reruns a preempted job.

**D21. Return-based sampler details** (paper D.1 leaves them open).
- G_b is normalized per bin by the best return attainable from that start (the clip may end inside the
  discount horizon). With one global maximum, every clip's last second would look like a failure and
  take the 40% branch.
- G_b starts at 0, so unvisited bins count as hard. Each visit updates it as an EMA with rate 0.2.
- The "past weight" is an EMA (rate 0.01) of the softmax distribution.
- Returns are Monte-Carlo over the sim episode. Truncation at 10 s barely matters, since γ^500 ≈ 0.007.

**D22. Episode structure.**
- An episode starts at a sampled frame, with BeyondMimic's reset noise on root pose, velocity and joints.
- It runs to the clip's last frame (`motion_end`, terminal, no bootstrap, D8), a failure, or a 10 s cap
  (truncation).
- The critic also gets the normalized time left in the clip (capped at 3 s). Otherwise it can't predict
  the return ending at the clip end.

**D23. Simulation buffers.**
- `nconmax` is 50. It is a contact pool shared by all worlds, and about 15 contacts per world are in use.
- `njmax` is 300. It is a hard per-world cap, and fallen robots reach about 280 rows; 160 overflowed.
- Measured: `nconmax` 100 cost 10.7 GB at 8192 envs, `nconmax` 40 cost 5.2 GB.

**D24. Kinematics in the env.**
- Robot and reference bodies come from the torch FK on (qpos, qvel), compiled, and are cached per env step.
- The reward, termination and (later) imagination share them.
- The FK model is the robot's spec as attached to the scene (names prefixed `robot/`). That makes it the
  exact simulated tree, and tests check it against mjlab's body data.

**D18. G1 control.** The algorithm core is robot-agnostic. If Kangaroo misses a gate, run the same code
on mjlab's G1 with the BONES G1 clips (`kangaroo_bones_seed/data/g1_npz`). That separates implementation
bugs from robot difficulty.

**D19. Sources.** Paper text only. The demo's license forbids reverse engineering, so never read its JS or
weights.

**D20. No extra data filtering.** (user) `final_kept` already applies the paper's name and feasibility
filters on G1, and again on Kangaroo after retargeting.

**D25. Tracked bodies: the paper's 14 (BeyondMimic's G1 set), mapped by function.**

| G1 body | Kangaroo body | why |
|---|---|---|
| pelvis | `base_link` | the root, as in the paper |
| torso_link | `pelvis_2_link` | after the two waist joints |
| hip_roll_link | `leg_*_3_link` | hip roll |
| knee_link | `leg_*_knee_link` | Kangaroo's serial leg is a hip-ankle slide; the knee exists only as the passive four-bar body |
| ankle_roll_link | `leg_*_5_link` | foot |
| shoulder_roll_link | `arm_*_2_link` | shoulder |
| elbow_link | `arm_*_4_link` | elbow |
| wrist_yaw_link | `arm_*_7_link` | last arm link |

The first version copied WBC's list. It came from the retarget's IK table, which uses rotations only for
these links. That list had two problems:
- **The ankle was tracked twice and the knee not at all.** It used `leg_*_4_link` as the "knee", but
  `leg_*_4_link`, `leg_*_5_link` and `leg_*_length_link` all sit at the ankle (measured 0.0 m apart).
- **The wrist was an end-effector frame.** It used `arm_*_tip_link`, which sits at `arm_*_7_link` in
  pal_mjlab but is rotated 45° in GMR.

**D26. Actuator gains: as in pal_mjlab PR #126 (`feat/7dof_rough`, = main).** (user)
- `kangaroo_constants.py` on this branch is that branch's file.
- The rule is BeyondMimic's: Kp = armature·ω², Kd = 2ζ·armature·ω with ζ = 2, at 3.5 Hz for every joint.
  Action scale is 0.25·effort/Kp. PAL's identified armatures, friction and effort limits are kept.
- The 1 Hz arm/waist split on MAC/WBC was an experiment for that task and is reverted here; it also
  affects the WBC tasks on this branch. With 3.5 Hz, arm and waist Kp rise 12× (e.g. 19 to 234 for the
  shoulder) and the legs are unchanged.
- The task uses the stock `get_kangaroo_grippers_robot_cfg()` and `KANGAROO_GRIPPERS_ACTION_SCALE`, so
  there is no gain logic of its own.

**D27. Arm/waist gains: measure, don't assume. 3.5 Hz vs 1 Hz A/B (2026-10-22).** (user)
- **Question.** MAC/WBC lowered the arm and waist frequency to 1 Hz "to improve tracking". With
  BeyondMimic's rule, Kangaroo's arms at 3.5 Hz have Kp/τ_max ≈ 4–6 per rad (G1: ≈ 0.6), so their action
  scale is 0.04–0.06 rad per unit. That makes reference arm poses need |a| ≈ 8 (median) to 41 (p95), and
  makes the action-rate penalty expensive.
- **Runs.** PPO-Sim on `dev`, 1,000 iterations, 8192 envs, seed 42:
  - `Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers` (3.5 Hz);
  - `...-ArmWaist1Hz` (arms and waist at 1 Hz, same rule).
  Both use `run_gain_ab.sh`.
- **Decision metric:** root-relative per-body position error on the `dev` benchmark (`eval.py`), arms
  first, paired per clip.
- **Guardrails:** failures, tracking-only normalized return, torque rate and saturation.
- **Result** (`m1/gain_ab.md`): 1 Hz is better on every metric. Arm error 21.2 → 15.3 cm (better on 90% of
  clips), legs 11.3 → 8.6 cm (95%), tracking return 0.53 → 0.62 (95%), failures 162 → 144, arm
  saturation 0.8% → 0.04%. It's a single seed, sim only, at 1,000 iterations.
- **Decision (user, 2026-10-22):** 1 Hz arms and waist for the 12 h PPO-Sim run.
  - Task `Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz`, experiment
    `supermimic_ppo_sim_1hz`, GPU task 67.
  - Started fresh from iteration 0 on commit 21d5962, for a clean baseline curve.
  - Hardware gains stay a separate question for PAL.

**D28. Passive joints: never in the actor's inputs; allowed in privileged inputs (flagged 2026-10-23).**
- **Actor.** It sees only the 28 actuated joints (`REGEX_ALL_ACTUATED_JOINTS` excludes `leg_*_femur_joint`
  and `leg_*_knee_joint`): proprio q, q̇ and actions, and the goal's reference q and q̇. Verified on the
  running job's observation table: proprio 10 × (28 + 28 + 3 + 3 + 28), goal 10 × (28 + 28 + 6).
- **Privileged inputs.** The critic's heading state (D4) has all 32 joints, including the passive q and
  q̇. The tracking reward uses the passive knee body (D25). Both are training-only.
- **Revisit if:**
  - the world model's observation head (M2/M3) starts feeding anything derived from passive joints back
    to the policy;
  - a distilled or student policy (Appendix B encoders) copies the critic's inputs;
  - the real robot turns out not to measure `leg_*_length_joint`, which counts as actuated here.
- **Guard:** `tests/supermimic/test_env.py::test_observation_layout` pins the actor groups' sizes.

**D29. M2 data and split (2026-10-23).**
- **Policies:** PPO-Sim (1 Hz arms/waist) checkpoints at iterations 1000, 4000 and 8800. Mixing skill
  levels gives good tracking, poor tracking and falls.
- **Collection:** 4096 envs × 500 steps each on `dev`.
  - Action noise N(0, 0.1²) on the deterministic policy.
  - Paper DR per episode; ξ is logged (friction 1, calibration 28, torso CoM 3).
  - `auto_reset=False`, so terminal states are logged and windows never span a reset (D8, D9).
- **Held-out set:** 10% of clips by crc32 of the clip name, the same in every data set and every script.
- **Revisit if:** M4's online replay replaces this offline data.

**D30. World-model configuration for M3/M4 (2026-10-23, `m2/world_model.md`).**
- **Training:** k = 8 windows (paper Eq. 1).
- **Network:** MLP 2048×3 ELU on standardized (state, action, ξ), with statistics frozen after
  computing them from 200k transitions.
- **Targets:** the state change plus the heading-frame pose change.
- **Loss scales:** 0.1 m and 0.1 rad for the integrated pose.
- **Checkpoint:** the best held-out 8-step loss.
- **Why:** each ablation is worse.
  - k = 1 diverges at 32 steps.
  - Absolute targets are 2× worse.
  - Width 1024 is 15–20% worse.
- **Revisit if:** the online loop (M4) shows the arm error limiting tracking. The arms carry the largest
  model error, and loss weighting by body error after FK is the first thing to try.

**D31. Failure probability in imagination: the paper's learned head, compared against the env's rule
at every step (user, 2026-10-23).**
- **Default (M3 and M4): the learned head d_ω, as in the paper.** It is trained with the model (BCE on
  the logged failure flags), online in M4.
- **The rule is one switch away.** The imagination config takes `termination = "learned" | "rule"`.
  - "rule" is d̂ = σ((m − 1)/τ), τ = 0.05, with m = max(|z − z_ref| / 0.25, |g_z − g_z,ref| / 0.8): the
    env's own failure condition, evaluated on the predicted state.
- **Both are always computed and logged,** whichever one drives the gradient (the other without
  gradient):
  - the AUC of each on freshly collected sim transitions, scored before the model trains on them;
  - the mean survival weight c at H under each, and the share of rollouts where they disagree (one
    below 0.5, the other above);
  - sim failures vs imagined failures of the same policy.
- **Why compare.** Offline (M2) the head reached AUC 0.937 (0.872 on an unseen policy, 0.954 even on
  true states), against 0.995 for the rule. The likely cause is that only 0.05% of transitions are
  failures. Online it may do better (user's hypothesis): early training has many more failures. But late
  training makes failures rare again, and the policy can exploit the head's false negatives.
- **Run the rule variant** (M4 ablation) if the head's online AUC stays below the rule's, or if sim
  failures clearly exceed imagined ones.

**D32. Analytic heads read the training env (M3, `rl/heads.py`).**
- **What they compute, from the predicted state:** reward r̂ (all reward terms, with the env's weights,
  stds and dt), the noise-free observation ō (Table A.1, per-term 10-frame history in the env's layout,
  calibration offset from ξ), the goal, the critic input (D7), the failure rule, and the clip end.
- **Unsupported terms raise.** An env term without an analytic head stops the run, so imagination
  never silently optimizes a different objective than the simulator.
- **Parity is tested, not assumed** (`test_heads.py`): along random-action rollouts with resets, the
  heads on the true states match the env's goal, critic input and 10-frame proprioception to 1e-4, its
  reward to 1e-5, and its failure and clip-end flags exactly.

**D33. Online replay rows (D9 implemented, `rl/replay.py`).**
- **One row per state.** A transition row holds the state and executed action. When an episode ends,
  its terminal state is written as an extra row without an action. Windows are k consecutive
  transition rows plus the next row, so they never span two episodes, and no next-state copy is
  stored. About 2.3 GB at 4096 envs × 1000 rows.
- **Imagination starts** are transition rows. Their 10 history rows are rebuilt into observations
  with fresh sensor noise. Rows before the episode's first row repeat it, like the env's backfill;
  the env copies one noisy frame there, while imagination draws noise per frame.

**D34. Normalization (paper silent).**
- **Actor and critic:** running statistics of the sim observations during collection, as in PPO-Sim.
  Inside imagination they are fixed and differentiable.
- **World model:** running mean/std over every collected transition (D4). Policy steps never change
  them.

**D35. Motion data from the CLI, one transferable dataset folder (user, 2026-10-23).**
- **Dataset folder** `kangaroo_bones_seed/dataset/` (README inside) holds everything:
  - `manifest.parquet`;
  - `motions/<date>/<clip>.pkl`, the retargeted GMR pickles;
  - `libraries/<subset>/`, the packed training libraries (clips.json, qpos.npy, qvel.npy).
  Copy it with rsync, or download it from the motion browser: the motions zip, or each library file
  resumably. `data/` stays the dataset pipeline's workspace; its old paths are symlinks into
  `dataset/`.
- **CLI options of the motion command:**
  - `dataset`: that folder; relative paths are resolved against the working directory.
  - `subset`: `smoke`, `dev` (2.5 h), `full`, or a number of hours (`10`, `10h`).
  - `library_path`: an already packed folder; overrides both.
- **Packed on first use** into `<dataset>/libraries/<subset>`. The pack is written to a temporary
  folder and renamed when complete, so an interrupted pack never looks valid.
- **Hour subsets** are stratified by category like `dev` (`dev` = `2.5h`) and nested (1 h ⊂ 10 h ⊂
  40 h), for the M5 data sweep.
- **Run configurations:**
  - local `dev`: bf16 and replay 500 (D16);
  - cluster `dev` and `full`: the paper's settings (TF32, replay 1000). `full` puts the 13.9 GB
    library on the GPU.
- **Resuming:** `scripts/supermimic/train.py` resumes the newest checkpoint of runs with the same
  `--agent.run-name`.

**D36. Judge and keep SuperMimic checkpoints by the benchmark (M4, 2026-10-24).**
- **Why:** M4's training oscillates, so the last checkpoint is often in a collapse (0.684 vs the best
  0.832). Training-time signals mix failures and sampled starts.
- **How:**
  - Benchmark every checkpoint with `eval.py` (1296 `dev` clips, about 2 min each).
  - Report the best one together with the share of late checkpoints in a collapse.
  - Compare runs at equal training hours, never at equal iterations.
- **Revisit when** training is stable. The paper's exact settings did not stabilize it (cluster run,
  2026-10-25: collapses up to 1287 failures at 8.8 h; the last checkpoint is good, but it comes 1 h after
  a dip). D36 stays.
- **Cluster runs:** the checkpoints are copied to helium and benchmarked on the local GPU, with the same
  protocol and no cluster budget (`m4/cluster_runs.md`).

**D37. Compile the imagination step and the world-model loss (user, 2026-10-25).**
- **Why:** a profile of one local policy update (1.09 s) shows about 70,000 small aten calls and
  1.1 s of CPU time. The update is bound by launching kernels, not by the GPU. The reproduction compiles
  the same step and measured 4.5×.
- **How:**
  - `make_step` is one imagined step as a function of tensors only; the sensor noise is sampled
    outside and passed in.
  - `torch.compile(dynamic=False)` on that step and on `window_loss`. The loop over H stays in Python.
  - One flag, `--agent.compile` (default on; replaces `compile_kinematics`), covers the kinematics
    too.
  - A test checks that the traced step and loss (`aot_eager`) give the eager rollout, objective, loss
    and gradients.
- **Measured on a Leonardo A100 (TF32, smoke test):** 1.12 s per iteration against 1.78 s uncompiled
  (policy 1.37 → 0.77 s, world model 0.25 → 0.19 s); one-time compile of about 12 min (7 graphs). It is
  1.6×, short of the reproduction's 4.5×: the backward pass, the critic fit and the eager parts of the
  loop remain. More speed work is open (CUDA graphs, compiling the critic fit).
- **Also removed** the unused "fp32" precision: only the paper's TF32 and local bf16 remain (user: no
  fp32 option).
- **Known limit:** locally, the compiled update's memory cache leaves too little for MuJoCo Warp
  (simulator out of memory at 16 GB). The stability runs therefore go to Leonardo.

**D38. Stability study (user, 2026-10-25, `m4/stability.md`).**
- Five 12 h runs on Leonardo in parallel, with the paper's settings: a second-seed baseline and four
  one-change ablations. The four are no critic, a failure head on the rule's features, a
  survival-weighted critic loss, and no privileged critic inputs.
- Their switches default to the paper's behavior and are removed after the study, except the change
  we adopt.
- A diagnostic, `scripts/supermimic/reward_extremes.py`, splits imagined rewards and the critic's error
  at the predicted fall. Every run now logs the imagined reward's extremes.
