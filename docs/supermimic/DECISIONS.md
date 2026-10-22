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

**D26. Actuator gains: BeyondMimic's rule at one natural frequency, 3.5 Hz for every joint.** (user)
- The rule: Kp = armature·ω², Kd = 2ζ·armature·ω with ζ = 2. Action scale is 0.25·effort/Kp.
- PAL's identified armatures, friction and effort limits are kept as they are.
- The 1 Hz arm/waist frequency on MAC/WBC was an experiment for that task. With 3.5 Hz the arm and
  waist Kp rise 12× (e.g. 19 to 234 for the shoulder) and the legs are unchanged.
- Built in `kangaroo/env_cfgs.py` (`kangaroo_robot_cfg`), so the WBC tasks stay as they are.
