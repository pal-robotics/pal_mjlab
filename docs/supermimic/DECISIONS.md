# SuperMimic on Kangaroo: decisions

The paper is arXiv 2610.28480. No code was released, so we build from the paper text only. Each entry
records the decision, why we took it, and what would make us revisit it. **(user)** marks decisions the
user made; the rest are proposals, open to change. Entries are numbered so commits and issues can cite
them, e.g. "D7".

**D0. The WBC branch is a starting point, not a constraint.** (user) Robot config, XML, gains, loader and
sampling change whenever that makes the implementation easier. Every change gets an entry here.

**D1. Branch and layout.**
- Branch `omm/supermimic`, created from `MAC/WBC`.
- Separate task in `src/pal_mjlab/tasks/supermimic/`; the WBC task is left alone (user).
- Docs live here; the library converter is `scripts/bones_to_library.py`; benchmarks are in
  `scripts/supermimic/`.

**D2. Interface: the basic joint-following tracker.** (user)
- The goal is the reference trajectory over K=10 future frames: reference q and q̇ plus root terms (D6).
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
- Contents:
  - reference q and q̇;
  - reference root orientation relative to the current robot root (6D);
  - reference root linear and angular velocity in the reference frame;
  - reference root height.
- No world position, consistent with no state estimation (the paper's proprio has no base linear
  velocity).

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
- Paper Table A.2 weights and stds, over WBC's 14 tracked bodies.
- The joint-limit penalty (−2) uses the hard limits: references exceed the 0.9× soft limits on up to 27%
  of frames.
- Kangaroo addition: hip/ankle convex-hull penalties (half-spaces, ReLU, differentiable). They are a
  hardware safety constraint, and a deviation from the paper.

**D13. Termination.** The paper's rule only: |e_z| > 0.25 m or tilt > 0.8. No end-effector termination,
no self-collision term.

**D14. Domain randomization.** (user)
- v1 uses the paper's set as ξ: foot friction, joint calibration offsets, pelvis_2 CoM. That is 1 + 32 + 3
  dims, sampled per episode (`mode="reset"`).
- Kangaroo sys-ID DR (frictionloss, damping, armature) joins ξ before any hardware work.

**D15. Sim settings: 5 ms × 4 substeps, as in the paper.** (user)
- Solver iterations 10, ls_iterations 20.
- The four-bar's solref 0.004 gets clamped at 5 ms. M0 measured that as fine: loop residual 0.17 mm mean,
  3.2 mm max, no NaNs.
- Gains: start from PAL's armature-derived values; changeable.

**D16. Two configs.**
- "paper": the exact hyperparameters and TF32, for a Lambda H100.
- "dev": the local 16 GB GPU.
  - The same nets and batch 4096, but bf16 autocast instead of TF32: 7.7 GB measured vs 12.7 GB.
  - Replay of 500 steps per env, and data subsets.
  - Smaller nets only if memory forces it.
- Every deviation is logged in the run config.

**D17. Baseline: PPO-Sim.** The same env through mjlab's rsl_rl PPO, with BeyondMimic hyperparameters (the
paper's Fig. 4a control).

**D18. G1 control.** The algorithm core is robot-agnostic. If Kangaroo misses a gate, run the same code
on mjlab's G1 with the BONES G1 clips (`kangaroo_bones_seed/data/g1_npz`). That separates implementation
bugs from robot difficulty.

**D19. Sources.** Paper text only. The demo's license forbids reverse engineering, so never read its JS or
weights.

**D20. No extra data filtering.** (user) `final_kept` already applies the paper's name and feasibility
filters on G1, and again on Kangaroo after retargeting.
