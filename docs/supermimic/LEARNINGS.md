# SuperMimic on Kangaroo: learnings

Facts and measurements, newest at the bottom of each section. Decisions that follow from them are in
DECISIONS.md (Dn).

## Paper (arXiv 2610.28480, read in full)

**No code is released.** The browser demo runs the world model (WM), but its license forbids reverse
engineering, so we implement from the text only (D19).

**What the paper pins down.**
- The loop (Alg. A.1):
  - collect with action noise N(0, 0.1²) into a FIFO replay of 1000 steps per env;
  - fit the WM on k=8 autoregressive windows (Eq. 1);
  - BPTT through the frozen WM over H=32 steps from replay states and their ξ, with survival weight
    c_{t+1} = c_t(1 − d̂);
  - critic on λ-returns (λ 0.95, γ 0.99, 5 epochs × 4 minibatches, no target net).
- Networks:
  - WM MLP 2048×3;
  - encoder 2048-1024-512-256 on K=10 future frames;
  - decoder 2048-1024-512-256 on (z, 10 obs);
  - z 32-d with tanh;
  - critic 256-128.
- Optimization:
  - Adam, lr 2e-4, with β = (0.9, 0.95) for the policy;
  - gradient clipping: global norm 1, then element-wise 0.1;
  - cosine decay to 0.03× over 250k iterations, 320k in total;
  - 4096 envs, 4096 imagined rollouts, "10 batches of 2048 windows", UTD 0.4.
- Observations: q−q0, q̇, IMU angular velocity, projected gravity, a_{t−1}, with a 10-frame history and no
  base linear velocity.
- Reward: BeyondMimic's, minus self-collision, with the joint limit at −2 and every term × dt.
- Termination: |e_z| > 0.25 m or tilt > 0.8.
- DR: friction, ±0.01 rad calibration, torso CoM. No pushes, because the deterministic WM can't model them.
- Sim: 5 ms × 4.

**Left open, decided by us (D4–D10):** the WM state layout and normalization, deltas vs absolute
prediction, which heads are learned, the goal-frame content, clip-end handling, and how to read UTD and
"10 batches".

**Compute sanity check.** Paper-size nets cost about 23 TFLOP per iteration (BPTT 4096×32 plus a 10×2048×8
WM fit). At ~200 TFLOPS TF32 that is ~9 it/s on an H100, consistent with 320k iterations in ~9.8 h. With
UTD 0.4 = 2.5 vector steps per iteration, it implies ~93k sim steps/s per GPU.

## Robot and data

- **The BONES retarget matches `kangaroo_grippers.xml`.** nq 39 / nv 38, the same 32 joints in the same
  order: 28 actuated, plus 4 passive (femur/knee) closed by `<connect>`.
- **Arm axis (D3).** With arm 5/7 at +z, FK equals the GMR model to 0.0 m on real clips
  (`tests/supermimic/test_kinematics.py`, fixture `bones_3clips.npz`; the test fails with the old −z axis,
  0.165 m off).
- **Joint ranges.** In 800 random clips (323k frames), 0% of actuated-joint frames are outside the hard
  range (passive femur 0.05%). But many are outside the 0.9× soft limits: arm_7 27%, pelvis_1 20%,
  leg_1/2 10–15%, arm_3 8–11%, ankles 9%. So joint-limit penalties use the hard limits (D12).
- **Kinematic structure.** The femur/knee branch is a sibling under `leg_*_3_link`. The tracked bodies sit
  on the serial chain through the `leg_*_length_joint` slide, so their FK doesn't need the passive joints.
  The IMU is on `pelvis_2_link`, after the two waist joints; the anchor is `base_link`.
- **XML quirks.**
  - The PAL XMLs exclude contacts with the scene's `terrain` body, so they only compile inside an mjlab
    scene.
  - `kangaroo_grippers.xml` has two `<exclude>`s naming geoms (`leg_*_ankle_bar_collision`) as bodies. The
    mjlab scene still compiles.
  - For standalone kinematics, `compile_kinematic_model()` drops all excludes.
- **Clip lengths.** Median 5.7 s, p95 15.7 s, max 180 s; 1,499 clips are under 2 s. The manifest's
  `g1_n_frames` equals the Kangaroo clip length (checked on 300 clips).
- **Packed library.** Converting from the pickles is fast: `smoke` (141 clips, 0.25 h) and `dev`
  (1,296 clips, 2.56 h, stratified by category) each build in about 2 s.

## WBC pipeline (MAC/WBC): what carries over

- **Reuse:**
  - the multi-clip `MotionLoader` segment logic and the per-clip 1 s bins (`_build_adaptive_bins`);
  - the Kangaroo actuator config and action scales;
  - later, for hardware: the convex-hull half-spaces (`velocity/mdp/rewards.py:71`, differentiable).
- **Don't reuse:**
  - the per-body library layout (~1 KB/frame, so 44 GB for the full set; D11);
  - the ONNX export that embeds the library;
  - the failure-count sampling (the paper's is return-based);
  - the sparse WBC actor obs (D2).

## mjlab 1.6.0 internals that matter

- **Step order:** actions → decimation physics → terminations → rewards → interval events → auto-reset →
  forward → command update → observations. Rewards see body poses one substep stale.
- **Auto-reset loses the terminal state.** Use `auto_reset=False` in our runner (D9).
- **The WBC `MotionCommand` teleports to a new clip at clip end** with no done flag. That is non-physical
  for a world model (D8).
- **ξ readback works.**
  - Friction: `sim.model.geom_friction[:, ids, 0]`.
  - CoM: `sim.model.body_ipos − sim.get_default_field("body_ipos")`.
  - Calibration: `robot.data.encoder_bias`, which is applied to both obs and actions.
- **Raw `sim.data` arrays are zero-copy views** reused by CUDA graphs, so clone before storing.
- **A custom runner plugs in via `register_mjlab_task(..., runner_cls=...)`.** rsl_rl's rollouts run under
  `inference_mode`, so they are not usable for BPTT.
- **No world-model code exists anywhere**, and mujoco_warp has `enable_backward=False`.
- **Import order.** Importing `pal_mjlab.robots` before `mjlab` triggers a circular import while mjlab
  registers tasks (only a warning, but the tasks don't register). Import `mjlab` first in scripts.
- **Dependencies.** Running `uv add` or a plain `uv sync` without `--extra cu128 --group supermimic` drops
  CUDA torch and pyarrow.

## M0 measurements (RTX 5060 Ti 16 GB, shared; 2026-10-22)

**Sim** (`m0/bench_sim.json`). Raw MuJoCo Warp, Kangaroo grippers, 4096 envs replaying `smoke` clips with
the PD and a pinned root. This excludes env-manager overhead.

| setting | envs | control steps/s | ms / vector step | loop residual mean / max | NaN envs |
|---|---|---|---|---|---|
| paper 5 ms × 4 | 1024 | 137k | 7.5 | 0.17 / 3.2 mm | 0 |
| paper 5 ms × 4 | 2048 | 178k | 11.5 | 0.17 / 3.2 mm | 0 |
| paper 5 ms × 4 | 4096 | 201k | 20.3 | 0.17 / 3.2 mm | 0 |
| WBC 2 ms × 10 | 4096 | 84k | 48.6 | 0.015 / 0.39 mm | 0 |

- 5 ms × 4 is 2.4× faster than 2 ms × 10.
- The clamped four-bar solref is fine: a 3 mm worst-case loop error is below the retarget's own loop
  residual (up to 11 mm).
- Sim memory is about 1.5–3 GB at 4096 envs (measured by device free-memory delta, so approximate).
- Paper collection costs 2.5 vector steps per iteration, i.e. about 50 ms of raw physics per iteration at
  4096 envs.

**Learning updates** (`m0/bench_bptt_*.json`). One iteration's learning work with random data and the
planned shapes:
- policy update: BPTT, H=32, through a frozen WM, with the FK reward (14 bodies + IMU) every step and a
  critic bootstrap;
- WM fit: 10 optimizer steps × 2048 windows × k=8;
- critic fit: 5 × 4 minibatches.

| nets | precision | batch | FK | policy update | WM fit | critic | peak mem |
|---|---|---|---|---|---|---|---|
| paper | fp32 | 4096 | eager | 5.38 s | 0.73 s | 0.09 s | 13.1 GB |
| paper | bf16 | 4096 | eager | 3.14 s | 0.27 s | 0.05 s | 8.0 GB |
| dev (½ width) | bf16 | 4096 | eager | 3.17 s | 0.09 s | 0.05 s | 4.5 GB |
| paper | bf16 | 4096 | none | 0.34 s | 0.27 s | 0.05 s | 7.3 GB |
| paper | bf16 | 4096 | `torch.compile` | **0.41 s** | 0.27 s | 0.05 s | **7.7 GB** |
| paper | bf16 | 2048 | `torch.compile` | 0.37 s | 0.27 s | 0.03 s | 4.3 GB |
| paper | tf32 | 4096 | `torch.compile` | 0.67 s | 0.49 s | 0.07 s | 12.7 GB |

- **Eager FK is launch-bound.** It costs 3–5 s per update whatever the batch (1k–4k) or network size. With
  `torch.compile` it adds only ~20% over no FK, so the analytic reward and obs heads (D5) are affordable,
  as long as the FK (or the whole imagination step) is compiled.
- **The paper's network sizes and batch fit locally in bf16 (7.7 GB).** TF32, the paper's precision, needs
  12.7 GB, which doesn't fit next to the sim (~2–3 GB), the replay and the resident services on a 16 GB
  card. So locally we run the paper nets in bf16 (D16).
- **Local iteration estimate: ~0.8 s.** That is 0.74 s of learning plus ~0.05 s of raw physics for
  2.5 vector steps, before env-manager overhead (to measure in M1). That gives ~4.5k it/h, so the paper's
  320k iterations would take ~70 h here versus ~10 h on an H100 (~8×).
- **Conclusion:** dev-subset studies (M1–M4) run locally; full-scale runs (M5) go to Lambda.

The compiled-FK rows above were measured before the `torch.compile` fix below. The buggy graph skipped
the root joint, about one body out of ~40, so the timings stand.

## M1: SuperMimic env + PPO-Sim (RTX 5060 Ti, `dev` library, 8192 envs)

| setting | ms / env step | device memory |
|---|---|---|
| paper DR every episode (CoM at reset), `nconmax` 100 | 235 | 10.8 GB |
| CoM offset at startup, `nconmax` 100 | 72 | 10.8 GB |
| CoM offset at startup, `nconmax` 40 | 72 | 5.2 GB |
| FK not compiled (CoM at reset) | 283 | 10.8 GB |

- **CoM DR at reset was the bottleneck.** `dr.body_com_offset` triggers `set_const` over all worlds
  whenever any env resets, which with 8192 envs is every step. Hence CoM at startup (D14).
- **Memory scales with `nconmax`, not `njmax`.** mujoco_warp's contact and CCD scratch buffers scale with
  the contact pool (nconmax × worlds). About 15 contacts per world are in use. `njmax` is a hard
  per-world cap that fallen robots push to about 280, so it stays at 300 (D23).
- **PPO-Sim at paper size:** 4.98 s per iteration (2.0 s collection of 196k steps, 2.9 s learning), about
  720 iterations per hour. Peak device memory is 14.4 GB of 16.3, which fits only while nav-infer is
  stopped.
- **Bug found and fixed: `torch.compile` with mujoco enums.** Inside a compiled function, `int_kind ==
  mujoco.mjtJoint.mjJNT_FREE` (a pybind11 enum) evaluated False. The compiled FK silently skipped the free
  joint and put every robot at the default root pose, so falls were never detected and the reward
  compared the reference with itself. Fixed by comparing plain ints and building constants once in
  `__init__`, plus a compiled-vs-eager test. It's the same family as GMR's int-vs-`mjtEq` set lookup bug.
- **Tests must step the robot off the reference.** The first FK-parity test ran at reset, where robot and
  reference coincide, so it couldn't catch that bug. The env tests now step random actions and check
  that falls terminate.
- **Name clash:** `CommandTerm.time_left` is mjlab's resampling timer; don't shadow it.
- **Robot specs are prefixed in the scene.** Entity specs attached to the scene have names prefixed
  `<entity>/`, so the FK model built from `robot.spec.copy()` uses prefixed body names.

## Audit of inherited assumptions (2026-10-22)

**Kangaroo geometry.**
- The serial leg is hip, then a slide, then the ankle: `leg_*_length_link`, `leg_*_4_link` and
  `leg_*_5_link` share one point, the ankle.
- The physical knee exists only in the passive four-bar (`leg_*_knee_link`).
- `arm_*_tip_link` coincides with `arm_*_7_link` in pal_mjlab, but GMR's model rotates it 45° in yaw.

**The retarget's IK table** (`smplx/g1_to_kangaroo_7dof.json`).
- It maps G1's knee to `leg_*_4_link` and the wrist to `arm_*_tip_link`, with rotation weights only.
  Position weights are 0, except for the feet.
- WBC's tracked bodies were copied from this table, which is how the ankle ended up tracked twice. See
  D25.

**Gains.**
- MAC/WBC lowered the arm and waist natural frequency to 1 Hz in experimental commits for its own task.
- PAL main and the paper's method (BeyondMimic) use one frequency for all joints. See D26.

**Goal and DR.**
- Reference root velocity and height in the goal came from WBC's sparse interface. BeyondMimic's
  command, which the paper uses, has only reference q and q̇ plus root orientation. See D6.
- The calibration-offset DR defaulted to all 32 joints, including the four passive ones, which have no
  encoder. It is now restricted to the 28 actuated joints. See D14.

**Check after the changes** (8192 envs): 76 ms per step, 6.4 GB, finite observations.

## Repo state found on MAC/WBC (not caused by us)

- `tests/test_kangaroo_velocity_task_launch.py`: all 12 velocity tasks fail at env construction on GPU.
  The new `joint_torques_magnitude` metric (`velocity/mdp/metrics.py:28`, commits dea1381/ea69429) maps
  every joint to an actuator, including the passive femur/knee. This is unrelated to SuperMimic and left
  as is.
- These launch tests need CUDA; on CPU they fail earlier with "no CUDA-capable device". The other 46 tests
  pass on CPU.
