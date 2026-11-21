# SuperMimic as a reusable learning method: state and roadmap

Working notes for continuing this refactor in a fresh session. Read this first, then the
code. Branch: `MAC/wbc-supermimic` (pushed). Last updated 2026-11-20, after M1.

## 1. The idea

A colleague implemented SuperMimic (arXiv 2610.28480: motion tracking trained by
backpropagating through a learned, differentiable world model) on branch `omm/wbc`, as one
self-contained motion-tracking task (`tasks/supermimic/`, algorithm under
`tasks/supermimic/rl/`).

Goal of this branch: a **clean reimplementation where SuperMimic is a reusable learning
method**, not something hardcoded to one task. The method lives in `pal_mjlab/rl/supermimic`;
a task only provides its env plus a small, explicit interface. End objective: SuperMimic can
train a different task (WBC on our own motion pipeline, and ultimately **locomotion**, which
is not motion tracking at all).

Rule used so far: **what the env knows by name is configured; what the heads recognize by
function stays recognized by function.**

## 2. Where things stand

| Commit | Content |
|---|---|
| `ef8cad3` | Bugfixes that affected every task: `_joint_names` read a freed temporary `MjSpec` (no pal_mjlab task loaded); `pal_mjlab/__init__` imports `mjlab` first (import cycle through the task registry). |
| `1e8cccb` | **M0**: the colleague's task ported verbatim to `tasks/supermimic/`, running on `rl/supermimic` (algorithm identical to `omm/wbc` but for imports). Action scales built from actuators (the robot's per-joint arm overrides would have given arm 5 a 32x scale). |
| `8d11905` | **M1**: no hardcoded env names. `EnvNamesCfg` (`agent.env_names`) + the runner's `obs_groups`; obs terms mapped by function; `ROOT_STATE_DIM` / `joint_state`; test training with every name changed. |

Verification done:
- `pytest tests/supermimic`: **69 passed** (67 ported + 2 new).
- Training parity vs the colleague's code (same seed/config, 100 iterations, logs in
  `logs/rsl_rl/kangaroo_supermimic/*_ref_theirs` and `*_m1_mine`): same 35 logged metrics;
  iterations 5-20 agree within 0-2%, which is *tighter* than two runs of the colleague's own
  code (GPU training is not bit-reproducible); later drift goes both ways (it 100 return
  0.761 ref vs 0.914 mine, world-model loss 0.282 vs 0.285). Only run-to-run spread up to
  it 20 was measured.

Layout:
```
rl/supermimic/        the method
  supermimic.py         world model / policy / critic updates, imagination (Eqs. 1-3)
  world_model.py        learned dynamics + learned termination head
  heads.py              analytic reward / observation / goal heads   <- most task coupling
  replay.py             replay buffer (stores clip/frame today)
  runner.py             mjlab runner: collection, training loop, checkpoints
  state.py              heading state (floating base) + 2 tracking helpers
  names.py, config.py   EnvNamesCfg, SuperMimicRunnerCfg
  export.py             ONNX export + metadata
utils/math/kinematics.py  TorchKinematics (differentiable FK), rot6d, ...
tasks/supermimic/     the Kangaroo motion-tracking task (owns its names; fine)
```

## 3. What is still task-specific (audit after M1)

Line numbers at `8d11905`, files under `rl/supermimic/`.

1. **Task imports inside the method**: `heads.py:34, 60-64` (lazy, `TODO(M2)`):
   `sm_obs`, `sm_rewards`, `tracking`, `BodyState`; used at `181, 188, 193, 292`.
2. **The reference is a `(clip, frame)` pair everywhere**:
   - replay: `replay.py:37-38, 57-58, 67, 73, 127-129, 172-173`
   - collector: `runner.py:88-89, 272`
   - imagination: `supermimic.py:110-119, 148, 172-189`
   - episode end = clip end: `supermimic.py:89-91, 175-184, 372, 422`
   - heads indexing the library: `heads.py:185-303` (`reference`, `ref_root`, `goal`,
     `critic_input`, `clip_end`, `next_frame`)
3. **`MotionLibraryCommand`-only attributes**:
   - robot state read from the command (`cmd.robot_qpos/qvel`), not the robot entity:
     `runner.py:69-71`
   - `cmd.library`, `cmd.joint_ids` ("library joints" = state = obs = action joints):
     `heads.py:68, 108, 119, 151`, `export.py:71, 83`
   - `cmd.fk` (tracked bodies): `heads.py:132`
   - `cmd.cfg.num_future_frames`, `time_left_cap_s`: `heads.py:69-70`, `export.py:92`
   - return-based start sampler checkpointed via `cmd.sampler`: `runner.py:160, 190`
4. **Closed whitelists of terms (anything else raises)**:
   - obs `heads.py:86-101`: `joint_pos_rel` (biased), `joint_vel_rel`, `builtin_sensor`
     (assumed IMU gyro), task `imu_projected_gravity`, `last_action`; noise uniform
     additive only (`105`)
   - rewards `heads.py:142-152`: task `TRACKING_ERRORS` (exp kernels on body errors),
     `action_rate_l2`, `joint_pos_limits_hard`; `reward_terms` fixed to 3 buckets
     (`275-296`)
   - critic `heads.py:159-160, 261-267`: exactly 5 terms in order (heading state, motion
     goal, context, root pose error (4), time left (1)), hand-assembled
   - context ξ `heads.py:113, 118, 207`: task `sim_context`, fixed layout
     (`ξ[0]` friction, `ξ[1:1+nj]` encoder offsets, then CoM offset)
5. **Termination head defined vs the reference**: `world_model.py:97-102, 136` (two
   features: reference root height and tilt minus the state's), fed by `heads.ref_root`
   (`supermimic.py:119, 322-327`).
6. **Tracking helpers in the method package**: `state.py:98` `root_pose_error`,
   `state.py:117` `reference_goal`; `state.py` docstring speaks of "library joints".
7. **Export**: docstring/contract is a motion goal (`export.py:1-24`); library joints =
   action joints and `goal_num_future_frames` (`71-74, 92`); PD gains via
   `gainprm`/`biasprm` (builtin position actuators, `76, 86-87`); `goal_root_body`
   (`93`).
8. **Robot assumptions (acceptable as the robot contract, should be explicit)**: floating
   base heading state (`state.py`), one IMU site (`heads.py:129-130`, `export.py:70`),
   joint-position actions (`export.py:48`), obs joints = state joints = action joints
   (`heads.py:173`), `auto_reset=False` (`runner.py:201`, a genuine method requirement).
   **Flag**: root height is relative to the env origin, not the terrain (`runner.py:70`):
   on rough terrain the state's height and the height-based termination lose meaning.

Blocking for locomotion: 1-5. Should move to the task: 6-7. Robot contract: 8.

## 4. Roadmap

M0, M1: done. Each next milestone keeps the Kangaroo motion task training identically
(check with the parity protocol in section 6).

### M2: task/method boundary (main milestone)
**M2a, `CommandModel` protocol** (the command's differentiable twin, implemented per task):
- `state(env) -> Tensor`: an opaque command state, stored in replay instead of
  `clip`/`frame`
- `advance(cmd_state)`, `goal(cmd_state, robot_state)`, `done(cmd_state)` (replaces
  clip end), `termination_ref(cmd_state)` (replaces `ref_root`),
  `critic_extras(...)` (replaces root-pose error / time left)
- optional `state_dict` / `load_state_dict` (the sampler)
- implement `MotionLibraryCommandModel` in `tasks/supermimic/`, moving `reference_goal`,
  `root_pose_error`, `ref_root`, time left, the export goal metadata there
- read the robot state from the robot entity, not `cmd.robot_qpos`
- check `torch.compile(dynamic=False)` still works with the opaque state (fixed shape)

M2a progress (uncommitted, 2026-11-21), in dependency order:
- [x] step 0: `rl/supermimic/command_model.py` (protocol, `build_command_model`),
  `SuperMimicRunnerCfg.command_model` (the class itself; `tyro.conf.Suppress`, so not a
  CLI option; `agent.yaml` records it as `!!python/name:...`), `tasks/supermimic/mdp/command_model.py`
  (`MotionLibraryCommandModel`, plus a motion-only `reference(c)` for the tracking
  rewards). The runner builds it but nothing uses it yet. `reference_goal` and
  `root_pose_error` moved from `state.py` to `tasks/supermimic/mdp/tracking.py`.
  `tests/supermimic/test_command_model.py`: model == env and model == heads, bit-exact.
- [x] step 1: robot state from the robot (`SimCollector.state`): raw free-joint + action-joint
  qpos/qvel on the action term's entity (not `robot.data.joint_pos`, which has no free
  joint and includes the passive joints; not `root_link_*`, whose angular velocity is in
  the world frame). Bit-exact vs the command's (`test_command_model.py`).
- [ ] step 2: checkpoint `command_model.state_dict()` instead of `cmd.sampler`
- [ ] step 3: consumers (`supermimic.py` step / imagine / world-model update) call the
  model with `c = {"clip", "frame"}` built at batch unpacking; delete `goal`, `reference`,
  `ref_root`, `critic_input`'s extras, `clip_end`, `next_frame` from `heads.py`
- [ ] step 4: collector records `command_model.state()`; replay stores generic `c` columns
- [ ] step 5: parity run

**M2b, term-twin registry** (replaces the whitelists):
- `register_twin(env_term_func, twin_fn)`; a twin gets one imagined batch (state,
  kinematics, command state, action, prev action, context) plus the term's `params`
- heads walk `cfg.rewards` and the proprio group and look up twins; missing twin raises
  (keep the colleague's safety property)
- mjlab builtin twins in `rl/supermimic`; task twins next to the task's terms
  (`TRACKING_ERRORS`, `joint_pos_limits_hard`, `sim_context` layout)
- `reward_terms` = sum over registered terms, not 3 fixed buckets

**M2c, export**: generic metadata in the method, goal metadata from the `CommandModel`.

Done when `rl/supermimic` imports nothing from `tasks/`, parity holds, and a missing twin
fails with a clear message.

### M3: contract tests
- Twin consistency: each twin vs the env term's value on logged sim states.
- `CommandModel`: `advance`/`done` vs the env command over real rollouts.

### M4: WBC with SuperMimic
- Decide: `CommandModel` for our own `tasks/WBC` `MotionCommand` (recommended: one motion
  pipeline, CSV->NPZ, online loading) vs porting the colleague's command.
- Twins for WBC rewards; check `angular_momentum_penalty`, CoP/CoM are computable from
  the predicted state.
- Register a `Mjlab-WBC-...-SuperMimic` variant (`auto_reset=False`).

### M5: locomotion (final objective)
- `VelocityCommandModel`: constant command over the horizon, `done` never (or timeout),
  `termination_ref` = nominal height + upright.
- Twins for `track_lin_vel_exp`, `track_ang_vel_exp`, orientation, action rate.
- **Open research question**: contact rewards (`feet_air_time`, slip, contact forces) have
  no kinematics-only twin. Options: (1) world model also predicts contacts, (2) learned
  reward head for those terms, (3) drop them for a first proof. Start with (3), then (1).
- Rough terrain: fix the terrain-relative height issue (section 3, item 8).

### Open decisions
- Is the return-based start sampler (Appendix D.1) a method feature (generic reset
  curriculum hook) or task-owned? Current lean: task-owned for now.
- Which `MotionCommand` M4 builds on (see M4).

## 5. Dev environment (Windows laptop, RTX 3060 6 GB): read before running anything

- **CUDA torch is installed by hand into `.venv`** (`torch 2.10.0+cu128`,
  `torchvision 0.25.0`). pyproject has no PyTorch index, so on Windows `uv sync` /
  `uv run` silently reinstall the **CPU** build. Use `.venv/Scripts/python.exe` or
  `.venv/Scripts/train.exe` directly, or `uv run --no-sync`. To restore:
  `uv pip install --python .venv/Scripts/python.exe torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch --reinstall-package torchvision`
- **No Triton on Windows** -> no `torch.compile`: pass `--agent.compile False` and
  `--env.commands.motion.compile-kinematics False`.
- **Set `PYTHONUTF8=1`**, otherwise mjlab's manager tables crash the cp1252 console.
- **VRAM**: `num_rollouts=4096` needs ~12 GB (BPTT through 32 steps); use
  `--agent.algorithm.policy.num-rollouts 512`. `--env.scene.num-envs 1024` is fine.
- **Never run two trainings at once on this GPU**: both stalled (VRAM spills to shared
  memory). ~11 s/iteration alone without compile.
- `test_play_launches` needs symlinks; it falls back to copying on Windows.
- Import-order trap is fixed (`pal_mjlab/__init__` imports mjlab first); keep it that way.

**Motion data (never commit data)**: the real library (`artifacts/kangaroo_grippers_library`,
from the SOMA retargeter) is not available. A stand-in was built from two DCMR LAFAN clips
(jumps1_subject1, sprint1_subject2; Kangaroo-grippers CSVs in
`../DCMR/data/check/batch/`, 30 fps) resampled to 50 Hz:
```
.venv/Scripts/python.exe -I artifacts/tools/dcmr_to_library.py artifacts/kangaroo_grippers_library 30 50 \
  jumps1_subject1=jumps=../DCMR/data/check/batch/jumps1_subject1.csv \
  sprint1_subject2=sprint=../DCMR/data/check/batch/sprint1_subject2.csv
```
`artifacts/` is gitignored; both helper scripts live in `artifacts/tools/` (not in git).

## 6. Parity protocol (run after every milestone)

1. `PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest tests/supermimic -q`
2. Short seeded training (identical flags for every run you compare):
   ```
   PYTHONUTF8=1 MJLAB_WARP_QUIET=1 .venv/Scripts/train.exe Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers \
     --env.scene.num-envs 1024 --env.commands.motion.library-path <abs path to library> \
     --env.commands.motion.compile-kinematics False --agent.compile False \
     --agent.algorithm.policy.num-rollouts 512 --agent.seed 42 --agent.max-iterations 100 \
     --agent.save-interval 100000 --agent.log-interval 5 --agent.run-name <name>
   ```
3. Compare with the M1 run (`logs/rsl_rl/kangaroo_supermimic/2026-11-20_01-33-23_m1_mine`):
   `.venv/Scripts/python.exe -I artifacts/tools/compare_runs.py <run A> <run B>`.
   Expect agreement within a few % up to ~it 20, same curve shapes after (no bit-exactness).
   To reproduce the colleague's reference, check out `omm/wbc` in a worktree and run step 2
   with `PYTHONPATH=<worktree>/src`.
