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

**Status after M2 (2026-11-21):** items 1-7 are resolved (M2a: command state, command
model, robot state from the robot; M2b: single-definition terms, named ξ parts; the
tracking helpers live in the task; M2c: the export). `rl/supermimic` imports nothing from
`tasks/` and reads no task attribute. Left: item 8, the robot contract, now explicit
(state joints = action joints, FK bodies/sites in `ImaginationCfg`, position actuators
for the export); the terrain-relative height flag still stands. The audit below is kept
as it was written.

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
- implement `MotionLibraryCommandModel` in `tasks/supermimic/`, moving `reference_goal`,
  `root_pose_error`, `ref_root`, time left, the export goal metadata there
- read the robot state from the robot entity, not `cmd.robot_qpos`
- check `torch.compile(dynamic=False)` still works with the opaque state (fixed shape)

M2a progress (done 2026-11-21, one commit per step), in dependency order:
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
- [x] step 2: the sampler is the command's (not the method's, not the task's): the
  runner checkpoints every command term with `state_dict`/`load_state_dict` under
  `"commands"` (`pal_mjlab/rl/command_state.py`, strict on resume); old checkpoints'
  `"sampler"` key still loads. `state_dict` left the `CommandModel` protocol (the twin is
  imagination-only). WBC's `OnlineMotionCommand` adaptive sampler is not checkpointed yet
  (left for M4: give it `state_dict`).
- [x] step 3: imagination and the world-model update call the `CommandModel` (`goal`,
  `critic_extras` via `supermimic.critic_input`, `termination_ref`, `done`, `advance`);
  `SuperMimic`, `make_step`, `imagine` take it; `replay_command_state` builds `c` from the
  replay's clip/frame columns (removed in step 4). `heads.py` lost `goal`, `reference`,
  `ref_root`, `critic_input`, `clip_end`, `next_frame`; it keeps the command model only for
  the tracking rewards' `reference(c)` (TODO(M2b)). The critic-layout check is unchanged
  (moves to twins in M2b). `Rollout.clip_end` is now `Rollout.end`.
- [x] step 4: the collector records `command_model.state()` (it no longer holds the
  command term); replay stores c as given (columns allocated from the first add, layout
  checked after) and returns it as `"command"` (per transition in windows, per start
  row in starts); `replay_command_state` is gone. Nothing in `rl/supermimic` reads
  clip/frame; what is left of the motion task there is M2b (heads) and M2c (export).
- [x] step 5: parity run (2026-11-21, `logs/rsl_rl/kangaroo_supermimic/2026-11-21_18-25-30_m2a_mine`,
  the new reference for M2b): same 35 metrics as M1; it 5-20 within 0-2% (return 1.472 vs
  1.438 at it 20); later drift both ways (it 100: WM loss 0.285 vs 0.221, episode length
  29.4 vs 23.2). Every phase ~30% slower, env collection included: GPU shared with
  another app during the run, not the code.

**M2b, single-definition terms** (replace the whitelists in `heads.py`). Decided
2026-11-21, after considering twin dicts (two definitions per term + consistency tests) and
an "imagined env" (running mjlab's own terms on predicted state): each obs/reward term is
defined **once**, as a pure function of one step's tensors, and the env's term is generated
from it, so sim and imagination agree by construction.
- `Batch`: state, pose, FK (configured bodies/sites), `c` + command model, action,
  prev action, ξ. In imagination it is predicted; in the sim it is built from the true
  state (`heading_state(qpos, qvel)`, `command_model.state()`, action manager, ξ term),
  cached per step (invalidated when its inputs change, so post-reset obs are right).
- `@imaginable` turns `fn(b, **params)` into an mjlab term `(env, **params)`; the heads
  call `fn` on imagined batches and raise once, listing every term that is not imaginable.
- Timing matches mjlab: rewards see the step's target `c` (command updated after rewards),
  observations the advanced `c`.
- Config lives in the env (mjlab evaluates obs terms while building the env, before the
  runner): `SuperMimicEnvCfg.imagination = ImaginationCfg(...)` (today's `EnvNamesCfg`
  fields + ξ term + FK bodies/sites). FK over all 49 computable bodies costs ~50% more
  than the 12 tracked + IMU, so the bodies are listed.
- The command model comes from the command term (`cmd.command_model()`), like its
  sampler; the runner cfg's `command_model` field goes.
- mjlab builtins used by SuperMimic tasks get imaginable versions in `rl/supermimic`
  (`joint_pos_rel`, `joint_vel_rel`, `last_action`, IMU gyro, `action_rate_l2`); PPO tasks
  keep mjlab's. Limit made explicit: a term needing what the state lacks (contacts) can't
  be written, exactly as it couldn't be imagined.
- Goal and critic groups go through the same mechanism; `CommandModel` then shrinks to
  `state`/`advance`/`done`/`termination_ref` (+ what the task's own terms call).
- Steps (one commit each, parity kept):
  1. [x] config into the env: `EnvNamesCfg` became `ImaginationCfg`, on
     `SuperMimicEnvCfg.imagination` (`names.py`; CLI `--env.imagination.*`); the runner
     cfg lost `env_names` and `command_model`; `MotionLibraryCommand.command_model()`
     builds the model, `command_model_of(env, name)` fetches and checks it.
     (`--env.imagination.failure-terms X` does not parse, as `--agent.env-names...`
     didn't before: a tyro quirk with that tuple field.)
  2. [x] `rl/supermimic/imagination.py`: `Batch` (state, pose, qpos/qvel, FK, `c` +
     model, last two actions, ξ, `spec`), `BatchSpec` (state joints = action joints,
     `joint_cols`, body/site indices, q0, limits), `BatchSource` (`sim()` cached per env
     step + qpos/qvel/`c`, `imagined(...)`), `@imaginable` / `imagined_fn`.
     `ImaginationCfg` gained `bodies`, `sites`, `context` (the Kangaroo: 12 tracked
     bodies, `imu`, the critic's `sim_context` config). `test_imagination.py`.
  3. [x] rewards are `@imaginable`: `rl/supermimic/terms.py` (`action_rate_l2`), the
     task's 6 tracking terms (robot bodies from the batch FK, reference from
     `command_model.reference(c)`; no `command_name` param: the batch's command) and
     `joint_pos_limits_hard`. The heads walk every weighted reward term, raise once
     listing the non-imaginable ones, and return `reward_terms(b)` per env term. Env
     rewards vs the old formulas: max 1.1e-6 (local vs world frame rounding).
     `Batch.memo` caches what several terms derive (the motion's robot/reference
     bodies), and `anchor_failure` reads the sim batch too: without both, a sim step ran
     FK 4x instead of 2x (446 vs 158 ms/step at 1024 envs); now 157 vs 151 ms.
  4. [x] proprio obs are `@imaginable` (`terms.py`: `joint_pos_rel` (biased via ξ's
     `encoder_bias`), `joint_vel_rel`, `last_action`, `site_ang_vel`,
     `site_projected_gravity`); the IMU obs are the IMU site's FK (a predicted state has
     no sensors), env obs vs the old sensor-based ones: max 4.8e-6. ξ declares named
     parts (`sim_context.context_layout` -> `ContextPart`s). Imagined obs see the
     advanced `c`, as the env's; replay starts also give the history rows' command
     states and previous actions. A joint term on joints outside the state now fails
     when the env is built. Cost: one more FK per reset (the reset envs' IMU obs), ~35
     ms per sim step here (FK is launch-bound without `torch.compile`: ~31 ms per call
     whatever the rows), ~1% of a training iteration.
  5. [x] goal and critic groups are `@imaginable` (`terms.heading_state`,
     `terms.context`, the task's `motion_goal`, `motion_root_pose_error`,
     `motion_time_left`); the heads concatenate each group's terms in the env's order
     (`goal(b)`, `critic_input(b)`), and refuse noise/history outside proprioception.
     `CommandModel` is now `state`/`advance`/`done`/`termination_ref`; the motion model
     keeps `goal(c, qpos)`, `root_pose_error(c, qpos)`, `time_left(c)` for its terms.
     Imagination passes the batch along (the step's observation batch is the next
     step's goal/critic batch: still one FK per imagined step). The ξ term is its own
     config (`ImaginationCfg.context`), read by the critic through `terms.context`.
     Env goal/critic obs vs the old ones: 0 / 2.4e-7. `rl/supermimic` imports nothing
     from `tasks/`.
  6. clean-up: `MotionLibraryCommand.goal` / `at_clip_end` call the command model (one
     definition each); its unused `heading_state`, `root_pose_error`, `clip_time_left`
     and the heads' `kinematics` are gone; docstrings.
     Parity (2026-11-21, GPU not shared): `2026-11-21_22-00-42_m2b_mine` and its rerun
     `2026-11-21_22-16-46_m2b_mine_rerun` match M1/M2a within 0-2% to it 20 (protocol
     passed). After it 50 both sit on the worse side of M1/M2a (it 100: actor grad
     norm 27.8 / 10.9 vs 1.1 / 1.4, WM loss 0.39 / 0.36 vs 0.29 / 0.22); run #1
     collapsed (return -0.49), the rerun did not (0.67): not deterministic, so likely
     run-to-run spread, but 2 vs 2 runs cannot exclude a small real change. Not the IMU:
     the FK site frame matches the XML's (`imu` site quat composed with its body; gyro and
     framequat on that site), and FK vs the sensors differ by <= 4.8e-6 (mjlab runs
     forward() before observations, so the sensors are not stale); every sim value
     changed by M2b is within 5e-6 of before. An extra M2a sample (the `b8df475` code,
     `2026-11-21_22-38-08_m2a_mine_rerun`) went through the same rough patch at it
     70-90 (return ~0.5, critic loss 3.6, grad norm 11-14, WM loss 0.35) and recovered
     by it 100 (return 0.89): the late spread is the config's, not M2b's. Only M2b #1's
     collapse has no M2a counterpart (1 of 2 vs 0 of 3 with M1). **To do
     before long trainings:** compare several seeds over more iterations on Linux
     (compile, num_rollouts 4096).

**M2c, export** [x] (2026-11-21): `export.py` reads the robot and joints from the action
term (`BatchSpec`: `joint_names`, `q0`), refuses non-position actuators instead of exporting
wrong PD gains, and lists each observation term's site (`observation_sites`, aligned with
`observation_names`; `imu_site` kept when there is one). Goal metadata comes from the
command term's optional `export_metadata()` (`MotionLibraryCommand`:
`goal_num_future_frames`, `goal_root_body`, and the goal's layout in its docstring); a
key clash with the generic metadata raises. The Kangaroo's metadata is identical to
before, plus `observation_sites`.

Done when `rl/supermimic` imports nothing from `tasks/`, parity holds, and a missing twin
fails with a clear message.

### M3: contract tests [x] (2026-11-21)
Reframed after M2b (no twins: the env and imagination run the same functions; what can
differ is their inputs). `rl/supermimic/contracts.py`: `check_contract(env, steps,
end_terms)` rolls the env out with random actions (ends and resets included) and reports
the max error per check: each weighted reward term and each observation term on the
simulator state rebuilt as a prediction, with the command state the env used (targeted
for rewards, advanced for observations), and the command model's `advance` and `done`
vs the env (exact `done` with the given episode-end terms). Terms without a single
definition are listed as missing. A new task's test just calls it
(`test_contracts.py`). Kangaroo, 80 steps / 8 ends: command exact, obs <= 3.3e-6,
rewards <= 3.8e-6.

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
- ~~Return-based start sampler: method or task?~~ Decided (2026-11-21): owned by the
  command term, checkpointed per command term. Later, possibly one binned start sampler
  with a pluggable score shared by `MotionLibraryCommand` (returns) and WBC's
  `OnlineMotionCommand` (failure counts); decide with M4.
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
3. Compare with the latest reference run (M2a: `logs/rsl_rl/kangaroo_supermimic/2026-11-21_18-25-30_m2a_mine`; M1: `2026-11-20_01-33-23_m1_mine`):
   `.venv/Scripts/python.exe -I artifacts/tools/compare_runs.py <run A> <run B>`.
   Expect agreement within a few % up to ~it 20, same curve shapes after (no bit-exactness).
   To reproduce the colleague's reference, check out `omm/wbc` in a worktree and run step 2
   with `PYTHONPATH=<worktree>/src`.
