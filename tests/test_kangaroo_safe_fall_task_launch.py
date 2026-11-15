"""Smoke and integration tests for the Kangaroo safe-fall task."""

import math

import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.scripts.train import TrainConfig, run_train
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.viewer.debug_visualizer import NullDebugVisualizer
from pal_mjlab.tasks.safe_fall.mdp.events import (
  FAMILY_MOVING,
  FAMILY_STANDING,
  STATE_JOINT_DOWN,
  STATE_TRIPPED,
  STATE_WALKING,
)
from pal_mjlab.tasks.safe_fall.mdp.faults import (
  get_actuator_faults,
  replace_actuator_fault_cfg,
)

TASK = "Mjlab-SafeFall-Flat-Pal-Kangaroo"
NUM_ENVS = 2
NUM_STEPS = 4

# The actor layout of the Kangaroo walking policies this task warm-starts from.
WALKER_ACTOR_OBS_DIM = 83
WALKER_ACTION_DIM = 22
# The task appends one flag per actuator: whether it is down.
ACTOR_OBS_DIM = WALKER_ACTOR_OBS_DIM + WALKER_ACTION_DIM


class StepOnlyViewer:
  """Viewer stub that steps the policy instead of opening a window."""

  def __init__(self, env, policy, **kwargs):
    del kwargs
    self.env = env
    self.policy = policy

  def run(self) -> None:
    obs = self.env.get_observations()
    for _ in range(NUM_STEPS):
      with torch.inference_mode():
        actions = self.policy(obs)
      obs = self.env.step(actions)[0]


class _RecordingVisualizer(NullDebugVisualizer):
  """Keeps the marks a term draws for one environment."""

  def __init__(self, env_idx: int):
    super().__init__(env_idx=env_idx)
    self.spheres: list[np.ndarray] = []
    self.cylinders: list[tuple[np.ndarray, np.ndarray]] = []

  def add_sphere(self, center, radius, color, label=None) -> None:
    self.spheres.append(np.asarray(center))

  def add_cylinder(self, start, end, radius, color, label=None) -> None:
    self.cylinders.append((np.asarray(start), np.asarray(end)))


def _train_cfg(monkeypatch) -> TrainConfig:
  device = get_test_device()
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0" if device == "cuda" else "")
  cfg = TrainConfig.from_task(TASK)
  cfg.env.scene.num_envs = NUM_ENVS
  cfg.agent.logger = "tensorboard"
  cfg.agent.num_steps_per_env = NUM_STEPS
  cfg.agent.algorithm.num_mini_batches = 1
  cfg.agent.max_iterations = 1
  return cfg


def _make_env(num_envs: int, **reset_params) -> ManagerBasedRlEnv:
  """Environment with healthy actuators and the given reset parameters."""
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = num_envs
  cfg.curriculum = {}
  replace_actuator_fault_cfg(cfg, count_probabilities=(1.0,))
  cfg.events["reset_fall_start"].params.update(reset_params)
  return ManagerBasedRlEnv(cfg=cfg, device=get_test_device())


def test_one_task_is_registered():
  registered = {t for t in list_tasks() if t.startswith("Mjlab-SafeFall-")}
  assert registered == {TASK}


def test_missing_state_buffer_falls_back_to_standing_starts(tmp_path):
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = NUM_ENVS
  cfg.curriculum = {}
  cfg.events["reset_fall_start"].params["state_buffer_path"] = str(
    tmp_path / "missing.npz"
  )
  with pytest.warns(UserWarning, match="record_walker_states"):
    env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    reset_term = env.event_manager.get_term_cfg("reset_fall_start").func
    assert torch.all(reset_term.family == FAMILY_STANDING)
  finally:
    env.close()


def test_train_launches(tmp_path, monkeypatch):
  log_dir = tmp_path / "train"
  run_train(TASK, _train_cfg(monkeypatch), log_dir)
  assert (log_dir / "model_0.pt").exists()


def test_play_launches(monkeypatch):
  monkeypatch.setattr("mjlab.scripts.play.NativeMujocoViewer", StepOnlyViewer)
  run_play(
    TASK,
    PlayConfig(
      agent="zero", num_envs=NUM_ENVS, device=get_test_device(), viewer="native"
    ),
  )


def test_warm_start_loads_the_actor_and_keeps_it_frozen(tmp_path, monkeypatch):
  source_dir = tmp_path / "source"
  run_train(TASK, _train_cfg(monkeypatch), source_dir)
  source = torch.load(source_dir / "model_0.pt", map_location="cpu", weights_only=False)

  cfg = _train_cfg(monkeypatch)
  cfg.agent.init_actor_checkpoint = str(source_dir / "model_0.pt")
  cfg.agent.init_action_std = 0.3
  cfg.agent.actor_freeze_iterations = 1
  warm_dir = tmp_path / "warm"
  run_train(TASK, cfg, warm_dir)
  warm = torch.load(warm_dir / "model_0.pt", map_location="cpu", weights_only=False)

  source_actor, warm_actor = source["actor_state_dict"], warm["actor_state_dict"]
  assert source_actor["mlp.0.weight"].shape == (512, ACTOR_OBS_DIM)
  assert source_actor["mlp.6.weight"].shape[0] == WALKER_ACTION_DIM
  # The single iteration was a critic-only one: the network is the source's.
  for key in source_actor:
    if key.startswith("mlp."):
      assert torch.equal(source_actor[key], warm_actor[key]), key
  # Exploration restarts from the requested std.
  std = warm_actor["distribution.std_param"]
  assert torch.allclose(std, torch.full_like(std, 0.3))
  # The critic was not loaded and did learn.
  assert not torch.equal(
    source["critic_state_dict"]["mlp.0.weight"],
    warm["critic_state_dict"]["mlp.0.weight"],
  )


def test_reset_starts_standing_on_the_ground_with_a_kick():
  env = _make_env(
    64,
    state_buffer_path=None,
    min_kick_speed=1.0,
    max_kick_speed=2.0,
    max_kick_ang_speed=0.0,
    tip_fraction=0.0,
  )
  try:
    env.reset()
    robot = env.scene["robot"]
    reset_term = env.event_manager.get_term_cfg("reset_fall_start").func

    assert torch.all(reset_term.family == FAMILY_STANDING)
    assert torch.all(reset_term.tip_angle == 0.0)
    assert torch.all((reset_term.kick_speed >= 1.0) & (reset_term.kick_speed <= 2.0))
    velocity = robot.data.root_link_lin_vel_w
    assert torch.allclose(
      torch.linalg.vector_norm(velocity[:, :2], dim=-1),
      reset_term.kick_speed,
      atol=1e-3,
    )
    assert torch.allclose(velocity[:, 2], torch.zeros_like(velocity[:, 2]), atol=1e-3)

    # Soles within a couple of centimetres of the ground: no drop at reset.
    height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    assert torch.all((height > 0.9) & (height < 0.92))

    # Same actor layout as the walker, with its velocity command held at zero.
    obs = env.observation_manager.compute()
    assert obs["actor"].shape == (64, ACTOR_OBS_DIM)
    assert torch.all(env.command_manager.get_command("twist") == 0.0)
  finally:
    env.close()


def test_tipped_starts_rest_on_the_edge_of_the_feet():
  """A tipped start is tilted by the sampled angle, turning, and on the ground."""
  angle_range = (math.radians(20.0), math.radians(50.0))
  env = _make_env(
    64,
    state_buffer_path=None,
    max_kick_speed=0.0,
    max_kick_ang_speed=0.0,
    joint_pos_noise=0.0,
    tip_fraction=1.0,
    tip_angle_range=angle_range,
    tip_rate_range=(1.0, 1.0),
  )
  try:
    env.reset()
    robot = env.scene["robot"]
    reset_term = env.event_manager.get_term_cfg("reset_fall_start").func
    env_ids = torch.arange(env.num_envs, device=env.device)

    angle = reset_term.tip_angle
    assert torch.all((angle >= angle_range[0]) & (angle <= angle_range[1]))
    tilt = torch.acos(-robot.data.projected_gravity_b[:, 2])
    assert torch.allclose(tilt, angle, atol=1e-3)

    # The lowest point of the feet is just above the ground: no drop and no
    # penetration, whatever the tip direction.
    lowest = reset_term._lowest_ground_geom_height(env, env_ids)
    lowest = lowest - env.scene.env_origins[:, 2]
    assert torch.all((lowest > 0.0) & (lowest < 0.01))

    # Turning at 1 rad/s about a horizontal axis, the base moving with it.
    ang_vel = robot.data.root_link_ang_vel_w
    assert torch.allclose(
      torch.linalg.vector_norm(ang_vel, dim=-1), torch.ones_like(angle), atol=1e-3
    )
    assert torch.allclose(ang_vel[:, 2], torch.zeros_like(angle), atol=1e-3)
    assert torch.all(robot.data.root_link_lin_vel_w[:, 2] < 0.0)

    # With nothing done about it, every one of them ends up on the ground.
    action = torch.zeros(
      env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    for _ in range(100):
      env.step(action)
    height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    assert torch.all(height < 0.6)
  finally:
    env.close()


def test_reset_samples_moving_starts_from_a_state_buffer(tmp_path):
  probe = _make_env(NUM_ENVS)
  robot = probe.scene["robot"]
  joint_names = np.array(robot.joint_names)
  default_joint_pos = robot.data.default_joint_pos[0].cpu().numpy()
  probe.close()

  # Two recognizable states: default pose moving forward at 0.7 and 1.1 m/s.
  num_states = 2
  root_vel = np.zeros((num_states, 6), dtype=np.float32)
  root_vel[:, 0] = (0.7, 1.1)
  path = tmp_path / "states.npz"
  np.savez(
    path,
    joint_names=joint_names,
    root_height=np.full(num_states, 0.93, dtype=np.float32),
    root_quat=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (num_states, 1)),
    root_vel=root_vel,
    joint_pos=np.tile(default_joint_pos, (num_states, 1)),
    joint_vel=np.zeros((num_states, len(joint_names)), dtype=np.float32),
  )

  env = _make_env(
    64,
    state_buffer_path=str(path),
    moving_fraction=1.0,
    max_kick_speed=0.0,
    max_kick_ang_speed=0.0,
    height_range=(0.0, 0.0),
    tip_fraction=0.0,
  )
  try:
    env.reset()
    robot = env.scene["robot"]
    reset_term = env.event_manager.get_term_cfg("reset_fall_start").func
    assert torch.all(reset_term.family == FAMILY_MOVING)

    height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    assert torch.allclose(height, torch.full_like(height, 0.93), atol=1e-4)
    # The recorded speed survives the random heading applied at reset.
    speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
    is_slow = torch.isclose(speed, torch.full_like(speed, 0.7), atol=1e-3)
    is_fast = torch.isclose(speed, torch.full_like(speed, 1.1), atol=1e-3)
    assert torch.all(is_slow | is_fast)
    assert is_slow.any() and is_fast.any()
  finally:
    env.close()


def test_kneeling_is_the_best_paid_way_to_rest():
  fractions = load_env_cfg(TASK).rewards["safe_state"].params["contact_rest_fractions"]
  on_arms = fractions[r"arm_(left|right)_[1-4]_link"]
  on_torso = fractions[r"pelvis_[12]_link"]
  on_pelvis = fractions[r"(base_link|leg_(left|right)_[123]_link)"]
  # Nothing but knees and feet on the ground pays 1.
  assert 1.0 > on_arms > on_torso > on_pelvis > 0.0


def test_gentle_kicks_stay_below_their_maximum():
  env = _make_env(
    512,
    state_buffer_path=None,
    min_kick_speed=0.0,
    max_kick_speed=10.0,
    gentle_fraction=0.5,
    gentle_max_kick_speed=2.5,
  )
  try:
    env.reset()
    reset_term = env.event_manager.get_term_cfg("reset_fall_start").func
    # Half of the kicks below 2.5 m/s, plus a quarter of the other half.
    gentle = (reset_term.kick_speed <= 2.5).float().mean()
    assert 0.55 < gentle < 0.70
    assert reset_term.kick_speed.max() > 8.0
  finally:
    env.close()


def test_forward_falls_point_where_the_robot_faces():
  half_angle = math.radians(30.0)
  env = _make_env(
    256,
    state_buffer_path=None,
    min_kick_speed=3.0,
    max_kick_speed=3.0,
    max_kick_ang_speed=0.0,
    tip_fraction=0.0,
    forward_fraction=1.0,
    forward_half_angle=half_angle,
  )
  try:
    robot = env.scene["robot"]
    params = env.event_manager.get_term_cfg("reset_fall_start").params

    def kick_angle() -> torch.Tensor:
      """Direction of the start velocity in the robot's own frame, 0 = ahead."""
      velocity = quat_apply_inverse(
        robot.data.root_link_quat_w, robot.data.root_link_lin_vel_w
      )
      return torch.atan2(velocity[:, 1], velocity[:, 0]).abs()

    env.reset()
    angle = kick_angle()
    assert torch.all(angle <= half_angle + 1e-3)
    assert angle.max() > 0.5 * half_angle

    # Without it the kicks go in every direction.
    params["forward_fraction"] = 0.0
    env.reset()
    ahead = (kick_angle() <= half_angle).float().mean()
    assert 0.05 < ahead < 0.35
    assert kick_angle().max() > math.radians(150.0)

    # A forward fall that starts tipped is tipped forwards as well.
    params.update(
      forward_fraction=1.0,
      tip_fraction=1.0,
      tip_angle_range=(math.radians(20.0), math.radians(40.0)),
      max_kick_speed=0.0,
      min_kick_speed=0.0,
    )
    env.reset()
    gravity = robot.data.projected_gravity_b
    assert torch.all(
      gravity[:, 0] > math.sin(math.radians(20.0)) * math.cos(half_angle) - 0.02
    )
  finally:
    env.close()


def test_moving_starts_use_the_walker_states_of_what_went_wrong(tmp_path):
  probe = _make_env(NUM_ENVS)
  robot = probe.scene["robot"]
  joint_names = np.array(robot.joint_names)
  default_joint_pos = robot.data.default_joint_pos[0].cpu().numpy()
  actuator_names = list(probe.action_manager.get_term("joint_pos").target_names)
  probe.close()

  # One state per kind and failed actuator, told apart by its base height.
  num_actuators = len(actuator_names)
  kind = np.array([STATE_WALKING, STATE_TRIPPED] + [STATE_JOINT_DOWN] * num_actuators)
  fault_joint = np.array([-1, -1, *range(num_actuators)])
  height = np.array(
    [0.93, 0.94] + [0.95 + 0.001 * j for j in range(num_actuators)], dtype=np.float32
  )
  num_states = len(kind)
  root_vel = np.zeros((num_states, 6), dtype=np.float32)
  root_vel[:, 0] = 1.0
  path = tmp_path / "states.npz"
  np.savez(
    path,
    joint_names=joint_names,
    fault_joint_names=np.array(actuator_names),
    kind=kind,
    fault_joint=fault_joint,
    root_height=height,
    root_quat=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (num_states, 1)),
    root_vel=root_vel,
    joint_pos=np.tile(default_joint_pos, (num_states, 1)),
    joint_vel=np.zeros((num_states, len(joint_names)), dtype=np.float32),
  )

  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 512
  cfg.curriculum = {}
  replace_actuator_fault_cfg(cfg, count_probabilities=(0.5, 0.3, 0.2))
  cfg.events["reset_fall_start"].params.update(
    state_buffer_path=str(path),
    moving_fraction=1.0,
    fault_state_fraction=1.0,
    trip_state_fraction=0.5,
    as_recorded_fraction=0.5,
    as_recorded_max_kick_speed=0.0,
    min_kick_speed=3.0,
    max_kick_speed=3.0,
    max_kick_ang_speed=0.0,
    height_range=(0.0, 0.0),
    tip_fraction=1.0,
    forward_fraction=0.0,
    gentle_fraction=0.0,
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    robot = env.scene["robot"]
    reset_term = env.event_manager.get_term_cfg("reset_fall_start").func
    faults = get_actuator_faults(env, cfg.events["actuator_faults"].params["fault_cfg"])
    first = faults.first_failed
    state_kind = reset_term.state_kind
    height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]

    # An episode with an actuator down starts from a state recorded with that
    # very actuator down.
    down = first >= 0
    assert down.any() and (~down).any()
    assert torch.equal(down, faults.failed.any(dim=-1))
    assert torch.all(faults.failed[down, first[down]])
    assert torch.all(state_kind[down] == STATE_JOINT_DOWN)
    assert torch.allclose(height[down], 0.95 + 0.001 * first[down].float(), atol=2e-4)

    # The others start from walking or from a trip.
    tripped = state_kind == STATE_TRIPPED
    walking = state_kind == STATE_WALKING
    assert torch.all((tripped | walking)[~down])
    assert tripped.any() and walking.any()
    assert torch.allclose(
      height[tripped], torch.full_like(height[tripped], 0.94), atol=2e-4
    )

    # A state recorded after a trip or with a joint down is never tipped, and
    # half of them are left as recorded.
    disturbed = ~walking
    assert torch.all(reset_term.tip_angle[disturbed] == 0.0)
    assert torch.all(reset_term.tip_angle[walking] > 0.0)
    as_recorded = reset_term.as_recorded
    assert not as_recorded[walking].any()
    assert as_recorded[disturbed].any() and (~as_recorded[disturbed]).any()
    assert torch.all(reset_term.kick_speed[as_recorded] == 0.0)
    assert torch.all(reset_term.kick_speed[~as_recorded] == 3.0)
    speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
    assert torch.allclose(
      speed[as_recorded], torch.ones_like(speed[as_recorded]), atol=1e-3
    )
  finally:
    env.close()


def test_forearm_torso_contact_is_measured_but_not_penalized():
  """Pressing a forearm on the chest is seen by the shield sensor only."""
  env = _make_env(NUM_ENVS, state_buffer_path=None, max_kick_speed=0.0)
  try:
    env.reset()
    robot = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos.clone()
    folded = {
      "arm_left_1_joint": -1.482,
      "arm_left_2_joint": 1.357,
      "arm_left_3_joint": -2.585,
      "arm_left_4_joint": 2.259,
    }
    for name, value in folded.items():
      joint_pos[:, robot.joint_names.index(name)] = value
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    # Hold the folded arm with the position actuators for one control step.
    action_term = env.action_manager.get_term("joint_pos")
    action = (joint_pos - robot.data.default_joint_pos)[:, action_term.target_ids]
    env.step(action / action_term.scale)

    shield = env.scene["forearm_torso_contact"].data.force_history
    assert torch.linalg.vector_norm(shield, dim=-1).amax() > 1.0
    # The penalized self-collision sensor only watches contacts with the legs.
    hits = env.scene["self_collision"].data.force_history
    assert torch.linalg.vector_norm(hits, dim=-1).amax() == 0.0
  finally:
    env.close()


def test_shove_adds_a_horizontal_velocity_during_the_episode():
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = NUM_ENVS
  cfg.events["reset_fall_start"].params.update(
    state_buffer_path=None,
    max_kick_speed=0.0,
    max_kick_ang_speed=0.0,
    tip_fraction=0.0,
  )
  cfg.curriculum = {}
  replace_actuator_fault_cfg(cfg, count_probabilities=(1.0,))
  cfg.events["shove"].interval_range_s = (0.2, 0.2)
  cfg.events["shove"].params["speed_range"] = (2.0, 2.0)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    robot = env.scene["robot"]
    action = torch.zeros(
      env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    speeds = []
    for _ in range(14):
      env.step(action)
      speeds.append(
        torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
      )
    speeds = torch.stack(speeds)
    # Standing still until the shove at 0.2 s, then moving at about 2 m/s.
    assert torch.all(speeds[:8] < 0.3)
    assert torch.all(speeds.max(dim=0).values > 1.2)
    assert torch.all(speeds.max(dim=0).values < 2.3)
  finally:
    env.close()


def test_arm_action_scale_is_the_one_of_a_slower_arm():
  from pal_mjlab.robots import KANGAROO_ACTION_SCALE, NATURAL_FREQ

  scale = load_env_cfg(TASK).actions["joint_pos"].scale
  factor = (NATURAL_FREQ / (2.0 * math.pi * 1.0)) ** 2
  assert factor == pytest.approx(3.5**2)
  for joint, walker_scale in KANGAROO_ACTION_SCALE.items():
    expected = walker_scale * (factor if joint.startswith("arm_") else 1.0)
    assert scale[joint] == pytest.approx(expected)
  # The walking task's own scale is untouched.
  assert KANGAROO_ACTION_SCALE["arm_.*_1_joint"] == pytest.approx(0.053, abs=1e-3)


def test_checkpoint_with_another_action_scale_commands_the_same_targets(
  tmp_path, monkeypatch
):
  """A policy saved with a smaller arm scale is converted when it is loaded."""
  from pal_mjlab.scripts.eval_fall import load_actor_policy

  run_train(TASK, _train_cfg(monkeypatch), tmp_path)
  path = tmp_path / "model_0.pt"
  saved = torch.load(path, map_location="cpu", weights_only=False)
  task_scale = saved["infos"]["action_scale"]
  assert len(task_scale) == WALKER_ACTION_DIM
  # A spread as wide as a trained policy's, not the one of a single iteration.
  actor_state = saved["actor_state_dict"]
  actor_state["obs_normalizer._std"].fill_(2.0)
  actor_state["obs_normalizer._var"].fill_(4.0)
  torch.save(saved, path)

  # The same network, declared to come from a task with a 4x smaller arm scale.
  ratio = {j: 4.0 if j.startswith("arm_") else 1.0 for j in task_scale}
  saved["infos"]["action_scale"] = {j: s / ratio[j] for j, s in task_scale.items()}
  other_path = tmp_path / "other_scale.pt"
  torch.save(saved, other_path)
  # And one from before the scale was saved: taken to be the walking task's.
  del saved["infos"]["action_scale"]
  legacy_path = tmp_path / "legacy.pt"
  torch.save(saved, legacy_path)

  from mjlab.rl import RslRlVecEnvWrapper

  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = NUM_ENVS
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=get_test_device()))
  try:
    device = str(env.unwrapped.device)
    manager = env.unwrapped.observation_manager
    terms = manager.active_terms["actor"]
    dims = [math.prod(d) for d in manager.group_obs_term_dim["actor"]]
    start = sum(dims[: terms.index("actions")])
    columns = slice(start, start + WALKER_ACTION_DIM)
    names = env.unwrapped.action_manager.get_term("joint_pos").target_names
    k = torch.tensor([ratio[n] for n in names], device=device)

    obs, _ = env.reset()
    obs["actor"] = torch.randn_like(obs["actor"])
    with torch.inference_mode():
      reference = load_actor_policy(env, TASK, str(path), device)(obs).clone()
      # In the converted policy's world the last action is k times smaller.
      converted_obs = obs.clone()
      converted_obs["actor"][:, columns] = obs["actor"][:, columns] / k
      converted = load_actor_policy(env, TASK, str(other_path), device)(converted_obs)
      # Outputs k times smaller on a k times larger scale: the same targets.
      assert torch.allclose(converted * k, reference, rtol=1e-4, atol=1e-5)
      assert not torch.allclose(converted, reference, atol=1e-3)

      # Without a saved scale the checkpoint is taken to be on the walking
      # task's: the arms' is 3.5**2 times smaller than this task's.
      k = torch.where(k > 1.0, 3.5**2, 1.0)
      legacy_obs = obs.clone()
      legacy_obs["actor"][:, columns] = obs["actor"][:, columns] / k
      legacy = load_actor_policy(env, TASK, str(legacy_path), device)(legacy_obs)
      # Looser: the float32 rounding of the outputs is multiplied by 12.
      assert torch.allclose(legacy * k, reference, rtol=1e-3, atol=2e-4)
  finally:
    env.close()


def test_failed_actuators_go_limp_and_are_reported_to_the_policy():
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 64
  cfg.curriculum = {}
  # Three failures in every episode, on leg actuators only.
  replace_actuator_fault_cfg(
    cfg,
    count_probabilities=(0.0, 0.0, 0.0, 1.0),
    joint_weights={r"leg_.*": 1.0, r".*": 0.0},
    later_onset_range_s=(0.1, 0.5),
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    obs, _ = env.reset()
    fault_cfg = cfg.events["actuator_faults"].params["fault_cfg"]
    faults = get_actuator_faults(env, fault_cfg)
    assert len(faults.joint_names) == WALKER_ACTION_DIM
    is_leg = torch.tensor(
      [n.startswith("leg_") for n in faults.joint_names], device=env.device
    )
    planned = faults.planned()
    assert torch.all(planned.sum(dim=-1) == 3)
    assert not planned[:, ~is_leg].any()
    # One is down from the first step, the other two follow in 0.1-0.5 s.
    assert torch.all(faults.failed.sum(dim=-1) == 1)
    later = faults.onset_step[planned & ~faults.failed]
    assert torch.all((later >= 5) & (later <= 25))
    # The flags are the last actor observations, in action order.
    assert obs["actor"].shape == (64, ACTOR_OBS_DIM)
    assert torch.equal(obs["actor"][:, -WALKER_ACTION_DIM:], faults.failed.float())

    healthy_force = torch.zeros(WALKER_ACTION_DIM, device=env.device)
    for _ in range(30):
      action = torch.randn(env.num_envs, WALKER_ACTION_DIM, device=env.device)
      obs = env.step(action)[0]
      force = env.sim.data.actuator_force[:, faults._ctrl_ids].abs()
      # A failed actuator applies no force, whatever it is commanded.
      assert torch.all(force[faults.failed] == 0.0)
      healthy_force = torch.maximum(
        healthy_force, torch.where(faults.failed, 0.0, force).amax(dim=0)
      )
      assert torch.equal(obs["actor"][:, -WALKER_ACTION_DIM:], faults.failed.float())
    assert torch.equal(faults.failed, planned)
    assert torch.all(healthy_force > 0.0)

    # A viewer marks the three failed leg actuators of the robot it shows:
    # a ball and a bar per hinge, a bar along the leg for a leg length.
    marks = _RecordingVisualizer(env_idx=3)
    # (Sensors draw their own marks, so the term is asked directly.)
    env.event_manager.get_term_cfg("actuator_faults_onset").func.debug_vis(marks)
    names = [n for n, f in zip(faults.joint_names, faults.failed[3], strict=True) if f]
    slides = sum("length" in n for n in names)
    assert len(marks.cylinders) == 3
    assert len(marks.spheres) == 3 - slides
    root = env.scene["robot"].data.root_link_pos_w[3].cpu().numpy()
    for center in marks.spheres:
      assert np.linalg.norm(center - root) < 1.2

    # A new episode without failures: every actuator works again.
    faults.cfg.count_probabilities = (1.0,)
    faults._count_probabilities = torch.tensor([1.0], device=env.device)
    faults.max_count = 0
    env.reset()
    assert not faults.failed.any()
    gain = env.sim.model.actuator_gainprm[:, faults._ctrl_ids, 0]
    assert torch.all(gain > 0.0)
  finally:
    env.close()


def test_fault_mix_and_leg_weighting():
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 2048
  cfg.curriculum = {}
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    fault_cfg = cfg.events["actuator_faults"].params["fault_cfg"]
    faults = get_actuator_faults(env, fault_cfg)
    planned = faults.planned()
    count = planned.sum(dim=-1)
    for k, expected in enumerate((0.40, 0.40, 0.15, 0.05)):
      assert float((count == k).float().mean()) == pytest.approx(expected, abs=0.04)
    is_leg = torch.tensor(
      [n.startswith("leg_") for n in faults.joint_names], device=env.device
    )
    # 12 leg actuators at 4x the chance of the 10 others: 48 / 58.
    leg_share = float(planned[:, is_leg].sum() / planned.sum())
    assert leg_share == pytest.approx(48 / 58, abs=0.04)
    # With any failure, one is there from the first step (a later one can
    # start in the first step as well).
    assert torch.equal(faults.failed.any(dim=-1), count > 0)
  finally:
    env.close()


def test_actor_from_before_the_fault_flags_acts_the_same(tmp_path, monkeypatch):
  """An 83-input actor is widened and ignores the fault flags."""
  from mjlab.rl import RslRlVecEnvWrapper
  from pal_mjlab.scripts.eval_fall import load_actor_policy

  run_train(TASK, _train_cfg(monkeypatch), tmp_path)
  path = tmp_path / "model_0.pt"
  saved = torch.load(path, map_location="cpu", weights_only=False)
  actor_state = saved["actor_state_dict"]
  assert actor_state["mlp.0.weight"].shape == (512, ACTOR_OBS_DIM)
  # The reference does not look at the flags; the old one never had them.
  actor_state["mlp.0.weight"][:, WALKER_ACTOR_OBS_DIM:] = 0.0
  torch.save(saved, path)
  for key, value in list(actor_state.items()):
    if value.dim() == 2 and value.shape[1] == ACTOR_OBS_DIM:
      actor_state[key] = value[:, :WALKER_ACTOR_OBS_DIM].clone()
  old_path = tmp_path / "before_flags.pt"
  torch.save(saved, old_path)

  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = NUM_ENVS
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=get_test_device()))
  try:
    device = str(env.unwrapped.device)
    obs, _ = env.reset()
    obs["actor"] = torch.randn_like(obs["actor"])
    with torch.inference_mode():
      reference = load_actor_policy(env, TASK, str(path), device)(obs).clone()
      old_policy = load_actor_policy(env, TASK, str(old_path), device)
      assert torch.allclose(old_policy(obs), reference, atol=1e-6)
      flipped = obs.clone()
      flipped["actor"][:, WALKER_ACTOR_OBS_DIM:] = 1.0
      assert torch.allclose(old_policy(flipped), reference, atol=1e-6)
  finally:
    env.close()
