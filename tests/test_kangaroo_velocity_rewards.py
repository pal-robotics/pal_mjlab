"""Tests for reward manager functionality."""

from unittest.mock import Mock

import mujoco
import pytest
import torch
import math
from conftest import get_test_device

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.rewards import *
from mjlab.tasks.velocity.mdp.rewards import *
from pal_mjlab.tasks.velocity.mdp.rewards import *
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_flat_env_cfg

PARTIALLY_ACTUATED_ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="base_geom" type="cylinder" size="0.1 0.05" mass="1.0"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="actuated_joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="link1_geom" type="box" size="0.05 0.05 0.2" mass="0.5"/>
        <body name="link2" pos="0 0 0.4">
          <joint name="passive_joint" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
          <geom name="link2_geom" type="box" size="0.05 0.05 0.15" mass="0.3"/>
          <body name="link3" pos="0 0 0.3">
            <joint name="actuated_joint2" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
            <geom name="link3_geom" type="box" size="0.05 0.05 0.1" mass="0.2"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


class SimpleTestReward:
  """A simple class-based reward for testing that tracks state."""

  def __init__(self, cfg: RewardTermCfg, env):
    self.num_envs = env.num_envs
    self.device = env.device
    self.current_air_time = torch.zeros((self.num_envs, 1), device=self.device)

  def __call__(self, env, **kwargs):
    self.current_air_time += 0.01
    return torch.ones(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | None = None, env=None):
    if env_ids is not None and len(env_ids) > 0:
      self.current_air_time[env_ids] = 0


class StatelessReward:
  """A stateless class-based reward without reset method."""

  def __init__(self, cfg: RewardTermCfg, env):
    pass

  def __call__(self, env, **kwargs):
    return torch.ones(env.num_envs)


@pytest.fixture
def mock_env():
  """Create a mock environment for testing."""
  env = Mock()
  env.num_envs = 4
  env.device = "cpu"
  env.step_dt = 0.01
  env.max_episode_length_s = 10.0
  robot = Mock()
  env.scene = {"robot": robot}
  env.command_manager.get_command = Mock(
    return_value=torch.tensor([[1.0, 0.0, 0.0]] * 4)
  )
  return env

@pytest.fixture
def mock_asset_cfg():
  asset_cfg = Mock()
  asset_cfg.name = "robot"

  return asset_cfg

@pytest.fixture
def class_reward_config():
  """Config with a class-based reward."""
  return {
    "term": RewardTermCfg(
      func=SimpleTestReward,
      weight=1.0,
      params={},
    )
  }


@pytest.fixture
def function_reward_config():
  """Config with a function-based reward."""
  return {
    "term": RewardTermCfg(
      func=lambda env: torch.ones(env.num_envs),
      weight=1.0,
      params={},
    )
  }

@pytest.fixture
def stateless_reward_config():
  """Config with a stateless class-based reward."""
  return {
    "term": RewardTermCfg(
      func=StatelessReward,
      weight=1.0,
      params={},
    )
  }


def test_track_linear_velocity (mock_env, mock_asset_cfg):
  env = mock_env
  class command_manager :
    def __init__(self):

      dummy_vel_command = torch.zeros((env.num_envs, 3), device=env.device)

      self.active_terms = {
        "twist" : dummy_vel_command,
      }

    def get_command(self, command_name):
      if command_name not in self.active_terms.keys():
        return None
      
      return self.active_terms[command_name]
    
  env.scene["robot"].data.root_link_lin_vel_b = torch.ones((env.num_envs, 3), device = env.device)

  env.command_manager = command_manager()

  value = track_linear_velocity(env, std = 1.0, command_name="twist", asset_cfg=mock_asset_cfg)

  assert value.shape == (env.num_envs,),(
    f"Track linear velocity returned tensor of wrong shape, expected {(env.num_envs,)} got {value.shape}"
  )
  assert value[0] == math.exp(-3.0), "Track linear velocity reward returned incorrect value"