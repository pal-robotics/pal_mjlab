from dataclasses import dataclass

from mjlab_kangaroo.tasks.talos_locomotion.rough_env_cfg import (
  PalTalosRoughEnvCfg,
)


@dataclass
class PalTalosFlatEnvCfg(PalTalosRoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    assert self.scene.terrain is not None
    self.scene.terrain.terrain_type = "plane"
    self.scene.terrain.terrain_generator = None
    self.curriculum.terrain_levels = None

    self.curriculum.command_vel = None

    assert self.events.push_robot is not None
    self.events.push_robot.params["velocity_range"] = {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
    }


@dataclass
class PalTalosFlatEnvCfg_PLAY(PalTalosFlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)