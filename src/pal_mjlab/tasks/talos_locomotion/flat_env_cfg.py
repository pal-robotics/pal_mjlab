from dataclasses import dataclass

from pal_mjlab.tasks.talos_locomotion.rough_env_cfg import (
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
        


@dataclass
class PalTalosFlatEnvCfg_PLAY(PalTalosFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Effectively infinite episode length.
        self.episode_length_s = int(1e9)

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None

        # self.commands.twist.ranges.lin_vel_x = (-1.5, 2.0)
        # self.commands.twist.ranges.ang_vel_z = (-0.7, 0.7)


        self.commands.twist.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.twist.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.twist.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.twist.ranges.heading = (0.0, 0.0)