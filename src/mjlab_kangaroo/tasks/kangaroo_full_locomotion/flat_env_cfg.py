from dataclasses import dataclass

from mjlab_kangaroo.tasks.kangaroo_full_locomotion.rough_env_cfg import (
    KangFullRoughEnvCfg,
)


@dataclass
class KangFullFlatEnvCfg(KangFullRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        assert self.scene.terrain is not None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None


@dataclass
class KangFlatEnvCfg_PLAY(KangFullFlatEnvCfg):
    pass
