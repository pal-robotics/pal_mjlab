from dataclasses import dataclass

from mjlab_kangaroo.tasks.kangaroo_locomotion.rough_env_cfg import (
    KangRoughEnvCfg,
)


@dataclass
class KangFlatEnvCfg(KangRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        assert self.scene.terrain is not None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None


@dataclass
class KangFlatEnvCfg_PLAY(KangFlatEnvCfg):
    pass
