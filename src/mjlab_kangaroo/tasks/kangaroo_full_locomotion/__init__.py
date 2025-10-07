import gymnasium as gym

gym.register(
    id="mjlab_kangaroo-Velocity-Rough-Kangaroo-Full",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KangFullRoughEnvCfg",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:KangFullPPORunnerCfg",
    },
)

gym.register(
    id="mjlab_kangaroo-Velocity-Rough-Kangaroo-Full-Play",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KangFullRoughEnvCfg_PLAY",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:KangFullPPORunnerCfg",
    },
)

gym.register(
    id="mjlab_kangaroo-Velocity-Flat-Kangaroo-Full",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KangFullFlatEnvCfg",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:KangFullPPORunnerCfg",
    },
)

gym.register(
    id="mjlab_kangaroo-Velocity-Flat-Kangaroo-Full-Play",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KangFullFlatEnvCfg_PLAY",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:KangFullPPORunnerCfg",
    },
)
