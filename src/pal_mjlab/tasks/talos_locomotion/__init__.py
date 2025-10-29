import gymnasium as gym

gym.register(
    id="Mjlab-Velocity-Rough-Pal-Talos",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:PalTalosRoughEnvCfg",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:PalTalosPPORunnerCfg",
    },
)

gym.register(
    id="Mjlab-Velocity-Rough-Pal-Talos-Play",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:PalTalosRoughEnvCfg_PLAY",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:PalTalosPPORunnerCfg",
    },
)

gym.register(
    id="Mjlab-Velocity-Flat-Pal-Talos",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PalTalosFlatEnvCfg",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:PalTalosPPORunnerCfg",
    },
)

gym.register(
    id="Mjlab-Velocity-Flat-Pal-Talos-Play",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PalTalosFlatEnvCfg_PLAY",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:PalTalosPPORunnerCfg",
    },
)
