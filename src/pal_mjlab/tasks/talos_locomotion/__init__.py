import gymnasium as gym

gym.register(
    id="Mjlab-Velocity-Rough-Pal-Talos",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfgs:PAL_TALOS_ROUGH_ENV_CFG",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:PalTalosPPORunnerCfg",
    },
)

gym.register(
    id="Mjlab-Velocity-Flat-Pal-Talos",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfgs:PAL_TALOS_FLAT_ENV_CFG",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:PalTalosPPORunnerCfg",
    },
)
