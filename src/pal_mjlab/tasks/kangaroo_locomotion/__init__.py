import gymnasium as gym

gym.register(
    id="Mjlab-Velocity-Rough-Pal-Kangaroo",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfgs:PAL_KANGAROO_ROUGH_ENV_CFG",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:PalKangarooPPORunnerCfg",
    },
)

gym.register(
    id="Mjlab-Velocity-Flat-Pal-Kangaroo",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfgs:PAL_KANGAROO_FLAT_ENV_CFG",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:PalKangarooPPORunnerCfg",
    },
)
