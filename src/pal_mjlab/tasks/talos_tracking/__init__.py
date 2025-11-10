import gymnasium as gym

gym.register(
  id="Mjlab-Tracking-Flat-Pal-Talos",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.env_cfgs:PAL_TALOS_FLAT_TRACKING_ENV_CFG",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PalTalosFlatPPORunnerCfg",
  },
)


gym.register(
  id="Mjlab-Tracking-Flat-Pal-Talos-No-State-Estimation",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.env_cfgs:PAL_TALOS_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:PalTalosFlatPPORunnerCfg",
  },
)