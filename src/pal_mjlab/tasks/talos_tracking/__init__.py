import gymnasium as gym

gym.register(
  id="Mjlab-Tracking-Flat-Pal-Talos",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TalosFlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:TalosFlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Pal-Talos-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TalosFlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:TalosFlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Pal-Talos-No-State-Estimation",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TalosFlatNoStateEstimationEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:TalosFlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Pal-Talos-No-State-Estimation-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TalosFlatNoStateEstimationEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:TalosFlatPPORunnerCfg",
  },
)