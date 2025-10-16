import gymnasium as gym

gym.register(
  id="Mjlab-Velocity-Rough-Kangaroo",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KangRoughEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:KangPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Rough-Kangaroo-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:KangRoughEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:KangPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Kangaroo",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KangFlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:KangPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Kangaroo-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:KangFlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:KangPPORunnerCfg",
  },
)
