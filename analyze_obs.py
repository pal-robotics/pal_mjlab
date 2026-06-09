import math

import mjlab.tasks  # noqa: F401 – populates task registry
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Manipulation-Lift-Cube-Keypoints-Pal-Tiago-Pro-v0"

configure_torch_backends()
device = "cpu"

env_cfg = load_env_cfg(TASK_ID, play=True)
env_cfg.scene.num_envs = 100
agent_cfg = load_rl_cfg(TASK_ID)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# Let's step 1 time with zero actions
action_shape = env_wrapped.unwrapped.action_space.shape
actions = torch.zeros(action_shape, device=env_wrapped.unwrapped.device)
obs_td, rewards, terminated, extras = env_wrapped.step(actions)

# Now inspect observations in obs_td
print("=" * 80)
print(
  f"Ranges of observations across {env_cfg.scene.num_envs} simulations after 1 step:"
)
print("=" * 80)

inner_env = env_wrapped.unwrapped
obs_manager = inner_env.observation_manager

for group_name in obs_td.keys():
  print(f"\nGroup: '{group_name}'")
  print("-" * 80)

  group_tensor = obs_td[group_name]  # shape: (100, dim)
  names = obs_manager.active_terms.get(group_name, [])
  shapes = obs_manager.group_obs_term_dim.get(group_name, [])

  cursor = 0
  for name, shape in zip(names, shapes):
    dim = math.prod(shape)
    term_tensor = group_tensor[:, cursor : cursor + dim]  # shape: (100, dim)

    # Calculate min, max across all envs for each element/index in the term
    min_vals = torch.min(term_tensor, dim=0).values.tolist()
    max_vals = torch.max(term_tensor, dim=0).values.tolist()
    mean_vals = torch.mean(term_tensor, dim=0).tolist()

    print(f"  {name:30s} (shape={shape})")
    for i in range(dim):
      print(
        f"    idx {i:2d}: min={min_vals[i]:10.5f} | max={max_vals[i]:10.5f} | mean={mean_vals[i]:10.5f}"
      )
    cursor += dim

env_wrapped.close()
