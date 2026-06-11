import math
import torch
import mjlab.tasks  # noqa: F401 – populates task registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0"

configure_torch_backends()
device = "cpu"

env_cfg = load_env_cfg(TASK_ID, play=True)
env_cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(TASK_ID)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

inner_env = env_wrapped.unwrapped
obs_manager = inner_env.observation_manager

# We will collect observations over 50 resets
num_resets = 50
group_samples = {}

for step in range(num_resets):
  obs_td, _ = env_wrapped.reset()
  for group_name, tensor in obs_td.items():
    if group_name not in group_samples:
      group_samples[group_name] = []
    group_samples[group_name].append(tensor.clone().cpu())

print("=" * 80)
print(f"Observation Statistics over {num_resets} Resets")
print("=" * 80)

for group_name, list_tensors in group_samples.items():
  print(f"\nGroup: '{group_name}'")
  print("-" * 80)
  
  group_tensor = torch.cat(list_tensors, dim=0)
  
  names = obs_manager.active_terms.get(group_name, [])
  shapes = obs_manager.group_obs_term_dim.get(group_name, [])
  
  cursor = 0
  for name, shape in zip(names, shapes):
    dim = math.prod(shape)
    term_tensor = group_tensor[:, cursor : cursor + dim]  # shape: (50, dim)
    
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
