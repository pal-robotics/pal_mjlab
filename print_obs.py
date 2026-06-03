"""Print the full actor observation at step 0 using the zero agent."""

import mjlab.tasks  # noqa: F401 – populates task registry

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Manipulation-Lift-Cube-Keypoints-Pal-Tiago-Pro-v0"

configure_torch_backends()
device = "cpu"

env_cfg = load_env_cfg(TASK_ID, play=True)
env_cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(TASK_ID)

# Headless. The wrapper calls env.reset() internally in __init__.
env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# get_observations() returns a TensorDict
obs_td = env_wrapped.get_observations()
actor_obs = obs_td["actor"]  # shape: (1, 29)

print("=" * 70)
print(f"Task             : {TASK_ID}")
print(f"Actor obs shape  : {actor_obs.shape}")
print("=" * 70)

inner_env = env_wrapped.unwrapped
obs_manager = inner_env.observation_manager

print("\nAll observation groups:", list(obs_td.keys()))

# active_terms:       dict[group -> list[str]]   (term names, in order)
# group_obs_term_dim: dict[group -> list[tuple]]  (shapes, same order)
names  = obs_manager.active_terms.get("actor", [])       # e.g. ['joint_pos', ...]
shapes = obs_manager.group_obs_term_dim.get("actor", []) # e.g. [(7,), (7,), ...]

print("\n--- 'actor' group – per-term breakdown (env 0) ---")
actor_flat = actor_obs[0]  # (29,)
cursor = 0
import math
for name, shape in zip(names, shapes):
    dim = math.prod(shape)
    vals = actor_flat[cursor : cursor + dim].tolist()
    print(f"  {name:45s}  shape={str(shape):10s}  {vals}")
    cursor += dim

print("\n--- Full flat 'actor' obs (env 0) ---")
print(actor_obs[0].tolist())

# ---- camera group (keypoints) ----
if "camera" in obs_td.keys():
    cam_obs = obs_td["camera"][0]  # (12,)
    cam_names  = obs_manager.active_terms.get("camera", [])
    cam_shapes = obs_manager.group_obs_term_dim.get("camera", [])

    print("\n--- 'camera' group – per-term breakdown (env 0) ---")
    cursor = 0
    for name, shape in zip(cam_names, cam_shapes):
        dim = math.prod(shape)
        vals = cam_obs[cursor : cursor + dim].tolist()
        print(f"  {name:45s}  shape={str(shape):10s}  {vals}")

        # Pretty-print keypoints as (u, v) pairs if shape is flat (2*N,)
        if dim % 2 == 0:
            pairs = [(round(vals[i], 4), round(vals[i+1], 4)) for i in range(0, dim, 2)]
            print(f"    → as (x,y) pixel pairs: {pairs}")
        cursor += dim

env_wrapped.close()
