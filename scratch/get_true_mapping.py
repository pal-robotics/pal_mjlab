import os
import torch
from mjlab.envs import make_env
from pal_mjlab.tasks.tracking.kangaroo.env_cfgs import pal_kangaroo_flat_tracking_env_cfg

# Create the environment config
cfg = pal_kangaroo_flat_tracking_env_cfg(play=True)

# Define action names by looking at the joint position action manager
# In mjlab/IsaacLab, we can look at the terms in the action manager
actuator_names = cfg.actions["joint_pos"].actuator_names
print("--- AUTHORITATIVE ACTION NAMES (22) ---")
for i, name in enumerate(actuator_names):
    print(f"{i}: {name}")

# Now for observations. We need to see the order of terms in the observation group 'actor'
print("\n--- AUTHORITATIVE OBSERVATION TERMS ---")
actor_obs = cfg.observations["actor"].terms
for term_name, term_cfg in actor_obs.items():
    # Attempt to estimate dimension if possible, but the order is the key
    print(f"Term: {term_name}")
