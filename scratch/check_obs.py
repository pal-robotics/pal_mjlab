
import torch
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.tracking.kangaroo.env_cfgs import pal_kangaroo_flat_tracking_env_cfg

def check_obs():
    cfg = pal_kangaroo_flat_tracking_env_cfg()
    # We need to mock the device and other things to avoid full mujoco init if possible, 
    # but ManagerBasedRlEnv might need a real sim.
    # Let's try to just look at the observation manager config.
    
    env = ManagerBasedRlEnv(cfg, device="cpu")
    obs_mgr = env.unwrapped.observation_manager
    sim = env.unwrapped.sim
    print(f"Num DOFs (nv): {sim.model.nv}")
    print(f"Num Actuators (nu): {sim.model.nu}")
    
    print("Observation Groups:", obs_mgr.group_obs_dim.keys())
    
    for group, dim in obs_mgr.group_obs_dim.items():
        print(f"\nGroup: {group}, Total Dim: {dim}")
        current_idx = 0
        for term_name, term_dim in obs_mgr.group_obs_term_dim[group].items():
            print(f"  - {term_name}: {term_dim} (Slice: {current_idx}:{current_idx + term_dim})")
            current_idx += term_dim

if __name__ == "__main__":
    check_obs()
