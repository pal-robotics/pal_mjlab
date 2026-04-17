from pal_mjlab.tasks.tracking.kangaroo.env_cfgs import pal_kangaroo_flat_tracking_env_cfg

cfg = pal_kangaroo_flat_tracking_env_cfg()
print("Actor Observations:")
for k in cfg.observations["actor"].terms.keys():
    print(f"  - {k}")

print("\nCritic Observations:")
for k in cfg.observations["critic"].terms.keys():
    print(f"  - {k}")

# Check noise
print("\nNoise check:")
print(f"base_lin_acc actor noise: {cfg.observations['actor'].terms['base_lin_acc'].noise}")
print(f"imu_projected_gravity actor noise: {cfg.observations['actor'].terms['imu_projected_gravity'].noise}")
