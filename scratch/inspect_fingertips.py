import torch
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK_ID = "Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0"

def main():
    device = "cpu"
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env.reset()

    robot = env.scene["robot"]
    box = env.scene["box"]

    print("Robot site names:", robot.site_names)
    fingertip_site_names = [s for s in robot.site_names if "fingertip" in s]
    print("Fingertip site names:", fingertip_site_names)

    # Get fingertip site IDs
    site_ids, _ = robot.find_sites(fingertip_site_names, preserve_order=True)
    print("Fingertip site IDs:", site_ids)

    # Get fingertip positions
    site_pos = robot.data.site_pos_w[0, site_ids]
    print("Fingertip positions:\n", site_pos)

    # Squeeze axis
    squeeze_dir = site_pos[0] - site_pos[1]
    squeeze_dir_norm = squeeze_dir / torch.norm(squeeze_dir)
    print("Squeeze direction (normalized):", squeeze_dir_norm.tolist())

    # Let's inspect env.sim.data.site_xmat
    # site_xmat shape: [num_envs, num_sites, 9] or [num_envs, num_sites, 3, 3]
    site_xmat = env.sim.data.site_xmat
    print("site_xmat shape:", site_xmat.shape)

    for idx, name in zip(site_ids, fingertip_site_names):
        xmat = site_xmat[0, idx].reshape(3, 3)
        print(f"\nSite: {name} (ID: {idx})")
        print("Rotation matrix:\n", xmat)
        # Check which column is closest to the squeeze direction
        for col_idx in range(3):
            col = xmat[:, col_idx]
            dot = torch.dot(col, squeeze_dir_norm)
            print(f"  Axis {col_idx} dot squeeze_dir: {dot.item():.4f}")

    env.close()

if __name__ == "__main__":
    main()
