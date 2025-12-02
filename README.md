# PAL Robotics in mjlab

This repository showcase the implementation of [PAL](https://pal-robotics.com/)'s robots into [mjlab](https://github.com/mujocolab/mjlab). 

> [!WARNING]
> As mjlab is still in early development, this repository may be impacted by breaking changes. If an issue were to arise when running one of the scripts, feel free to open an issue or contribute to the project. Thank you for your understanding!

## What's mjlab?

mjlab is a project to have the [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) API using [MjWarp](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html) as the backend. If you’re wondering about the motivation behind it or how it differs from Newton, you can learn more about it [here](https://github.com/mujocolab/mjlab/blob/main/docs/motivation.md).

It’s really easy to install, and sim-to-real has been tested successfully on the G1 and Go1 for RL locomotion and motion imitation, see the announcement [tweet](https://x.com/kevin_zakka/status/1972757435707424898?t=4Ho4ovrCAEOWTCxVG3BKrw&s=19) for videos.

## Quickstart

Clone the repository.

```bash
git clone https://github.com/pal-robotics/pal_mjlab.git && cd pal_mjlab
```

List available environments.

```bash
uv run list_envs --keyword pal
```

Use the dummy agents.

```bash
uv run play Mjlab-Velocity-Flat-Pal-Talos --agent zero # send zero actions to the robot
uv run play Mjlab-Velocity-Flat-Pal-Talos --agent random # send random actions to the robot
```

### Velocity Tracking

Train the policy.

```bash
uv run train Mjlab-Velocity-Flat-Pal-Talos --env.scene.num-envs 4096
```

Evaluate the policy.

```bash
uv run play Mjlab-Velocity-Flat-Pal-Talos --wandb-run-path your-org/mjlab/run-id
```

### Motion Imitation

Using [GMR](https://github.com/YanjieZe/GMR), create a `.csv` file using one of the motion provided by the [LaFAN1 dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

> [!NOTE]
> For ease of use, we provide `talos_walking.csv`.

```bash
cd path/to/GMR
# retarget the wanted motion to Talos
python scripts/bvh_to_robot.py --bvh_file path/to/motion.bvh --save_path path/to/motion.pkl --robot pal_talos --rate_limit --format lafan1
# convert the file to be mjlab-compatible
python scripts/batch_gmr_pkl_to_csv.py --folder path/to/folder
```

Convert from csv to npz.

```bash
MUJOCO_GL=egl uv run -m pal_mjlab.scripts.csv_to_npz \
    --input-file path/to/motion.csv \
    --output-name motion_name \
    --input-fps 30 \
    --output-fps 50 \
    --render
```

Train the policy.

```bash
MUJOCO_GL=egl uv run train Mjlab-Tracking-Flat-Pal-Talos --registry-name your-org/csv_to_npz/motion_name
```

Evaluate the policy.

```bash
uv run play Mjlab-Tracking-Flat-Pal-Talos-Play --wandb-run-path your-org/mjlab/run-id
```

## A few of our results

> [!NOTE]
> We only display a few and they're not necessarly up-to-date.

<table>
  <tr>
    <td width="50%">
      <figure>
        <video src="https://github.com/user-attachments/assets/a378a306-f65d-448b-a6c8-3d40c3f7b3ce" controls muted loop playsinline style="width:100%; height:auto;"></video>
        <figcaption align="center"><em>Velocity Tracking for Talos</em></figcaption>
      </figure>
    </td>
    <td width="50%">
      <figure>
        <video src="https://github.com/user-attachments/assets/c0d6c444-a0a2-4c93-99a5-895ae7d31317" controls muted loop playsinline style="width:100%; height:auto;"></video>
        <figcaption align="center"><em>Motion Imitation for Talos</em></figcaption>
      </figure>
    </td>
  </tr>
  <tr width="50%">
    <figure>
      <video src="https://github.com/user-attachments/assets/cce8f9d2-d5e1-4218-828e-88d0ae1e0cdb" controls muted loop playsinline style="width:100%; height:auto;"></video>
      <figcaption align="center"><em>Velocity Tracking for Kangaroo</em></figcaption>
    </figure>
  </tr>
</table>

## Acknowledgements

We're grateful to the people behind mjlab, PAL Robotics, MuJoCo Warp, Isaac Lab, and the [Inria HUCEBOT Team](https://team.inria.fr/hucebot/).
