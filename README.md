# PAL Robotics in MjLab

This repository showcase the implementation of PAL's robots into MjLab. 

> [!WARNING]
> As MjLab is still in early development, this repository may be impacted by breaking changes. If an issue were to arise when running one of the scripts, feel free to open an issue or contribute to the project. Thanks you for your understanding!

https://github.com/user-attachments/assets/7935ed66-16ae-4fcb-8b3f-3f2cc3d8589b

> [!WARNING]
> Kangaroo locomotion is still WIP.

## What's MjLab?

MjLab is a project to have the Isaac Lab API using MjWarp as the backend. You can find it [here](https://github.com/mujocolab/mjlab). If you’re wondering about the motivation behind it or how it differs from Newton, you can learn more [here](https://github.com/mujocolab/mjlab/blob/main/docs/motivation.md).

It’s really easy to install, and sim-to-real has been tested successfully on the G1 and Go1 for RL locomotion and motion imitation, see more [here](https://x.com/kevin_zakka/status/1972757435707424898?t=4Ho4ovrCAEOWTCxVG3BKrw&s=19).

## Quickstart

Clone the repository.

```bash
git clone https://github.com/louislelay/pal_mjlab.git && cd pal_mjlab
```

List available environments.

```bash
uv run pal_list_envs
```

Use the dummy agents.

```bash
uv run pal_play Mjlab-Velocity-Flat-Pal-Talos --agent zero # send zero actions to the robot
uv run pal_play Mjlab-Velocity-Flat-Pal-Talos --agent random # send random actions to the robot
```

### Velocity Tracking

Train the policy.

```bash
uv run pal_train Mjlab-Velocity-Flat-Pal-Talos --env.scene.num-envs 4096
```

Evaluate the policy.

```bash
uv run pal_play Mjlab-Velocity-Flat-Pal-Talos-Play --wandb-run-path your-org/mjlab/run-id
```

### Motion Tracking

Using [GMR](https://github.com/YanjieZe/GMR), create a `.csv` file using one of the motion provided by the [LaFAN1 dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

> [!NOTE]
> For ease of use, we provide `talos_dancing.csv`.

```bash
cd path/to/GMR
# retarget the wanted motion to Talos
python scripts/bvh_to_robot.py --bvh_file path/to/motion.bvh --save_path path/to/motion.pkl --robot pal_talos --rate_limit --format lafan1
# convert the file to be mjlab-compatible
python scripts/batch_gmr_pkl_to_csv.py --folder path/to/folder
```

Convert from csv to npz.

```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
    --input-file path/to/motion.csv \
    --output-name motion_name \
    --input-fps 30 \
    --output-fps 50 \
    --render
```

Train the policy.

```bash
MUJOCO_GL=egl uv run pal_train Mjlab-Tracking-Flat-Pal-Talos --registry-name your-org/csv_to_npz/motion_name
```

Evaluate the policy.

```bash
uv run pal_play Mjlab-Tracking-Flat-Pal-Talos-Play --wandb-run-path your-org/mjlab/run-id
```

## Acknowledgements

We're grateful to the people behind MjLab, PAL Robotics, MuJoCo Warp and Isaac Lab.
