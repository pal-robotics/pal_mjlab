# PAL Robotics in MjLab

This repository showcase the implementation of PAL's robots into MjLab. 

> [!WARNING]
> As MjLab is still in early development, this repository may be impacted by breaking changes. If an issue were to arise when running one of the scripts, feel free to open an issue or contribute to the project. Thanks you for your understanding!

![Demo 1](medias/talos_velocity.gif)

<!-- | ![Demo 1](medias/talos_velocity.gif) | ![Demo 2](medias/kang_velocity.gif) |
|:--:|:--:|
| Légende 1 | Légende 2 | -->


## What's MjLab?

MjLab is a project to have the Isaac Lab API using MjWarp as the backend. You can find it [here](https://github.com/mujocolab/mjlab). If you’re wondering about the motivation behind it or how it differs from Newton, you can learn more [here](https://github.com/mujocolab/mjlab/blob/main/docs/motivation.md).

It’s really easy to install, and sim-to-real has been tested successfully on the G1 and Go1 for RL locomotion and motion imitation, see more [here](https://x.com/kevin_zakka/status/1972757435707424898?t=4Ho4ovrCAEOWTCxVG3BKrw&s=19).

## Quickstart

### Clone the repository

```bash
git clone https://github.com/louislelay/mjlab_kangaroo.git && cd mjlab_kangaroo
```

### List available environments

```bash
uv run pal_list_envs
```

### Dummy agents

```bash
uv run pal_play Mjlab-Velocity-Flat-Talos --agent zero # send zero actions to the robot
uv run pal_play Mjlab-Velocity-Flat-Talos --agent random # send random actions to the robot
```

### Train

```bash
uv run pal_train Mjlab-Velocity-Flat-Talos --env.scene.num-envs 4096
```

### Play

```bash
uv run pal_play Mjlab-Velocity-Flat-Talos-Play --wandb-run-path your-org/mjlab/run-id
```

## Acknowledgements

- PAL Robotics
- MjLab
- MuJoCo Warp
- Isaac Lab