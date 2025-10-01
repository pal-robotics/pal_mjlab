# Kangaroo in MjLab

This repository showcase the implementation of PAL's Kangaroo robot into MjLab. 

> As MjLab is still in early development, this repository may be impacted by breaking changes. If an issue were to arise when running one of the scripts, feel free to open an issue or contribute to the project. Thanks you for your understanding!

## What's MjLab?

MjLab is a project to have the Isaac Lab API using MjWarp as the backend. You can find it [here](https://github.com/mujocolab/mjlab). If you’re wondering about the motivation behind it or how it differs from Newton, you can learn more [here](https://github.com/mujocolab/mjlab/blob/main/docs/motivation.md).

It’s really easy to install, and sim-to-real has been tested successfully on the G1 and Go1 for RL locomotion and motion imitation, see more [here](https://x.com/kevin_zakka/status/1972757435707424898?t=4Ho4ovrCAEOWTCxVG3BKrw&s=19).

## Quickstart

### Clone the repository

```bash
git clone https://github.com/BDX-R/BDX-R-MjLab.git && cd mjlab_kangaroo
```

### List available environments

```bash
uv run scripts/list_envs.py
```

### Dummy agents

```bash
uv run scripts/zero_agent.py Mjlab-Velocity-Flat-BDX-R
```

or

```bash
uv run scripts/random_agent.py Mjlab-Velocity-Flat-BDX-R
```

### Train

```bash
uv run scripts/train.py Mjlab-Velocity-Flat-BDX-R --env.scene.num-envs 4096
```

### Play

```bash
uv run scripts/play.py --task Mjlab-Velocity-Flat-BDX-R-Play --wandb-run-path your-org/mjlab/run-id
```

## Repository current state

<!-- TODO Louis -->
Here's the last result available:

![video](medias/kang_mjlab.mp4)

Our current focus to improve this work is on:

- todo
- todo

## Acknowledgements

- PAL Robotics
- MjLab
- MuJoCo Warp
- Isaac Lab