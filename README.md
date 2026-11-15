[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![mjlab](https://img.shields.io/badge/mjlab-1.6.0-76B900.svg)](https://mujocolab.github.io/mjlab/v1.6.0/index.html)

# PAL Robotics in mjlab

This repository showcases the implementation of [PAL Robotics](https://pal-robotics.com/)' robots in [mjlab](https://github.com/mujocolab/mjlab).

![Project banner](static/pal_mjlab_viser.png)

## What is mjlab?

mjlab brings the [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) API to [MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html).
It's lightweight, easy to install, and has been validated for sim-to-real transfer on the G1 and Go1 robots for RL locomotion and motion imitation.
See the [announcement thread](https://x.com/kevin_zakka/status/1972757435707424898) for videos, or read about the [motivation behind mjlab](https://github.com/mujocolab/mjlab/blob/main/docs/motivation.md).

## Installation

Install uv.
```bash
curl -LsSf https://astral.sh/uv/install.sh \| sh
```

Clone the repository.

```bash
git clone https://github.com/pal-robotics/pal_mjlab.git
cd pal_mjlab
uv sync
```

## Quick Start

List available environments.

```bash
uv run list_envs --keyword pal
```

Test with dummy agents.

```bash
uv run play Mjlab-Velocity-Flat-Pal-Kangaroo --agent zero    # send zero actions
uv run play Mjlab-Velocity-Flat-Pal-Kangaroo --agent random  # send random actions
```


## Velocity Tracking

Train a locomotion policy.

```bash
uv run train Mjlab-Velocity-Flat-Pal-Kangaroo --env.scene.num-envs 4096
```

Evaluate a trained policy.

```bash
uv run play Mjlab-Velocity-Flat-Pal-Kangaroo --wandb-run-path your-org/mjlab/run-id
```


## Motion Imitation

We added support to [GMR](https://github.com/YanjieZe/GMR) to retarget animations from the [LaFAN1 dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) for PAL robots (KANGAROO and Talos).

### Retargeting a new motion

First, use GMR to retarget and convert a motion file. Note that for convenience we provide a few retargeted motions under `motions` folder.

Then convert the CSV to NPZ format.

```bash
uv run -m pal_mjlab.scripts.csv_to_npz \
    --input-file motions/kang_walk.csv \
    --output-name kang_walk \
    --input-fps 30 \
    --output-fps 50 \
    --robot-name kangaroo \
    --render True
```

### Training and evaluation

Train.

```bash
uv run train Mjlab-Tracking-Flat-Pal-Kangaroo \
    --registry-name your-org/csv_to_npz/kang_walk \
    --env.scene.num-envs 4096
```

Evaluate.

```bash
uv run play Mjlab-Tracking-Flat-Pal-Kangaroo --wandb-run-path your-org/mjlab/run-id
```

## Safe Fall (KANGAROO fallback controller)

A fallback policy that a supervisor switches to when a fall or an actuator fault is detected. From the instant it takes over it must come to rest with the least damage: on its knees when it can, forward onto knees and forearms when it cannot, and never on the torso or the pelvis actuators.
The damage objective follows [Ha & Liu, IROS 2015](https://faculty.cc.gatech.edu/~sha9/projects/ha2015mcs/index.html): the cost of a fall is its largest single impact, not the total.

The task uses its own robot model, with the real link meshes as collision geometry instead of the idealized capsules. Its actor observations are the velocity task's plus one flag per actuator that is down, and its actions are the same, so it is trained starting from a walking policy. Failed actuators go limp but still report their state.

Record start states from the walking policy: walking, a detection time after a foot was caught, and a detection time after an actuator went limp. Without this file every episode starts standing still.

```bash
uv run python -m pal_mjlab.scripts.record_walker_states --checkpoint /path/to/walker/model.pt
```

Train, starting from the walking policy's actor.

```bash
uv run train Mjlab-SafeFall-Flat-Pal-Kangaroo --env.scene.num-envs 4096 \
    --agent.init-actor-checkpoint /path/to/walker/model.pt \
    --agent.actor-freeze-iterations 20
```

Evaluate a controller and print the damage metrics by kick speed, by how the episode ends and by what failed. `zero` and `damping` are passive baselines; `policy` takes any actor checkpoint, including a walking policy.

```bash
uv run python -m pal_mjlab.scripts.eval_fall --controller damping
uv run python -m pal_mjlab.scripts.eval_fall --controller policy --checkpoint /path/to/model.pt
```

Watch a trained policy. Failed actuators are marked in red.

```bash
uv run play Mjlab-SafeFall-Flat-Pal-Kangaroo --checkpoint-file /path/to/model.pt
```

The contact and constraint buffers (`nconmax`, `njmax`) are sized from measured peaks. After changing the collision model or the disturbances, re-measure them with `uv run python -m pal_mjlab.scripts.measure_fall_buffers`.

## Results

<video src="https://github.com/user-attachments/assets/32c0e534-ead9-400e-afe5-42174a6b739b" controls muted loop playsinline style="width:100%;"></video>

## Contributing

Contributions are welcome!
Please open an issue to discuss proposed changes or report bugs.
As mjlab is in early development, breaking changes may occur—thank you for your patience.

## Acknowledgements

Thanks to the teams behind [mjlab](https://github.com/mujocolab/mjlab), [PAL Robotics](https://pal-robotics.com/), [MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html), [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html), and the [Inria HUCEBOT Team](https://team.inria.fr/hucebot/).

## License

See [LICENSE](LICENSE) for details.
