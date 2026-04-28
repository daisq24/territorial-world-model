# Memory Maze — Territorial Memory Experiment

Path A of the territorial world model project. Compares a **FlatMemory** baseline
against a **TerritorialMemory** that fuses two signals:

1. Physical partition of the maze (rooms via connected components on the layout).
2. Familiarity built up online from visitation counts (dynamic, exploration-based).

Memory Maze (Pasukonis & Hafner, ICLR 2023) is a small 3D maze RL benchmark for
spatial long-term memory. The `-ExtraObs-` variants expose ground-truth
`agent_pos`, `maze_layout`, and `targets_pos`, which lets us evaluate memory
quality directly without training a full RL policy.

## Setup (on your local machine)

One command:

```bash
cd memory_maze_experiment
bash setup.sh
```

This creates a project-local `.venv/` inside the folder and installs everything
from `requirements.txt`. Subsequent sessions just need `source .venv/bin/activate`.

If you hit a MuJoCo rendering error on a headless machine, set:

```bash
export MUJOCO_GL=egl       # GPU headless
# or
export MUJOCO_GL=osmesa    # CPU-only; needs apt install libosmesa6-dev
# or
export MUJOCO_GL=glfw      # desktop with a display
```

## Sanity check

Confirm the environment runs on your machine:

```bash
python env_probe.py
```

Expected: prints observation keys, action space, one random rollout of 50 steps,
saves `probe_output.png` (top-down snapshot).

## Run the comparison

```bash
python experiment.py --size 9x9 --episodes 20 --seed 0
```

Outputs go into `outputs/`:

- `metrics.json` — per-episode target-localization accuracy and success rate
- `comparison.png` — summary bar chart of FlatMemory vs TerritorialMemory
- `familiarity_map_ep{N}.png` — visualization of the dual-source territory for sample episodes

## Files

- `region_utils.py` — extracts rooms from `maze_layout` via connected components.
- `memory_models.py` — `FlatMemory` and `TerritorialMemory` classes.
- `env_probe.py` — minimal sanity check, confirms the environment works.
- `experiment.py` — runs both memory models on N episodes and compares.

## Evaluation metric (why offline, not RL)

We use an **offline probing** evaluation: the agent follows a fixed exploration
policy (random walk or shortest-path-to-frontier) for K steps, then at
evaluation time the memory is asked "where is the target of color C?" The
prediction is scored against ground-truth `targets_pos`.

This isolates memory quality from policy quality. It is the same protocol used
in the original Memory Maze paper's offline dataset. Training an RL agent comes
later, after we know the memory module is actually better.

## Next step after this runs

Path A.2: extend the FlatMemory → TerritorialMemory comparison to a 2-agent
setting (two rolling balls in the same maze, shared familiarity map vs
separate-but-negotiated territories). The codebase is organized to make this
a single-file change in `memory_models.py` + a wrapper for multi-agent Memory
Maze.
