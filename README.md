# Territorial World Model Prototype

This folder contains a lightweight, fully local prototype for a
territoriality-inspired world model in a grid navigation setting.

## Idea

The environment is split into multiple territories. The agent only observes:

- local `(x, y)` coordinates inside the current territory
- whether it is currently standing on a door cell

This creates state aliasing: identical local observations can occur in
different territories, but the transition dynamics and cross-boundary behavior
are not the same everywhere.

We compare two tabular world models:

- `FlatWorldModel`: predicts next state from local observation and action only
- `TerritorialWorldModel`: adds territory identity and explicit boundary events

## Files

- `territorial_nav_experiment.py`: environment, models, planning, and training
- `run_territorial_experiment.py`: one-command experiment runner
- `outputs/`: generated metrics and figure

## Run

```bash
python3 territorial_world_model/run_territorial_experiment.py
```

## Expected outcome

The territorial model should outperform the flat model on:

- one-step transition prediction
- navigation success with model-based planning
- boundary crossing prediction
