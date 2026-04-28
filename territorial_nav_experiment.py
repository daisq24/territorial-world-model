from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


Action = int
GlobalState = Tuple[int, int, int]
ObsState = Tuple[int, int, int]
PredictedObsState = Tuple[int, int, int]
PredictedTerritorialState = Tuple[int, int, int, int]


ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}


@dataclass(frozen=True)
class TerritorySpec:
    territory_id: int
    size: int
    doors: Tuple[Tuple[int, int], ...]
    slip: Dict[Action, Action]


class TerritorialGridEnv:
    """Grid environment with repeated local coordinates across territories.

    The agent observes only local position and whether it is on a door, which
    causes aliasing between territories. This makes boundary-aware modeling
    useful even in a compact setting.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        size = 4
        self.territories: Dict[int, TerritorySpec] = {
            0: TerritorySpec(
                territory_id=0,
                size=size,
                doors=((1, 3),),
                slip={},
            ),
            1: TerritorySpec(
                territory_id=1,
                size=size,
                doors=((1, 0), (2, 3)),
                slip={0: 3},
            ),
            2: TerritorySpec(
                territory_id=2,
                size=size,
                doors=((2, 0),),
                slip={3: 1},
            ),
        }
        self.boundary_links: Dict[Tuple[int, int, int, Action], GlobalState] = {
            (0, 1, 3, 3): (1, 1, 0),
            (1, 1, 0, 2): (0, 1, 3),
            (1, 2, 3, 3): (2, 2, 0),
            (2, 2, 0, 2): (1, 2, 3),
        }
        self.goal_state: GlobalState = (2, 3, 0)

    def reset(self) -> GlobalState:
        territory = self.rng.choice(list(self.territories.keys()))
        spec = self.territories[territory]
        x = self.rng.randrange(spec.size)
        y = self.rng.randrange(spec.size)
        return (territory, x, y)

    def observe(self, state: GlobalState) -> ObsState:
        territory, x, y = state
        is_door = int((x, y) in self.territories[territory].doors)
        return (x, y, is_door)

    def is_goal(self, state: GlobalState) -> bool:
        return state == self.goal_state

    def step(self, state: GlobalState, action: Action) -> Tuple[GlobalState, Dict[str, int]]:
        territory, x, y = state
        spec = self.territories[territory]
        boundary_key = (territory, x, y, action)
        if boundary_key in self.boundary_links:
            return self.boundary_links[boundary_key], {"boundary_crossed": 1}

        effective_action = spec.slip.get(action, action)

        dx, dy = ACTION_DELTAS[effective_action]
        nx = min(max(x + dx, 0), spec.size - 1)
        ny = min(max(y + dy, 0), spec.size - 1)
        return (territory, nx, ny), {"boundary_crossed": 0}

    def valid_actions(self) -> Iterable[Action]:
        return ACTION_DELTAS.keys()


class FlatWorldModel:
    def __init__(self) -> None:
        self.transition_counts: Dict[Tuple[ObsState, Action], Counter] = defaultdict(Counter)
        self.boundary_counts: Dict[Tuple[ObsState, Action], Counter] = defaultdict(Counter)

    def update(
        self,
        obs: ObsState,
        action: Action,
        next_obs: ObsState,
        boundary_crossed: int,
    ) -> None:
        key = (obs, action)
        self.transition_counts[key][next_obs] += 1
        self.boundary_counts[key][boundary_crossed] += 1

    def predict(self, obs: ObsState, action: Action) -> Tuple[PredictedObsState, float]:
        key = (obs, action)
        if key not in self.transition_counts:
            return obs, 0.0
        next_obs, count = self.transition_counts[key].most_common(1)[0]
        total = sum(self.transition_counts[key].values())
        return next_obs, count / total

    def predict_boundary(self, obs: ObsState, action: Action) -> int:
        key = (obs, action)
        if key not in self.boundary_counts:
            return 0
        return self.boundary_counts[key].most_common(1)[0][0]


class TerritorialWorldModel:
    def __init__(self) -> None:
        self.transition_counts: Dict[Tuple[int, ObsState, Action], Counter] = defaultdict(Counter)
        self.boundary_counts: Dict[Tuple[int, ObsState, Action], Counter] = defaultdict(Counter)
        self.cross_counts: Dict[Tuple[int, ObsState, Action], Counter] = defaultdict(Counter)

    def update(
        self,
        territory: int,
        obs: ObsState,
        action: Action,
        next_territory: int,
        next_obs: ObsState,
        boundary_crossed: int,
    ) -> None:
        key = (territory, obs, action)
        self.transition_counts[key][next_obs] += 1
        self.boundary_counts[key][boundary_crossed] += 1
        self.cross_counts[key][next_territory] += 1

    def predict(
        self,
        territory: int,
        obs: ObsState,
        action: Action,
    ) -> Tuple[PredictedTerritorialState, float]:
        key = (territory, obs, action)
        if key not in self.transition_counts:
            return (territory, *obs), 0.0
        next_obs, count = self.transition_counts[key].most_common(1)[0]
        next_territory = self.cross_counts[key].most_common(1)[0][0]
        total = sum(self.transition_counts[key].values())
        return (next_territory, *next_obs), count / total

    def predict_boundary(self, territory: int, obs: ObsState, action: Action) -> int:
        key = (territory, obs, action)
        if key not in self.boundary_counts:
            return 0
        return self.boundary_counts[key].most_common(1)[0][0]


def collect_dataset(
    env: TerritorialGridEnv,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    dataset: List[dict] = []
    for _ in range(num_episodes):
        state = env.reset()
        for _ in range(steps_per_episode):
            action = rng.choice(list(env.valid_actions()))
            next_state, info = env.step(state, action)
            dataset.append(
                {
                    "state": state,
                    "obs": env.observe(state),
                    "action": action,
                    "next_state": next_state,
                    "next_obs": env.observe(next_state),
                    "boundary_crossed": info["boundary_crossed"],
                }
            )
            state = next_state
    return dataset


def train_models(dataset: List[dict]) -> Tuple[FlatWorldModel, TerritorialWorldModel]:
    flat = FlatWorldModel()
    territorial = TerritorialWorldModel()

    for sample in dataset:
        territory = sample["state"][0]
        next_territory = sample["next_state"][0]
        flat.update(sample["obs"], sample["action"], sample["next_obs"], sample["boundary_crossed"])
        territorial.update(
            territory,
            sample["obs"],
            sample["action"],
            next_territory,
            sample["next_obs"],
            sample["boundary_crossed"],
        )
    return flat, territorial


def evaluate_prediction(
    dataset: List[dict],
    flat: FlatWorldModel,
    territorial: TerritorialWorldModel,
) -> Dict[str, float]:
    flat_correct = 0
    territorial_correct = 0
    flat_boundary_correct = 0
    territorial_boundary_correct = 0
    flat_cross_boundary_correct = 0
    territorial_cross_boundary_correct = 0
    cross_boundary_total = 0
    flat_aliased_correct = 0
    territorial_aliased_correct = 0

    alias_keys = set()
    next_state_support: Dict[Tuple[ObsState, Action], set] = defaultdict(set)
    for sample in dataset:
        next_state_support[(sample["obs"], sample["action"])].add(
            (sample["next_state"][0], *sample["next_obs"])
        )
    for key, targets in next_state_support.items():
        if len(targets) > 1:
            alias_keys.add(key)

    for sample in dataset:
        territory = sample["state"][0]
        flat_pred, _ = flat.predict(sample["obs"], sample["action"])
        terr_pred, _ = territorial.predict(territory, sample["obs"], sample["action"])

        if tuple(flat_pred) == tuple(sample["next_obs"]):
            flat_correct += 1
        if tuple(terr_pred) == (sample["next_state"][0], *sample["next_obs"]):
            territorial_correct += 1
        if flat.predict_boundary(sample["obs"], sample["action"]) == sample["boundary_crossed"]:
            flat_boundary_correct += 1
        if territorial.predict_boundary(territory, sample["obs"], sample["action"]) == sample["boundary_crossed"]:
            territorial_boundary_correct += 1
        if sample["boundary_crossed"] == 1:
            cross_boundary_total += 1
            if tuple(flat_pred) == tuple(sample["next_obs"]):
                flat_cross_boundary_correct += 1
            if tuple(terr_pred) == (sample["next_state"][0], *sample["next_obs"]):
                territorial_cross_boundary_correct += 1
        if (sample["obs"], sample["action"]) in alias_keys:
            if tuple(flat_pred) == tuple(sample["next_obs"]):
                flat_aliased_correct += 1
            if tuple(terr_pred) == (sample["next_state"][0], *sample["next_obs"]):
                territorial_aliased_correct += 1

    n = len(dataset)
    aliased_total = sum(1 for sample in dataset if (sample["obs"], sample["action"]) in alias_keys)
    return {
        "flat_next_state_accuracy": flat_correct / n,
        "territorial_next_state_accuracy": territorial_correct / n,
        "flat_boundary_accuracy": flat_boundary_correct / n,
        "territorial_boundary_accuracy": territorial_boundary_correct / n,
        "flat_cross_boundary_accuracy": flat_cross_boundary_correct / cross_boundary_total,
        "territorial_cross_boundary_accuracy": territorial_cross_boundary_correct / cross_boundary_total,
        "flat_aliased_state_accuracy": flat_aliased_correct / aliased_total,
        "territorial_aliased_state_accuracy": territorial_aliased_correct / aliased_total,
    }


def _expand_flat_neighbors(
    env: TerritorialGridEnv,
    flat: FlatWorldModel,
    state: GlobalState,
) -> List[Tuple[Action, GlobalState]]:
    territory, x, y = state
    obs = env.observe(state)
    neighbors: List[Tuple[Action, GlobalState]] = []

    for action in env.valid_actions():
        predicted_obs, confidence = flat.predict(obs, action)
        if confidence <= 0.0:
            continue
        nx, ny, _ = predicted_obs
        next_state = (territory, nx, ny)
        if flat.predict_boundary(obs, action) == 1:
            candidates = [1, 2, 0]
            for next_territory in candidates:
                neighbors.append((action, (next_territory, nx, ny)))
        else:
            neighbors.append((action, next_state))
    return neighbors


def _expand_territorial_neighbors(
    env: TerritorialGridEnv,
    territorial: TerritorialWorldModel,
    state: GlobalState,
) -> List[Tuple[Action, GlobalState]]:
    territory = state[0]
    obs = env.observe(state)
    neighbors: List[Tuple[Action, GlobalState]] = []
    for action in env.valid_actions():
        predicted_state, confidence = territorial.predict(territory, obs, action)
        if confidence <= 0.0:
            continue
        next_territory, nx, ny, _ = predicted_state
        neighbors.append((action, (next_territory, nx, ny)))
    return neighbors


def plan_with_model(
    env: TerritorialGridEnv,
    model,
    start: GlobalState,
    max_depth: int,
    territorial: bool,
) -> Optional[List[Action]]:
    frontier = deque([(start, [])])
    visited = {start}

    while frontier:
        state, path = frontier.popleft()
        if env.is_goal(state):
            return path
        if len(path) >= max_depth:
            continue

        if territorial:
            neighbors = _expand_territorial_neighbors(env, model, state)
        else:
            neighbors = _expand_flat_neighbors(env, model, state)

        for action, next_state in neighbors:
            if next_state in visited:
                continue
            visited.add(next_state)
            frontier.append((next_state, path + [action]))
    return None


def execute_plan(env: TerritorialGridEnv, start: GlobalState, plan: Optional[List[Action]]) -> bool:
    if not plan:
        return False
    state = start
    for action in plan:
        state, _ = env.step(state, action)
        if env.is_goal(state):
            return True
    return env.is_goal(state)


def evaluate_planning(
    env: TerritorialGridEnv,
    flat: FlatWorldModel,
    territorial: TerritorialWorldModel,
    num_trials: int,
    max_depth: int,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    starts: List[GlobalState] = []
    for _ in range(num_trials):
        territory = rng.choice(list(env.territories.keys()))
        spec = env.territories[territory]
        starts.append((territory, rng.randrange(spec.size), rng.randrange(spec.size)))

    flat_success = 0
    territorial_success = 0
    flat_plan_length = []
    territorial_plan_length = []

    for start in starts:
        flat_plan = plan_with_model(env, flat, start, max_depth=max_depth, territorial=False)
        territorial_plan = plan_with_model(env, territorial, start, max_depth=max_depth, territorial=True)

        if execute_plan(env, start, flat_plan):
            flat_success += 1
            flat_plan_length.append(len(flat_plan or []))
        if execute_plan(env, start, territorial_plan):
            territorial_success += 1
            territorial_plan_length.append(len(territorial_plan or []))

    return {
        "flat_planning_success": flat_success / num_trials,
        "territorial_planning_success": territorial_success / num_trials,
        "flat_mean_plan_length": float(np.mean(flat_plan_length)) if flat_plan_length else math.nan,
        "territorial_mean_plan_length": float(np.mean(territorial_plan_length)) if territorial_plan_length else math.nan,
    }


def run_experiment(output_dir: Path) -> Dict[str, float]:
    env = TerritorialGridEnv(seed=7)
    train_data = collect_dataset(env, num_episodes=250, steps_per_episode=24, seed=11)
    test_data = collect_dataset(env, num_episodes=80, steps_per_episode=24, seed=29)
    flat, territorial = train_models(train_data)

    metrics = {}
    metrics.update(evaluate_prediction(test_data, flat, territorial))
    metrics.update(evaluate_planning(env, flat, territorial, num_trials=120, max_depth=20, seed=17))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    summary_lines = [
        "Territorial world model prototype",
        f"flat_next_state_accuracy={metrics['flat_next_state_accuracy']:.3f}",
        f"territorial_next_state_accuracy={metrics['territorial_next_state_accuracy']:.3f}",
        f"flat_aliased_state_accuracy={metrics['flat_aliased_state_accuracy']:.3f}",
        f"territorial_aliased_state_accuracy={metrics['territorial_aliased_state_accuracy']:.3f}",
        f"flat_cross_boundary_accuracy={metrics['flat_cross_boundary_accuracy']:.3f}",
        f"territorial_cross_boundary_accuracy={metrics['territorial_cross_boundary_accuracy']:.3f}",
        f"flat_boundary_accuracy={metrics['flat_boundary_accuracy']:.3f}",
        f"territorial_boundary_accuracy={metrics['territorial_boundary_accuracy']:.3f}",
        f"flat_planning_success={metrics['flat_planning_success']:.3f}",
        f"territorial_planning_success={metrics['territorial_planning_success']:.3f}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return metrics
