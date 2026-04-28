from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


ACTION_NAMES = ["north", "south", "west", "east"]
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_NAMES)}
OPPOSITE_ACTION = {
    ACTION_TO_ID["north"]: ACTION_TO_ID["south"],
    ACTION_TO_ID["south"]: ACTION_TO_ID["north"],
    ACTION_TO_ID["west"]: ACTION_TO_ID["east"],
    ACTION_TO_ID["east"]: ACTION_TO_ID["west"],
}


@dataclass
class NavigationPlan:
    name: str
    room_types: np.ndarray
    territories: np.ndarray
    observations: List[Tuple[int, int, int]]
    neighbors: Dict[int, Dict[int, int]]


class FlatNavModel:
    def __init__(self) -> None:
        self.next_obs_counts: Dict[Tuple[Tuple[int, int, int], int], Counter] = defaultdict(Counter)
        self.obs_to_territory: Dict[Tuple[int, int, int], Counter] = defaultdict(Counter)

    def update(
        self,
        obs: Tuple[int, int, int],
        action: int,
        next_obs: Tuple[int, int, int],
        territory: int,
    ) -> None:
        self.next_obs_counts[(obs, action)][next_obs] += 1
        self.obs_to_territory[obs][territory] += 1

    def predict_next_obs(self, obs: Tuple[int, int, int], action: int) -> Optional[Tuple[int, int, int]]:
        counter = self.next_obs_counts.get((obs, action))
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def infer_territory(self, obs: Tuple[int, int, int]) -> Optional[int]:
        counter = self.obs_to_territory.get(obs)
        if not counter:
            return None
        return counter.most_common(1)[0][0]


class TerritorialNavModel:
    def __init__(self) -> None:
        self.next_state_counts: Dict[Tuple[int, Tuple[int, int, int], int], Counter] = defaultdict(Counter)

    def update(
        self,
        territory: int,
        obs: Tuple[int, int, int],
        action: int,
        next_territory: int,
        next_obs: Tuple[int, int, int],
    ) -> None:
        self.next_state_counts[(territory, obs, action)][(next_territory, next_obs)] += 1

    def predict_next_state(
        self,
        territory: int,
        obs: Tuple[int, int, int],
        action: int,
    ) -> Optional[Tuple[int, Tuple[int, int, int]]]:
        counter = self.next_state_counts.get((territory, obs, action))
        if not counter:
            return None
        return counter.most_common(1)[0][0]


class NeuralCircuitNavModel:
    """A lightweight cortex-thalamus-BG-cerebellum inspired navigator.

    Cortex: state-value map over territorial observations.
    Thalamus: current heading / state integration.
    Basal ganglia + SN: action gating from value improvement and territory transitions.
    Cerebellum + IO: local correction via anti-oscillation and backtracking.
    """

    def __init__(self) -> None:
        self.state_value_sums: Dict[Tuple[int, int, Tuple[int, int, int]], float] = defaultdict(float)
        self.state_value_counts: Dict[Tuple[int, int, Tuple[int, int, int]], int] = defaultdict(int)
        self.territory_value_sums: Dict[Tuple[int, int], float] = defaultdict(float)
        self.territory_value_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self.transition_gain_sums: Dict[Tuple[int, int, int], float] = defaultdict(float)
        self.transition_gain_counts: Dict[Tuple[int, int, int], int] = defaultdict(int)

    def update_state_value(
        self,
        territory: int,
        goal_territory: int,
        obs: Tuple[int, int, int],
        distance: float,
    ) -> None:
        key = (territory, goal_territory, obs)
        self.state_value_sums[key] += distance
        self.state_value_counts[key] += 1
        self.territory_value_sums[(territory, goal_territory)] += distance
        self.territory_value_counts[(territory, goal_territory)] += 1

    def update_transition_gain(
        self,
        territory: int,
        goal_territory: int,
        next_territory: int,
        gain: float,
    ) -> None:
        key = (territory, goal_territory, next_territory)
        self.transition_gain_sums[key] += gain
        self.transition_gain_counts[key] += 1

    def estimate_state_value(
        self,
        territory: int,
        goal_territory: int,
        obs: Tuple[int, int, int],
    ) -> float:
        key = (territory, goal_territory, obs)
        if self.state_value_counts.get(key, 0) > 0:
            return self.state_value_sums[key] / self.state_value_counts[key]
        territory_key = (territory, goal_territory)
        if self.territory_value_counts.get(territory_key, 0) > 0:
            return self.territory_value_sums[territory_key] / self.territory_value_counts[territory_key]
        return 8.0

    def estimate_transition_gain(
        self,
        territory: int,
        goal_territory: int,
        next_territory: int,
    ) -> float:
        key = (territory, goal_territory, next_territory)
        if self.transition_gain_counts.get(key, 0) > 0:
            return self.transition_gain_sums[key] / self.transition_gain_counts[key]
        return 0.0


def _load_train_items() -> Sequence:
    with Path("RPlan/Interface/static/Data/data_train_converted.pkl").open("rb") as f:
        data = pickle.load(f)
    return data["data"]


def _degree_bucket(degree: int) -> int:
    return min(degree, 4)


def _bucketize(value: float, boundaries: np.ndarray) -> int:
    return int(np.searchsorted(boundaries, value, side="right"))


def _compute_geometry_bins(items: Sequence, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    areas: List[float] = []
    aspects: List[float] = []
    for item in items[:limit]:
        box = np.asarray(item.box)
        for x0, y0, x1, y1, _room_type in box:
            width = max(int(x1) - int(x0), 1)
            height = max(int(y1) - int(y0), 1)
            areas.append(float(width * height))
            aspects.append(float(width) / float(height))
    return (
        np.quantile(np.asarray(areas), [0.25, 0.5, 0.75]),
        np.quantile(np.asarray(aspects), [0.33, 0.66]),
    )


def _extract_room_type_profiles(items: Sequence, limit: int) -> np.ndarray:
    room_type_count = 13
    profiles = np.zeros((room_type_count, room_type_count + 5), dtype=np.float64)
    for item in items[:limit]:
        box = np.asarray(item.box)
        edge = np.asarray(item.edge)
        if len(box) == 0:
            continue
        degrees = np.zeros(len(box), dtype=np.int64)
        for src, dst, _etype in edge:
            degrees[src] += 1
            degrees[dst] += 1
            src_type = int(box[src, 4])
            dst_type = int(box[dst, 4])
            profiles[src_type, dst_type] += 1.0
            profiles[dst_type, src_type] += 1.0
        for idx, row in enumerate(box):
            room_type = int(row[4])
            profiles[room_type, 13 + _degree_bucket(int(degrees[idx]))] += 1.0
    sums = profiles.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return profiles / sums


def _kmeans(features: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = features[rng.choice(len(features), size=k, replace=False)].copy()
    assignments = np.zeros(len(features), dtype=np.int64)
    for _ in range(iters):
        distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_assignments = distances.argmin(axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for idx in range(k):
            mask = assignments == idx
            if mask.any():
                centroids[idx] = features[mask].mean(axis=0)
    return assignments


def build_territory_map(train_items: Sequence, limit: int = 20000, k: int = 4) -> Dict[int, int]:
    profiles = _extract_room_type_profiles(train_items, limit=limit)
    assignments = _kmeans(profiles, k=k, seed=7)
    return {room_type: int(assignments[room_type]) for room_type in range(len(assignments))}


def _direction_from_delta(dx: float, dy: float) -> int:
    if abs(dx) >= abs(dy):
        return ACTION_TO_ID["east"] if dx >= 0 else ACTION_TO_ID["west"]
    return ACTION_TO_ID["south"] if dy >= 0 else ACTION_TO_ID["north"]


def convert_plan(
    item,
    territory_map: Dict[int, int],
    area_bins: np.ndarray,
    aspect_bins: np.ndarray,
) -> Optional[NavigationPlan]:
    box = np.asarray(item.box)
    edge = np.asarray(item.edge)
    if len(box) < 4 or len(edge) == 0:
        return None

    centers = np.stack(
        [
            (box[:, 0] + box[:, 2]) / 2.0,
            (box[:, 1] + box[:, 3]) / 2.0,
        ],
        axis=1,
    )
    degrees = np.zeros(len(box), dtype=np.int64)
    raw_neighbors: Dict[int, List[int]] = defaultdict(list)
    for src, dst, _etype in edge:
        raw_neighbors[int(src)].append(int(dst))
        raw_neighbors[int(dst)].append(int(src))
        degrees[int(src)] += 1
        degrees[int(dst)] += 1

    neighbors: Dict[int, Dict[int, int]] = defaultdict(dict)
    for src, dsts in raw_neighbors.items():
        best: Dict[int, Tuple[float, int]] = {}
        for dst in dsts:
            dx = float(centers[dst, 0] - centers[src, 0])
            dy = float(centers[dst, 1] - centers[src, 1])
            action = _direction_from_delta(dx, dy)
            distance = abs(dx) + abs(dy)
            prev = best.get(action)
            if prev is None or distance < prev[0]:
                best[action] = (distance, dst)
        neighbors[src] = {action: dst for action, (_distance, dst) in best.items()}

    observations: List[Tuple[int, int, int]] = []
    room_types = box[:, 4].astype(int)
    territories = np.asarray([territory_map[int(t)] for t in room_types], dtype=int)
    for idx, row in enumerate(box):
        width = max(int(row[2]) - int(row[0]), 1)
        height = max(int(row[3]) - int(row[1]), 1)
        observations.append(
            (
                _bucketize(float(width * height), area_bins),
                _bucketize(float(width) / float(height), aspect_bins),
                _degree_bucket(int(degrees[idx])),
            )
        )

    return NavigationPlan(
        name=str(item.name),
        room_types=room_types,
        territories=territories,
        observations=observations,
        neighbors=dict(neighbors),
    )


def build_navigation_plans(
    items: Sequence,
    territory_map: Dict[int, int],
    area_bins: np.ndarray,
    aspect_bins: np.ndarray,
    limit: int,
) -> List[NavigationPlan]:
    plans: List[NavigationPlan] = []
    for item in items[:limit]:
        plan = convert_plan(item, territory_map, area_bins, aspect_bins)
        if plan is not None:
            plans.append(plan)
    return plans


def train_models(plans: Sequence[NavigationPlan]) -> Tuple[FlatNavModel, TerritorialNavModel]:
    flat = FlatNavModel()
    territorial = TerritorialNavModel()
    for plan in plans:
        for room_idx, action_map in plan.neighbors.items():
            for action, next_room in action_map.items():
                obs = plan.observations[room_idx]
                next_obs = plan.observations[next_room]
                territory = int(plan.territories[room_idx])
                next_territory = int(plan.territories[next_room])
                flat.update(obs, action, next_obs, territory)
                territorial.update(territory, obs, action, next_territory, next_obs)
    return flat, territorial


def train_neural_circuit_model(plans: Sequence[NavigationPlan]) -> NeuralCircuitNavModel:
    model = NeuralCircuitNavModel()
    for plan in plans:
        territory_ids = sorted(set(int(x) for x in plan.territories))
        for goal_territory in territory_ids:
            goal_rooms = [idx for idx, territory in enumerate(plan.territories) if int(territory) == goal_territory]
            if not goal_rooms:
                continue
            distances = _reverse_shortest_distances(plan, goal_rooms)
            for room_idx, obs in enumerate(plan.observations):
                distance = distances[room_idx]
                if distance is None:
                    continue
                territory = int(plan.territories[room_idx])
                model.update_state_value(territory, goal_territory, obs, float(distance))
                for _action, next_room in plan.neighbors.get(room_idx, {}).items():
                    next_distance = distances[next_room]
                    if next_distance is None:
                        continue
                    next_territory = int(plan.territories[next_room])
                    gain = float(distance - next_distance)
                    model.update_transition_gain(territory, goal_territory, next_territory, gain)
    return model


def _plan_actions_flat(
    model: FlatNavModel,
    start_obs: Tuple[int, int, int],
    goal_territory: int,
    max_depth: int,
) -> Optional[List[int]]:
    start_state = start_obs
    frontier = deque([(start_state, [])])
    visited = {start_state}
    while frontier:
        state, path = frontier.popleft()
        inferred_territory = model.infer_territory(state)
        if inferred_territory == goal_territory:
            return path
        if len(path) >= max_depth:
            continue
        for action in range(4):
            next_state = model.predict_next_obs(state, action)
            if next_state is None or next_state in visited:
                continue
            visited.add(next_state)
            frontier.append((next_state, path + [action]))
    return None


def _plan_actions_territorial(
    model: TerritorialNavModel,
    start_territory: int,
    start_obs: Tuple[int, int, int],
    goal_territory: int,
    max_depth: int,
) -> Optional[List[int]]:
    start_state = (start_territory, start_obs)
    frontier = deque([(start_state, [])])
    visited = {start_state}
    while frontier:
        (territory, obs), path = frontier.popleft()
        if territory == goal_territory:
            return path
        if len(path) >= max_depth:
            continue
        for action in range(4):
            next_state = model.predict_next_state(territory, obs, action)
            if next_state is None or next_state in visited:
                continue
            visited.add(next_state)
            frontier.append((next_state, path + [action]))
    return None


def _shortest_path_length(plan: NavigationPlan, start_room: int, goal_territory: int) -> Optional[int]:
    frontier = deque([(start_room, 0)])
    visited = {start_room}
    while frontier:
        room, depth = frontier.popleft()
        if int(plan.territories[room]) == goal_territory:
            return depth
        for next_room in plan.neighbors.get(room, {}).values():
            if next_room in visited:
                continue
            visited.add(next_room)
            frontier.append((next_room, depth + 1))
    return None


def _reverse_shortest_distances(plan: NavigationPlan, goal_rooms: Sequence[int]) -> List[Optional[int]]:
    reversed_neighbors: Dict[int, List[int]] = defaultdict(list)
    for room_idx, action_map in plan.neighbors.items():
        for next_room in action_map.values():
            reversed_neighbors[next_room].append(room_idx)

    distances: List[Optional[int]] = [None] * len(plan.observations)
    frontier = deque()
    for room in goal_rooms:
        distances[room] = 0
        frontier.append(room)
    while frontier:
        room = frontier.popleft()
        current_distance = distances[room]
        for prev_room in reversed_neighbors.get(room, []):
            if distances[prev_room] is not None:
                continue
            distances[prev_room] = current_distance + 1
            frontier.append(prev_room)
    return distances


def _execute_replanning_episode(
    plan: NavigationPlan,
    start_room: int,
    goal_territory: int,
    flat_model: FlatNavModel,
    territorial_model: TerritorialNavModel,
    territorial: bool,
    horizon: int,
) -> bool:
    current_room = start_room
    for _ in range(horizon):
        if int(plan.territories[current_room]) == goal_territory:
            return True
        obs = plan.observations[current_room]
        if territorial:
            action_plan = _plan_actions_territorial(
                territorial_model,
                int(plan.territories[current_room]),
                obs,
                goal_territory,
                max_depth=horizon,
            )
        else:
            action_plan = _plan_actions_flat(
                flat_model,
                obs,
                goal_territory,
                max_depth=horizon,
            )
        if not action_plan:
            return False
        action = action_plan[0]
        next_room = plan.neighbors.get(current_room, {}).get(action)
        if next_room is None:
            return False
        current_room = next_room
    return int(plan.territories[current_room]) == goal_territory


def _score_action_with_neural_circuit(
    model: NeuralCircuitNavModel,
    plan: NavigationPlan,
    current_room: int,
    action: int,
    next_room: int,
    goal_territory: int,
    heading: Optional[int],
    visit_counts: Dict[int, int],
    last_rooms: Sequence[int],
    stuck_steps: int,
) -> float:
    current_territory = int(plan.territories[current_room])
    next_territory = int(plan.territories[next_room])
    current_obs = plan.observations[current_room]
    next_obs = plan.observations[next_room]

    current_value = model.estimate_state_value(current_territory, goal_territory, current_obs)
    next_value = model.estimate_state_value(next_territory, goal_territory, next_obs)
    progress_score = current_value - next_value
    transition_score = 0.6 * model.estimate_transition_gain(current_territory, goal_territory, next_territory)

    heading_score = 0.0
    if heading is not None:
        if action == heading:
            heading_score += 0.35
        elif action == OPPOSITE_ACTION[heading]:
            heading_score -= 0.7

    territory_score = 0.0
    if next_territory == goal_territory:
        territory_score += 1.5
    elif next_territory != current_territory:
        territory_score += 0.4

    revisit_penalty = 0.9 * visit_counts.get(next_room, 0)
    loop_penalty = 0.75 if next_room in last_rooms[-3:] else 0.0
    degree_bucket = next_obs[2]
    cerebellar_bonus = 0.15 if degree_bucket >= 3 and next_room not in last_rooms[-2:] else 0.0
    rethink_bonus = 0.3 if stuck_steps >= 2 and next_room not in last_rooms else 0.0

    return (
        progress_score
        + transition_score
        + heading_score
        + territory_score
        + cerebellar_bonus
        + rethink_bonus
        - revisit_penalty
        - loop_penalty
    )


def _execute_neural_circuit_episode(
    plan: NavigationPlan,
    start_room: int,
    goal_territory: int,
    model: NeuralCircuitNavModel,
    horizon: int,
) -> bool:
    current_room = start_room
    heading: Optional[int] = None
    visit_counts: Dict[int, int] = defaultdict(int)
    recent_rooms: List[int] = [current_room]
    best_value_seen = model.estimate_state_value(
        int(plan.territories[current_room]),
        goal_territory,
        plan.observations[current_room],
    )
    stuck_steps = 0

    for _ in range(horizon):
        visit_counts[current_room] += 1
        if int(plan.territories[current_room]) == goal_territory:
            return True

        candidates = []
        for action, next_room in plan.neighbors.get(current_room, {}).items():
            score = _score_action_with_neural_circuit(
                model=model,
                plan=plan,
                current_room=current_room,
                action=action,
                next_room=next_room,
                goal_territory=goal_territory,
                heading=heading,
                visit_counts=visit_counts,
                last_rooms=recent_rooms,
                stuck_steps=stuck_steps,
            )
            candidates.append((score, action, next_room))
        if not candidates:
            return False

        candidates.sort(key=lambda item: (item[0], -visit_counts.get(item[2], 0)), reverse=True)
        _score, chosen_action, next_room = candidates[0]

        next_value = model.estimate_state_value(
            int(plan.territories[next_room]),
            goal_territory,
            plan.observations[next_room],
        )
        if next_value >= best_value_seen - 1e-6:
            stuck_steps += 1
        else:
            best_value_seen = next_value
            stuck_steps = 0

        heading = chosen_action
        current_room = next_room
        recent_rooms.append(current_room)
        recent_rooms = recent_rooms[-6:]

    return int(plan.territories[current_room]) == goal_territory


def sample_navigation_episodes(
    plans: Sequence[NavigationPlan],
    seed: int,
    max_episodes: int,
) -> List[Tuple[int, int, int, int]]:
    rng = random.Random(seed)
    candidates: List[Tuple[int, int, int, int]] = []
    for plan_idx, plan in enumerate(plans):
        for start_room in range(len(plan.observations)):
            start_territory = int(plan.territories[start_room])
            for goal_territory in sorted(set(int(x) for x in plan.territories)):
                if goal_territory == start_territory:
                    continue
                dist = _shortest_path_length(plan, start_room, goal_territory)
                if dist is None or dist < 2:
                    continue
                horizon = min(max(dist + 2, 4), 10)
                candidates.append((plan_idx, start_room, goal_territory, horizon))
    rng.shuffle(candidates)
    return candidates[:max_episodes]


def evaluate_navigation(
    plans: Sequence[NavigationPlan],
    flat_model: FlatNavModel,
    territorial_model: TerritorialNavModel,
    neural_circuit_model: NeuralCircuitNavModel,
    seed: int = 13,
    max_episodes: int = 1500,
) -> Dict[str, float]:
    episodes = sample_navigation_episodes(plans, seed=seed, max_episodes=max_episodes)
    flat_success = 0
    territorial_success = 0
    neural_circuit_success = 0
    oracle_total = 0
    for plan_idx, start_room, goal_territory, horizon in episodes:
        plan = plans[plan_idx]
        oracle_total += 1
        if _execute_replanning_episode(
            plan,
            start_room,
            goal_territory,
            flat_model,
            territorial_model,
            territorial=False,
            horizon=horizon,
        ):
            flat_success += 1
        if _execute_replanning_episode(
            plan,
            start_room,
            goal_territory,
            flat_model,
            territorial_model,
            territorial=True,
            horizon=horizon,
        ):
            territorial_success += 1
        if _execute_neural_circuit_episode(
            plan,
            start_room,
            goal_territory,
            neural_circuit_model,
            horizon=max(horizon + 2, 6),
        ):
            neural_circuit_success += 1
    total = max(len(episodes), 1)
    return {
        "num_navigation_episodes": float(len(episodes)),
        "flat_navigation_success": flat_success / total,
        "territorial_navigation_success": territorial_success / total,
        "neural_circuit_navigation_success": neural_circuit_success / total,
        "oracle_reachable_fraction": oracle_total / total,
    }


def summarize_territories(territory_map: Dict[int, int]) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for room_type, territory in territory_map.items():
        groups[int(territory)].append(int(room_type))
    return {key: sorted(values) for key, values in sorted(groups.items())}


def run_rplan_navigation_experiment(output_dir: Path) -> Dict[str, float]:
    all_items = list(_load_train_items())
    split_idx = 12000
    train_items = all_items[:split_idx]
    test_items = all_items[split_idx:split_idx + 3000]

    territory_map = build_territory_map(train_items, limit=len(train_items), k=4)
    area_bins, aspect_bins = _compute_geometry_bins(train_items, limit=len(train_items))
    train_plans = build_navigation_plans(train_items, territory_map, area_bins, aspect_bins, limit=len(train_items))
    test_plans = build_navigation_plans(test_items, territory_map, area_bins, aspect_bins, limit=len(test_items))

    flat_model, territorial_model = train_models(train_plans)
    neural_circuit_model = train_neural_circuit_model(train_plans)
    metrics = evaluate_navigation(test_plans, flat_model, territorial_model, neural_circuit_model)
    metrics["num_train_plans"] = float(len(train_plans))
    metrics["num_test_plans"] = float(len(test_plans))
    metrics["area_bins"] = [float(x) for x in area_bins]
    metrics["aspect_bins"] = [float(x) for x in aspect_bins]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rplan_navigation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "rplan_navigation_territory_map.json").write_text(
        json.dumps(summarize_territories(territory_map), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics
