from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import math
import pickle
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


RoomType = int
EdgeType = int
TerritoryId = int
Observation = Tuple[int, int, int]


@dataclass
class Transition:
    current_room_type: RoomType
    current_area_bucket: int
    current_aspect_bucket: int
    current_degree_bucket: int
    edge_type: EdgeType
    next_room_type: RoomType
    current_territory: TerritoryId
    next_territory: TerritoryId


class FlatRoomModel:
    def __init__(self) -> None:
        self.counts: Dict[Tuple[Observation, EdgeType], Counter] = defaultdict(Counter)

    def update(self, obs: Observation, edge_type: EdgeType, next_room_type: RoomType) -> None:
        self.counts[(obs, edge_type)][next_room_type] += 1

    def predict(self, obs: Observation, edge_type: EdgeType) -> RoomType:
        counter = self.counts.get((obs, edge_type))
        if not counter:
            return -1
        return counter.most_common(1)[0][0]


class TerritorialRoomModel:
    def __init__(self) -> None:
        self.counts: Dict[Tuple[TerritoryId, Observation, EdgeType], Counter] = defaultdict(Counter)

    def update(
        self,
        territory: TerritoryId,
        obs: Observation,
        edge_type: EdgeType,
        next_room_type: RoomType,
    ) -> None:
        self.counts[(territory, obs, edge_type)][next_room_type] += 1

    def predict(self, territory: TerritoryId, obs: Observation, edge_type: EdgeType) -> RoomType:
        counter = self.counts.get((territory, obs, edge_type))
        if not counter:
            return -1
        return counter.most_common(1)[0][0]


def _load_rplan_structs(path: Path, key: str) -> Sequence:
    with path.open("rb") as f:
        data = pickle.load(f)
    return data[key]


def _degree_bucket(degree: int) -> int:
    return min(degree, 4)


def _bucketize(value: float, boundaries: np.ndarray) -> int:
    return int(np.searchsorted(boundaries, value, side="right"))


def compute_geometry_bins(items: Sequence, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    areas: List[float] = []
    aspects: List[float] = []
    for item in items[:limit]:
        box = np.asarray(item.box)
        for x0, y0, x1, y1, _room_type in box:
            width = max(int(x1) - int(x0), 1)
            height = max(int(y1) - int(y0), 1)
            areas.append(float(width * height))
            aspects.append(float(width) / float(height))
    area_bins = np.quantile(np.asarray(areas), [0.25, 0.5, 0.75])
    aspect_bins = np.quantile(np.asarray(aspects), [0.33, 0.66])
    return area_bins, aspect_bins


def _extract_room_type_profiles(items: Sequence, limit: int) -> np.ndarray:
    room_type_count = 13
    edge_type_count = 10
    profiles = np.zeros((room_type_count, room_type_count + edge_type_count + 1), dtype=np.float64)

    for item in items[:limit]:
        box = np.asarray(item.box)
        edge = np.asarray(item.edge)
        if len(box) == 0:
            continue

        degrees = np.zeros(len(box), dtype=np.int64)
        for src, dst, etype in edge:
            degrees[src] += 1
            degrees[dst] += 1
            src_type = int(box[src, 4])
            dst_type = int(box[dst, 4])
            profiles[src_type, dst_type] += 1.0
            profiles[dst_type, src_type] += 1.0
            profiles[src_type, room_type_count + int(etype)] += 1.0
            profiles[dst_type, room_type_count + int(etype)] += 1.0

        for idx, row in enumerate(box):
            room_type = int(row[4])
            profiles[room_type, -1] += _degree_bucket(int(degrees[idx]))

    row_sums = profiles.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return profiles / row_sums


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


def build_territory_map(train_items: Sequence, limit: int = 30000, k: int = 4) -> Dict[int, int]:
    profiles = _extract_room_type_profiles(train_items, limit=limit)
    assignments = _kmeans(profiles, k=k, seed=7)
    return {room_type: int(assignments[room_type]) for room_type in range(len(assignments))}


def _plan_to_transitions(
    item,
    territory_map: Dict[int, int],
    area_bins: np.ndarray,
    aspect_bins: np.ndarray,
) -> List[Transition]:
    box = np.asarray(item.box)
    edge = np.asarray(item.edge)
    if len(box) == 0 or len(edge) == 0:
        return []

    degrees = np.zeros(len(box), dtype=np.int64)
    for src, dst, _ in edge:
        degrees[src] += 1
        degrees[dst] += 1

    transitions: List[Transition] = []
    for src, dst, etype in edge:
        src_type = int(box[src, 4])
        dst_type = int(box[dst, 4])
        src_w = max(int(box[src, 2]) - int(box[src, 0]), 1)
        src_h = max(int(box[src, 3]) - int(box[src, 1]), 1)
        dst_w = max(int(box[dst, 2]) - int(box[dst, 0]), 1)
        dst_h = max(int(box[dst, 3]) - int(box[dst, 1]), 1)
        src_obs = (
            _bucketize(float(src_w * src_h), area_bins),
            _bucketize(float(src_w) / float(src_h), aspect_bins),
            _degree_bucket(int(degrees[src])),
        )
        dst_obs = (
            _bucketize(float(dst_w * dst_h), area_bins),
            _bucketize(float(dst_w) / float(dst_h), aspect_bins),
            _degree_bucket(int(degrees[dst])),
        )
        transitions.append(
            Transition(
                current_room_type=src_type,
                current_area_bucket=src_obs[0],
                current_aspect_bucket=src_obs[1],
                current_degree_bucket=src_obs[2],
                edge_type=int(etype),
                next_room_type=dst_type,
                current_territory=territory_map[src_type],
                next_territory=territory_map[dst_type],
            )
        )
        transitions.append(
            Transition(
                current_room_type=dst_type,
                current_area_bucket=dst_obs[0],
                current_aspect_bucket=dst_obs[1],
                current_degree_bucket=dst_obs[2],
                edge_type=int(etype),
                next_room_type=src_type,
                current_territory=territory_map[dst_type],
                next_territory=territory_map[src_type],
            )
        )
    return transitions


def collect_transitions(
    items: Sequence,
    territory_map: Dict[int, int],
    area_bins: np.ndarray,
    aspect_bins: np.ndarray,
    limit: int,
) -> List[Transition]:
    transitions: List[Transition] = []
    for item in items[:limit]:
        transitions.extend(_plan_to_transitions(item, territory_map, area_bins, aspect_bins))
    return transitions


def train_models(transitions: Sequence[Transition]) -> Tuple[FlatRoomModel, TerritorialRoomModel]:
    flat = FlatRoomModel()
    territorial = TerritorialRoomModel()
    for tr in transitions:
        obs = (tr.current_area_bucket, tr.current_aspect_bucket, tr.current_degree_bucket)
        flat.update(obs, tr.edge_type, tr.next_room_type)
        territorial.update(tr.current_territory, obs, tr.edge_type, tr.next_room_type)
    return flat, territorial


def evaluate_models(transitions: Sequence[Transition], flat: FlatRoomModel, territorial: TerritorialRoomModel) -> Dict[str, float]:
    flat_correct = 0
    territorial_correct = 0
    alias_total = 0
    flat_alias_correct = 0
    territorial_alias_correct = 0

    supports: Dict[Tuple[Observation, EdgeType], set] = defaultdict(set)
    for tr in transitions:
        obs = (tr.current_area_bucket, tr.current_aspect_bucket, tr.current_degree_bucket)
        supports[(obs, tr.edge_type)].add(tr.next_room_type)
    alias_keys = {k for k, v in supports.items() if len(v) > 1}

    for tr in transitions:
        obs = (tr.current_area_bucket, tr.current_aspect_bucket, tr.current_degree_bucket)
        flat_pred = flat.predict(obs, tr.edge_type)
        terr_pred = territorial.predict(tr.current_territory, obs, tr.edge_type)
        if flat_pred == tr.next_room_type:
            flat_correct += 1
        if terr_pred == tr.next_room_type:
            territorial_correct += 1
        if (obs, tr.edge_type) in alias_keys:
            alias_total += 1
            if flat_pred == tr.next_room_type:
                flat_alias_correct += 1
            if terr_pred == tr.next_room_type:
                territorial_alias_correct += 1

    n = max(len(transitions), 1)
    alias_total = max(alias_total, 1)
    return {
        "num_transitions": float(len(transitions)),
        "flat_next_room_accuracy": flat_correct / n,
        "territorial_next_room_accuracy": territorial_correct / n,
        "flat_aliased_transition_accuracy": flat_alias_correct / alias_total,
        "territorial_aliased_transition_accuracy": territorial_alias_correct / alias_total,
        "aliased_transition_fraction": alias_total / n,
    }


def summarize_territories(territory_map: Dict[int, int]) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for room_type, territory in territory_map.items():
        groups[int(territory)].append(int(room_type))
    return {key: sorted(values) for key, values in sorted(groups.items())}


def run_rplan_experiment(output_dir: Path) -> Dict[str, float]:
    all_items = list(_load_rplan_structs(Path("RPlan/Interface/static/Data/data_train_converted.pkl"), "data"))
    split_idx = 12000
    train_items = all_items[:split_idx]
    test_items = all_items[split_idx:split_idx + 3000]

    territory_map = build_territory_map(train_items, limit=len(train_items), k=4)
    area_bins, aspect_bins = compute_geometry_bins(train_items, limit=len(train_items))
    train_transitions = collect_transitions(
        train_items,
        territory_map,
        area_bins,
        aspect_bins,
        limit=len(train_items),
    )
    test_transitions = collect_transitions(
        test_items,
        territory_map,
        area_bins,
        aspect_bins,
        limit=len(test_items),
    )
    flat, territorial = train_models(train_transitions)
    metrics = evaluate_models(test_transitions, flat, territorial)
    metrics["num_train_plans"] = float(len(train_items))
    metrics["num_test_plans"] = float(len(test_items))
    metrics["area_bins"] = [float(x) for x in area_bins]
    metrics["aspect_bins"] = [float(x) for x in aspect_bins]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rplan_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "rplan_territory_map.json").write_text(
        json.dumps(summarize_territories(territory_map), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics
