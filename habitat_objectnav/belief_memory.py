from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np


@dataclass
class TerritoryPrototype:
    signature: np.ndarray
    visits: int = 0
    best_goal_score: float = -1e9


@dataclass
class TerritorialBeliefState:
    heading_bias: Optional[int] = None
    last_action: Optional[int] = None
    last_signature: Optional[tuple[int, ...]] = None
    recent_signatures: Deque[tuple[int, ...]] = field(default_factory=lambda: deque(maxlen=12))
    recent_actions: Deque[int] = field(default_factory=lambda: deque(maxlen=8))
    recent_goal_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    recent_forward_clearance: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    stuck_steps: int = 0
    territory_counter: Counter[int] = field(default_factory=Counter)
    best_goal_score: float = -1e9


class TerritorialBeliefMemory:
    """Prototype-based territorial memory for single-agent navigation."""

    def __init__(self, max_territories: int = 16) -> None:
        self.max_territories = max_territories
        self.reset()

    def reset(self) -> None:
        self.state = TerritorialBeliefState()
        self.territories: dict[int, TerritoryPrototype] = {}
        self.next_territory_id = 0

    def infer_territory(self, signature_vec: np.ndarray, goal_score: float) -> int:
        if not self.territories:
            return self._allocate_territory(signature_vec, goal_score)

        best_id = None
        best_distance = float("inf")
        for territory_id, prototype in self.territories.items():
            distance = float(np.linalg.norm(signature_vec - prototype.signature))
            if distance < best_distance:
                best_distance = distance
                best_id = territory_id

        if best_id is None or (best_distance > 0.48 and len(self.territories) < self.max_territories):
            return self._allocate_territory(signature_vec, goal_score)

        prototype = self.territories[best_id]
        prototype.signature = 0.85 * prototype.signature + 0.15 * signature_vec
        prototype.visits += 1
        prototype.best_goal_score = max(prototype.best_goal_score, goal_score)
        return best_id

    def observe(self, territory_id: int, goal_score: float, forward_clearance: float, signature_key: tuple[int, ...]) -> None:
        self.state.territory_counter[territory_id] += 1
        self.state.best_goal_score = max(self.state.best_goal_score, goal_score)
        self.state.recent_goal_scores.append(goal_score)
        self.state.recent_forward_clearance.append(forward_clearance)
        self.state.recent_signatures.append(signature_key)

        repeated_observation = sum(1 for sig in self.state.recent_signatures if sig == signature_key)
        low_progress = goal_score < self.state.best_goal_score - 0.02
        blocked = forward_clearance < 0.75
        if repeated_observation >= 3 or (low_progress and blocked):
            self.state.stuck_steps += 1
        else:
            self.state.stuck_steps = max(self.state.stuck_steps - 1, 0)

    def territory_value(self, territory_id: int, goal_score: float) -> float:
        prototype = self.territories[territory_id]
        novelty_bonus = 0.18 / max(prototype.visits, 1)
        memory_bonus = 0.5 * max(prototype.best_goal_score - goal_score, 0.0)
        revisit_penalty = 0.08 * self.state.territory_counter[territory_id]
        return novelty_bonus + memory_bonus - revisit_penalty

    def stable_goal_score(self) -> float:
        if not self.state.recent_goal_scores:
            return 0.0
        recent = list(self.state.recent_goal_scores)[-4:]
        return float(np.mean(recent))

    def goal_score_variation(self) -> float:
        if len(self.state.recent_goal_scores) < 2:
            return 1.0
        recent = np.asarray(list(self.state.recent_goal_scores)[-4:], dtype=np.float32)
        return float(np.std(recent))

    def remember_action(self, action: int, signature_key: tuple[int, ...], turn_actions: tuple[int, int], move_forward: int) -> None:
        self.state.last_action = action
        self.state.recent_actions.append(action)

        if action in turn_actions:
            self.state.heading_bias = action
        elif action == move_forward and self.state.heading_bias in turn_actions:
            pass

        self.state.last_signature = signature_key

    def _allocate_territory(self, signature_vec: np.ndarray, goal_score: float) -> int:
        territory_id = self.next_territory_id
        self.next_territory_id += 1
        self.territories[territory_id] = TerritoryPrototype(
            signature=signature_vec.copy(),
            visits=1,
            best_goal_score=goal_score,
        )
        return territory_id
