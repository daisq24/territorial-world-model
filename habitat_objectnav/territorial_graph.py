from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class TerritoryNode:
    visits: int = 0
    doorway_evidence: float = 0.0
    best_goal_score: float = 0.0
    last_goal_score: float = 0.0


@dataclass
class TransitionEdge:
    crossings: int = 0
    cumulative_goal_delta: float = 0.0
    cumulative_boundary_strength: float = 0.0

    @property
    def mean_goal_delta(self) -> float:
        if self.crossings == 0:
            return 0.0
        return self.cumulative_goal_delta / self.crossings

    @property
    def mean_boundary_strength(self) -> float:
        if self.crossings == 0:
            return 0.0
        return self.cumulative_boundary_strength / self.crossings


@dataclass
class TerritorialContext:
    territory_id: int
    territory_value: float
    novelty_bonus: float
    revisit_penalty: float
    boundary_strength: float
    doorway_posterior: float
    unexplored_boundary_bias: float
    escape_bias: float
    return_bias: float
    transition_confidence: float
    outgoing_edges: int


class TerritorialGraphMemory:
    """Explicit region-transition memory for territory-aware navigation.

    The current graph is still lightweight, but it now keeps region nodes and
    transition edges instead of acting as a pure placeholder. That lets the
    planner reason about:

    - whether the agent is near a doorway-like boundary
    - whether the current territory appears exhausted
    - whether previous crossings tended to improve or worsen goal progress
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.nodes: dict[int, TerritoryNode] = {}
        self.edges: dict[int, dict[int, TransitionEdge]] = defaultdict(dict)
        self.recent_territories: deque[int] = deque(maxlen=8)
        self.last_territory_id: int | None = None
        self.last_goal_score: float | None = None

    def observe(
        self,
        *,
        territory_id: int,
        goal_score: float,
        boundary_strength: float,
        forward_clearance: float,
    ) -> None:
        node = self.nodes.setdefault(territory_id, TerritoryNode())
        node.visits += 1
        node.best_goal_score = max(node.best_goal_score, goal_score)
        node.last_goal_score = goal_score

        doorway_evidence = boundary_strength * min(max(forward_clearance / 1.8, 0.0), 1.0)
        node.doorway_evidence = 0.75 * node.doorway_evidence + 0.25 * doorway_evidence

        if self.last_territory_id is not None and self.last_territory_id != territory_id:
            edge = self.edges[self.last_territory_id].setdefault(territory_id, TransitionEdge())
            edge.crossings += 1
            previous_goal = 0.0 if self.last_goal_score is None else self.last_goal_score
            edge.cumulative_goal_delta += goal_score - previous_goal
            edge.cumulative_boundary_strength += boundary_strength

        self.recent_territories.append(territory_id)
        self.last_territory_id = territory_id
        self.last_goal_score = goal_score

    def context_for(
        self,
        *,
        territory_id: int,
        territory_value: float,
        territory_visits: int,
        boundary_strength: float,
    ) -> TerritorialContext:
        node = self.nodes.setdefault(territory_id, TerritoryNode())
        novelty_bonus = 0.18 / max(territory_visits, 1)
        revisit_penalty = 0.08 * territory_visits
        doorway_posterior = 0.55 * boundary_strength + 0.45 * node.doorway_evidence

        outgoing = self.edges.get(territory_id, {})
        positive_edges = sum(1 for edge in outgoing.values() if edge.mean_goal_delta > 0.0)
        negative_edges = sum(1 for edge in outgoing.values() if edge.mean_goal_delta < 0.0)
        outgoing_edges = len(outgoing)

        if outgoing_edges == 0:
            transition_confidence = 0.0
            return_bias = 0.0
        else:
            transition_confidence = min(outgoing_edges / 3.0, 1.0)
            return_bias = min(negative_edges / max(outgoing_edges, 1), 1.0)

        exhausted_region = territory_visits >= 3 and node.best_goal_score < 0.8
        unexplored_boundary_bias = 0.25 if doorway_posterior > 0.45 and outgoing_edges == 0 else 0.0
        escape_bias = 0.18 if exhausted_region else 0.0
        if positive_edges > 0:
            escape_bias += 0.08 * min(positive_edges, 2)

        return TerritorialContext(
            territory_id=territory_id,
            territory_value=territory_value,
            novelty_bonus=novelty_bonus,
            revisit_penalty=revisit_penalty,
            boundary_strength=boundary_strength,
            doorway_posterior=doorway_posterior,
            unexplored_boundary_bias=unexplored_boundary_bias,
            escape_bias=escape_bias,
            return_bias=return_bias,
            transition_confidence=transition_confidence,
            outgoing_edges=outgoing_edges,
        )
