from __future__ import annotations

from typing import Dict

import numpy as np


class TerritorialSubgoalPlanner:
    """Score action-level subgoals from territorial context.

    This is still a lightweight planner, but its interface is written as a
    planner rather than a flat policy helper so we can later replace the
    action-scoring logic with doorway selection or graph-level planning.
    """

    def score_actions(
        self,
        *,
        move_forward: int,
        turn_left: int,
        turn_right: int,
        stop: int,
        opposite_turn: dict[int, int],
        features,
        graph_context,
        belief_state,
    ) -> Dict[int, float]:
        depth_stats = features.depth_stats
        goal_score = features.goal_score

        scores = {
            move_forward: self._score_forward(features, graph_context, belief_state),
            turn_left: self._score_turn(turn_left, turn_left, turn_right, opposite_turn, depth_stats, goal_score, graph_context, belief_state),
            turn_right: self._score_turn(turn_right, turn_left, turn_right, opposite_turn, depth_stats, goal_score, graph_context, belief_state),
            stop: -5.0,
        }
        return scores

    def _score_forward(self, features, graph_context, belief_state) -> float:
        depth_stats = features.depth_stats
        progress_bonus = 1.2 * features.goal_score + 0.9 * np.clip(depth_stats["mid"] / 2.5, 0.0, 1.0)
        doorway_bonus = 0.35 * graph_context.doorway_posterior if depth_stats["mid"] > 0.9 else 0.0
        cross_boundary_bonus = graph_context.escape_bias + graph_context.unexplored_boundary_bias
        heading_bonus = 0.2 if belief_state.heading_bias is not None else 0.0
        blocked_penalty = 1.4 if depth_stats["mid"] < 0.6 else 0.0
        loop_penalty = 0.25 if belief_state.last_action is not None and belief_state.stuck_steps >= 2 else 0.0
        return (
            progress_bonus
            + doorway_bonus
            + cross_boundary_bonus
            + graph_context.territory_value
            + heading_bonus
            - blocked_penalty
            - loop_penalty
            - 0.15 * graph_context.return_bias
        )

    def _score_turn(
        self,
        action: int,
        turn_left: int,
        turn_right: int,
        opposite_turn: dict[int, int],
        depth_stats: dict[str, float],
        goal_score: float,
        graph_context,
        belief_state,
    ) -> float:
        if action == turn_left:
            free_side = depth_stats["left"]
            edge_side = depth_stats["left_edge"]
        else:
            free_side = depth_stats["right"]
            edge_side = depth_stats["right_edge"]

        opposite = opposite_turn[action]
        side_bonus = 0.85 * np.clip(free_side / 2.2, 0.0, 1.0)
        edge_bonus = 0.35 * np.clip(edge_side / 2.0, 0.0, 1.0)
        exploration_bonus = 0.22 if belief_state.stuck_steps >= 2 else 0.0
        doorway_search_bonus = 0.25 * graph_context.unexplored_boundary_bias
        region_scan_bonus = 0.18 if graph_context.doorway_posterior < 0.4 and graph_context.escape_bias > 0.0 else 0.0
        anti_oscillation = -0.75 if belief_state.last_action == opposite else 0.0
        heading_bonus = 0.2 if belief_state.heading_bias == action else 0.0
        over_turn_penalty = 0.28 if goal_score > 0.78 else 0.0
        return (
            side_bonus
            + edge_bonus
            + doorway_search_bonus
            + region_scan_bonus
            + 0.35 * graph_context.territory_value
            + exploration_bonus
            + heading_bonus
            + anti_oscillation
            - over_turn_penalty
        )
