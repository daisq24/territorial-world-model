from __future__ import annotations


class CerebellarLocalController:
    """Apply local corrections when the high-level choice is unsafe or unstable."""

    def needs_override(
        self,
        *,
        action: int,
        move_forward: int,
        turn_left: int,
        turn_right: int,
        action_scores: dict[int, float],
        forward_clearance: float,
        recent_actions,
        stuck_steps: int,
    ) -> bool:
        if stuck_steps >= 4:
            return True
        if action == move_forward and forward_clearance < 0.55:
            return True
        if len(recent_actions) >= 4 and list(recent_actions)[-4:] == [turn_left, turn_right, turn_left, turn_right]:
            return True
        best_turn = max(action_scores[turn_left], action_scores[turn_right])
        if action == move_forward and best_turn > action_scores[move_forward] + 0.35:
            return True
        return False

    def recovery_action(
        self,
        *,
        turn_left: int,
        turn_right: int,
        last_action: int | None,
        depth_stats: dict[str, float],
    ) -> int:
        left_score = depth_stats["left"] + 0.4 * depth_stats["left_edge"]
        right_score = depth_stats["right"] + 0.4 * depth_stats["right_edge"]

        if last_action == turn_left:
            left_score -= 0.3
        elif last_action == turn_right:
            right_score -= 0.3

        return turn_left if left_score >= right_score else turn_right
