from __future__ import annotations


class TerritorialStopVerifier:
    """Decide when current evidence is strong enough to stop."""

    def should_stop(
        self,
        *,
        goal_score: float,
        stable_goal_score: float,
        goal_variation: float,
        forward_clearance: float,
        close_range_support: float,
        spatial_focus: float,
    ) -> bool:
        if goal_score > 0.975 and close_range_support > 0.72:
            return True
        if stable_goal_score > 0.935 and goal_variation < 0.035 and close_range_support > 0.62 and forward_clearance < 1.15:
            return True
        if goal_score > 0.945 and spatial_focus > 0.58 and close_range_support > 0.58 and forward_clearance < 0.9:
            return True
        return False
