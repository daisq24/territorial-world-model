from __future__ import annotations

from belief_memory import TerritorialBeliefMemory
from local_controller import CerebellarLocalController
from openai_goal_verifier import OpenAIGoalVerifier
from perception import TerritorialPerception
from stop_verifier import TerritorialStopVerifier
from subgoal_planner import TerritorialSubgoalPlanner
from territorial_graph import TerritorialGraphMemory


STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3
OPPOSITE_TURN = {
    TURN_LEFT: TURN_RIGHT,
    TURN_RIGHT: TURN_LEFT,
}


class NeuralCircuitObjectNavPolicy:
    """HM3D-ready neural-circuit prototype.

    Cortex: estimate local goal value from RGB-D and memory.
    Thalamus: maintain heading, local context, and observation history.
    Basal ganglia / SN: score actions by expected value gain and boundary crossing.
    Cerebellum / IO: recover from loops, oscillation, and poor local geometry.
    """

    def __init__(
        self,
        max_territories: int = 16,
        *,
        use_openai_verifier: bool = False,
        openai_model: str = "gpt-4.1-mini",
        openai_api_key: str | None = None,
    ) -> None:
        self.max_territories = max_territories
        self.perception = TerritorialPerception()
        self.memory = TerritorialBeliefMemory(max_territories=max_territories)
        self.graph = TerritorialGraphMemory()
        self.subgoal_planner = TerritorialSubgoalPlanner()
        self.local_controller = CerebellarLocalController()
        self.stop_verifier = TerritorialStopVerifier()
        self.goal_label: str | None = None
        self.goal_verifier = OpenAIGoalVerifier(
            enabled=use_openai_verifier,
            model=openai_model,
            api_key=openai_api_key,
        )
        self.reset()

    def reset(self) -> None:
        self.memory.reset()
        self.graph.reset()
        self.goal_verifier.reset()

    def set_goal(self, goal_label: str | None) -> None:
        self.goal_label = goal_label

    def act(self, observation: dict) -> int:
        features = self.perception.extract(observation)
        territory_id = self.memory.infer_territory(features.signature_vec, features.goal_score)
        self.memory.observe(territory_id, features.goal_score, features.forward_clearance, features.signature_key)
        self.graph.observe(
            territory_id=territory_id,
            goal_score=features.goal_score,
            boundary_strength=features.boundary_strength,
            forward_clearance=features.forward_clearance,
        )

        should_attempt_stop = self.stop_verifier.should_stop(
            goal_score=features.goal_score,
            stable_goal_score=self.memory.stable_goal_score(),
            goal_variation=self.memory.goal_score_variation(),
            forward_clearance=features.forward_clearance,
            close_range_support=features.close_range_support,
            spatial_focus=features.spatial_focus,
        )
        if should_attempt_stop:
            if self.goal_verifier.available:
                verification = self.goal_verifier.verify(
                    rgb=observation.get("rgb"),
                    goal_label=self.goal_label,
                    signature_key=features.signature_key,
                )
                if verification.should_stop and verification.confidence >= 0.6:
                    self.memory.remember_action(
                        STOP,
                        features.signature_key,
                        turn_actions=(TURN_LEFT, TURN_RIGHT),
                        move_forward=MOVE_FORWARD,
                    )
                    return STOP
            else:
                self.memory.remember_action(
                    STOP,
                    features.signature_key,
                    turn_actions=(TURN_LEFT, TURN_RIGHT),
                    move_forward=MOVE_FORWARD,
                )
                return STOP

        territory_value = self.memory.territory_value(territory_id, features.goal_score)
        graph_context = self.graph.context_for(
            territory_id=territory_id,
            territory_value=territory_value,
            territory_visits=self.memory.territories[territory_id].visits,
            boundary_strength=features.boundary_strength,
        )
        action_scores = self.subgoal_planner.score_actions(
            move_forward=MOVE_FORWARD,
            turn_left=TURN_LEFT,
            turn_right=TURN_RIGHT,
            stop=STOP,
            opposite_turn=OPPOSITE_TURN,
            features=features,
            graph_context=graph_context,
            belief_state=self.memory.state,
        )
        action = max(action_scores, key=action_scores.get)

        if self.local_controller.needs_override(
            action=action,
            move_forward=MOVE_FORWARD,
            turn_left=TURN_LEFT,
            turn_right=TURN_RIGHT,
            action_scores=action_scores,
            forward_clearance=features.forward_clearance,
            recent_actions=self.memory.state.recent_actions,
            stuck_steps=self.memory.state.stuck_steps,
        ):
            action = self.local_controller.recovery_action(
                turn_left=TURN_LEFT,
                turn_right=TURN_RIGHT,
                last_action=self.memory.state.last_action,
                depth_stats=features.depth_stats,
            )

        self.memory.remember_action(
            action,
            features.signature_key,
            turn_actions=(TURN_LEFT, TURN_RIGHT),
            move_forward=MOVE_FORWARD,
        )
        return action
