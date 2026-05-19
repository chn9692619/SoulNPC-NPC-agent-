from copy import deepcopy
from typing import Dict, Any

from .schemas import CharacterState


EVENT_RULES: Dict[str, Dict[str, Any]] = {
    "player_helped_npc": {
        "emotion": "relieved",
        "trust_delta": 0.10,
        "affection_delta": 0.06,
        "stress_delta": -0.08,
        "distance_delta": -0.06,
        "action": "offer_warm_thanks",
        "importance": "medium",
    },
    "player_broke_promise": {
        "emotion": "disappointed",
        "trust_delta": -0.18,
        "affection_delta": -0.04,
        "stress_delta": 0.10,
        "distance_delta": 0.08,
        "action": "ask_for_explanation",
        "importance": "high",
    },
    "player_gave_gift": {
        "emotion": "softened",
        "trust_delta": 0.06,
        "affection_delta": 0.10,
        "stress_delta": -0.04,
        "distance_delta": -0.05,
        "action": "respond_shyly",
        "importance": "medium",
    },
    "player_lied": {
        "emotion": "suspicious",
        "trust_delta": -0.22,
        "affection_delta": -0.06,
        "stress_delta": 0.12,
        "distance_delta": 0.10,
        "action": "test_player",
        "importance": "high",
    },
    "player_returned_after_long_absence": {
        "emotion": "guarded",
        "trust_delta": -0.05,
        "affection_delta": -0.02,
        "stress_delta": 0.05,
        "distance_delta": 0.04,
        "action": "ask_why_returned",
        "importance": "medium",
    },
    "player_asked_personal_question": {
        "emotion": "uneasy",
        "trust_delta": -0.02,
        "affection_delta": 0.00,
        "stress_delta": 0.08,
        "distance_delta": 0.05,
        "action": "set_boundary",
        "importance": "medium",
    },
    "player_protected_npc": {
        "emotion": "moved",
        "trust_delta": 0.18,
        "affection_delta": 0.12,
        "stress_delta": -0.10,
        "distance_delta": -0.12,
        "action": "reveal_small_secret",
        "importance": "high",
    },
    "player_ignored_npc": {
        "emotion": "hurt",
        "trust_delta": -0.08,
        "affection_delta": -0.08,
        "stress_delta": 0.05,
        "distance_delta": 0.08,
        "action": "withdraw",
        "importance": "medium",
    },
    "player_apologized": {
        "emotion": "cautiously_receptive",
        "trust_delta": 0.06,
        "affection_delta": 0.04,
        "stress_delta": -0.05,
        "distance_delta": -0.03,
        "action": "accept_partially",
        "importance": "medium",
    },
    "player_completed_quest": {
        "emotion": "grateful",
        "trust_delta": 0.14,
        "affection_delta": 0.08,
        "stress_delta": -0.08,
        "distance_delta": -0.08,
        "action": "offer_new_lead",
        "importance": "high",
    },
    "player_failed_quest": {
        "emotion": "concerned",
        "trust_delta": -0.04,
        "affection_delta": 0.00,
        "stress_delta": 0.10,
        "distance_delta": 0.02,
        "action": "assess_damage",
        "importance": "high",
    },
}


def update_relationship_stage(state: CharacterState) -> str:
    if state.trust >= 0.78 and state.affection >= 0.60:
        return "trusted_partner"
    if state.trust >= 0.62:
        return "cautious_ally"
    if state.trust <= 0.25:
        return "distrusted"
    return "stranger"


def choose_goal(state: CharacterState, action: str) -> str:
    if state.stress >= 0.70:
        return "protect_self"
    if action in {"test_player", "ask_for_explanation", "ask_why_returned"}:
        return "evaluate_player_reliability"
    if action in {"reveal_small_secret", "offer_new_lead"}:
        return "share_controlled_information"
    if state.curiosity >= 0.70:
        return "learn_more_about_player"
    return "maintain_safe_distance"


def apply_event_rule(state: CharacterState, event_type: str) -> Dict[str, Any]:
    if event_type not in EVENT_RULES:
        event_type = "player_helped_npc"

    rule = EVENT_RULES[event_type]
    before = deepcopy(state).to_dict()

    state.trust += rule.get("trust_delta", 0.0)
    state.affection += rule.get("affection_delta", 0.0)
    state.stress += rule.get("stress_delta", 0.0)
    state.distance += rule.get("distance_delta", 0.0)
    state.mood = rule["emotion"]
    state.relationship_stage = update_relationship_stage(state)
    state.current_goal = choose_goal(state, rule["action"])
    state.clamp()

    after = state.to_dict()

    return {
        "event_type": event_type,
        "state_before": before,
        "state_after": after,
        "emotion": rule["emotion"],
        "action": rule["action"],
        "importance": rule["importance"],
        "deltas": {
            "trust_delta": rule.get("trust_delta", 0.0),
            "affection_delta": rule.get("affection_delta", 0.0),
            "stress_delta": rule.get("stress_delta", 0.0),
            "distance_delta": rule.get("distance_delta", 0.0),
        },
    }
