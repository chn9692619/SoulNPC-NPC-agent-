from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any


@dataclass
class CharacterState:
    mood: str = "calm"
    trust: float = 0.50
    affection: float = 0.30
    stress: float = 0.20
    curiosity: float = 0.40
    distance: float = 0.60
    current_goal: str = "observe_player"
    relationship_stage: str = "stranger"

    def clamp(self) -> None:
        for key in ["trust", "affection", "stress", "curiosity", "distance"]:
            value = getattr(self, key)
            setattr(self, key, max(0.0, min(1.0, float(value))))

    def to_dict(self) -> Dict[str, Any]:
        self.clamp()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MemoryItem:
    event_type: str
    summary: str
    importance: str = "medium"
    emotion: str = "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentTurnResult:
    npc_name: str
    player_event: str
    player_text: str
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    emotion: str
    action: str
    memory_added: Dict[str, Any]
    npc_dialogue: str
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
