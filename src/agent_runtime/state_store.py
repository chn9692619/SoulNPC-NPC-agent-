from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from src.character.cognitive_state import CharacterState, AffectVector, RelationshipVector
from src.character.event_model import compose_effect


def state_from_dict(obj: Dict[str, Any] | None) -> CharacterState:
    s = CharacterState()
    if not obj:
        return s
    s.name = obj.get("name", s.name)
    s.personality = obj.get("personality", s.personality)
    s.scene = obj.get("scene", s.scene)
    s.current_goal = obj.get("current_goal", s.current_goal)
    s.long_term_goal = obj.get("long_term_goal", s.long_term_goal)
    aff = obj.get("affect", {})
    rel = obj.get("relationship", {})
    for k, v in aff.items():
        if hasattr(s.affect, k):
            setattr(s.affect, k, float(v))
    for k, v in rel.items():
        if hasattr(s.relationship, k):
            setattr(s.relationship, k, float(v))
    return s


class RuntimeStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> CharacterState:
        if not self.path.exists():
            return CharacterState()
        return state_from_dict(json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, state: CharacterState):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def reset(self) -> CharacterState:
        if self.path.exists():
            self.path.unlink()
        state = CharacterState()
        self.save(state)
        return state


def apply_primitives_to_state(state: CharacterState, primitives: Dict[str, float], momentum: float = 0.65):
    before = state.to_dict()
    effect = compose_effect(primitives)
    state.affect.update(effect.get("affect", {}), momentum=momentum)
    state.relationship.update(effect.get("relationship", {}))
    after = state.to_dict()
    return before, after, effect
