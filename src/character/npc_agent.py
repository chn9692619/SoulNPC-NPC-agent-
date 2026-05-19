import yaml
from pathlib import Path
from typing import Dict, Any

from .schemas import CharacterState, MemoryItem, AgentTurnResult
from .rules import apply_event_rule
from .memory import MemoryBank, summarize_event
from .dialogue import generate_rule_dialogue
from .labels import format_state_zh


class SoulNPC:
    def __init__(self, config_path: str = "configs/character_ella.yaml"):
        self.config_path = Path(config_path)
        self.profile = self._load_profile(self.config_path)
        self.name = self.profile.get("name", "NPC")
        self.state = CharacterState.from_dict(self.profile.get("default_state", {}))
        self.memory = MemoryBank(max_items=20)

    def _load_profile(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def process_event(self, player_event: str, player_text: str = "") -> AgentTurnResult:
        evaluation = apply_event_rule(self.state, player_event)
        memory_summary = summarize_event(player_event, player_text, evaluation["emotion"])
        memory_item = MemoryItem(
            event_type=player_event,
            summary=memory_summary,
            importance=evaluation["importance"],
            emotion=evaluation["emotion"],
        )
        self.memory.add(memory_item)

        dialogue = generate_rule_dialogue(
            action=evaluation["action"],
            state_after=evaluation["state_after"],
            player_text=player_text,
        )

        return AgentTurnResult(
            npc_name=self.name,
            player_event=player_event,
            player_text=player_text,
            state_before=evaluation["state_before"],
            state_after=evaluation["state_after"],
            emotion=evaluation["emotion"],
            action=evaluation["action"],
            memory_added=memory_item.to_dict(),
            npc_dialogue=dialogue,
            debug={"deltas": evaluation["deltas"], "recent_memory": self.memory.recent_text()},
        )

    def reset(self) -> None:
        self.state = CharacterState.from_dict(self.profile.get("default_state", {}))
        self.memory = MemoryBank(max_items=20)

    def state_text(self) -> str:
        return format_state_zh(self.state.to_dict())
