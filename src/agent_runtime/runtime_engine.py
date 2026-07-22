from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from src.agent_runtime.event_parser import parse_event_primitives, explain_primitives
from src.agent_runtime.persistent_memory import PersistentMemoryRAG
from src.agent_runtime.state_store import RuntimeStateStore, apply_primitives_to_state
from src.agent_runtime.prompt_builder import build_dialogue_prompt
from src.character.policy import choose_action, generate_rule_dialogue


def estimate_memory_weights(primitives: Dict[str, float], before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
    primitive_strength = min(1.0, sum(abs(v) for v in primitives.values()) / 2.5)
    rel_before = before.get("relationship", {})
    rel_after = after.get("relationship", {})
    aff_before = before.get("affect", {})
    aff_after = after.get("affect", {})
    rel_change = sum(abs(float(rel_after.get(k, 0)) - float(rel_before.get(k, 0))) for k in set(rel_before) | set(rel_after))
    aff_change = sum(abs(float(aff_after.get(k, 0)) - float(aff_before.get(k, 0))) for k in set(aff_before) | set(aff_after))
    return {
        "importance": round(max(0.1, min(1.0, 0.25 + primitive_strength * 0.35 + rel_change * 0.80)), 3),
        "emotional_salience": round(max(0.1, min(1.0, 0.25 + aff_change * 1.10 + primitive_strength * 0.25)), 3),
        "relation_impact": round(max(0.1, min(1.0, 0.25 + rel_change * 1.20)), 3),
    }


class SoulNPCRuntime:
    def __init__(self, runtime_dir: str | Path):
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.state_store = RuntimeStateStore(self.runtime_dir / "agent_state.json")
        self.memory = PersistentMemoryRAG(self.runtime_dir / "memories.jsonl")

    def reset(self):
        state = self.state_store.reset()
        self.memory.reset()
        return state.to_dict()

    def step(self, player_event_text: str, use_memory: bool = True, write_memory: bool = True, top_k: int = 5) -> Dict[str, Any]:
        state = self.state_store.load()
        primitives = parse_event_primitives(player_event_text)
        retrieved = self.memory.retrieve(player_event_text, primitives, top_k=top_k) if use_memory else []
        memory_context = self.memory.context_text(retrieved)
        before, after, effect = apply_primitives_to_state(state, primitives)
        action = choose_action(after, primitives)
        rule_dialogue = generate_rule_dialogue(action, after)
        weights = estimate_memory_weights(primitives, before, after)
        new_memory = None
        if write_memory:
            new_memory = self.memory.add(
                content=f"玩家事件：{player_event_text}；系统解析：{', '.join([k + '=' + str(round(v,2)) for k,v in primitives.items()])}；角色反应倾向：{action}",
                event_primitives=primitives,
                importance=weights["importance"],
                emotional_salience=weights["emotional_salience"],
                relation_impact=weights["relation_impact"],
                pinned=weights["importance"] > 0.86 or weights["relation_impact"] > 0.86,
            )
        self.state_store.save(state)
        lora_prompt = build_dialogue_prompt(after, player_event_text, primitives, memory_context, action)
        return {
            "player_event_text": player_event_text,
            "event_primitives": primitives,
            "event_explanation": explain_primitives(primitives),
            "retrieved_memories": [(score, item.to_dict()) for score, item in retrieved],
            "memory_context": memory_context,
            "state_before": before,
            "state_after": after,
            "effect": effect,
            "action": action,
            "rule_dialogue": rule_dialogue,
            "new_memory": new_memory.to_dict() if new_memory else None,
            "memory_weights": weights,
            "lora_prompt": lora_prompt,
        }
