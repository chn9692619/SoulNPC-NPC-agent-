from __future__ import annotations
import random, json
from pathlib import Path
from typing import Dict, Any, List
from src.character.cognitive_state import CharacterState
from src.character.event_model import random_complex_event, compose_effect
from src.character.policy import choose_action, generate_rule_dialogue


def random_state(rng: random.Random) -> CharacterState:
    s = CharacterState()
    s.affect.valence = rng.uniform(0.32, 0.72)
    s.affect.arousal = rng.uniform(0.20, 0.70)
    s.affect.dominance = rng.uniform(0.20, 0.70)
    s.affect.safety = rng.uniform(0.25, 0.75)
    s.affect.stress = rng.uniform(0.08, 0.65)
    s.affect.curiosity = rng.uniform(0.15, 0.80)
    s.relationship.trust = rng.uniform(0.20, 0.80)
    s.relationship.intimacy = rng.uniform(0.05, 0.65)
    s.relationship.dependence = rng.uniform(0.05, 0.55)
    s.relationship.conflict = rng.uniform(0.00, 0.55)
    s.relationship.boundary = rng.uniform(0.25, 0.85)
    s.relationship.commitment = rng.uniform(0.05, 0.60)
    return s


def build_sample(i: int, rng: random.Random) -> Dict[str, Any]:
    state = random_state(rng)
    event_name, primitives, player_text = random_complex_event(rng)
    state_before = state.to_dict()
    effect = compose_effect(primitives)
    state.affect.update(effect["affect"], momentum=0.65)  # 样本生成阶段让变化更明显
    state.relationship.update(effect["relationship"])
    state_after = state.to_dict()
    action = choose_action(state_after, primitives)
    dialogue = generate_rule_dialogue(action, state_after)
    importance = min(1.0, 0.25 + sum(abs(v) for v in effect["relationship"].values()) + sum(abs(v) for v in effect["affect"].values())*0.5)
    sample_id = f"sample_{i:06d}"
    prompt = {
        "角色": state_after["name"],
        "人格": state_after["personality"],
        "场景": state_after["scene"],
        "事件": event_name,
        "事件原语权重": primitives,
        "玩家台词": player_text,
        "事件前状态": state_before,
        "事件后状态": state_after,
        "行为意图": action,
        "关键记忆": f"玩家事件：{event_name}；玩家说：{player_text}",
        "任务": "请生成一句符合角色人格、情绪、记忆和行为意图的中文 NPC 台词。"
    }
    sft_output = {
        "dialogue": dialogue,
        "action": action,
        "derived_mood": state_after["derived_mood"],
        "relationship_stage": state_after["relationship_stage"]
    }
    rejected = make_rejected(dialogue, action, state_after)
    return {
        "id": sample_id,
        "event_name": event_name,
        "event_primitives": primitives,
        "player_text": player_text,
        "state_before": state_before,
        "state_after": state_after,
        "derived_mood": state_after["derived_mood"],
        "action": action,
        "importance": importance,
        "dialogue": dialogue,
        "rejected_dialogue": rejected,
        "sft_prompt": prompt,
        "sft_output": sft_output,
    }


def make_rejected(dialogue: str, action: str, state_after: Dict[str, Any]) -> str:
    mood = state_after.get("derived_mood", "平静")
    if mood in ["失望", "受伤", "警惕", "紧张"]:
        return "没关系，我完全不在意。我们继续像以前一样吧。"
    if action in ["表达感谢", "透露小秘密"]:
        return "这件事对我没有任何意义。你不需要再提。"
    return "我没有任何感觉，也不需要回应。"


def generate_samples(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    return [build_sample(i, rng) for i in range(n)]


def write_jsonl(samples: List[Dict[str, Any]], path: str) -> str:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return str(p)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows
