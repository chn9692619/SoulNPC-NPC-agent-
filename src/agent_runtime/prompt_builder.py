from __future__ import annotations

import json
from typing import Dict, Any


def build_dialogue_prompt(state_after: Dict[str, Any], player_event_text: str, event_primitives: Dict[str, float], memory_context: str, action: str) -> str:
    payload = {
        "角色": state_after.get("name", "艾拉"),
        "人格": state_after.get("personality", "谨慎、敏感、外冷内热"),
        "场景": state_after.get("scene", "边境星港附近的一间旧酒馆"),
        "玩家自然语言事件": player_event_text,
        "系统解析的事件原语权重": event_primitives,
        "检索到的关键记忆": memory_context,
        "当前连续情绪向量": state_after.get("affect", {}),
        "当前关系向量": state_after.get("relationship", {}),
        "派生情绪摘要": state_after.get("derived_mood", "平静"),
        "关系摘要": state_after.get("relationship_stage", "谨慎接触"),
        "当前目标": state_after.get("current_goal", "观察玩家是否可靠"),
        "行为意图": action,
        "任务": "请生成一句自然、克制、符合角色人格、记忆、情绪和行为意图的中文 NPC 台词。不要解释，不要输出 JSON，只输出台词。",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
