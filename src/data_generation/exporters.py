import json
from pathlib import Path
from typing import List, Dict, Any

SYSTEM_PROMPT = "你是 SoulNPC 的角色化语言模型。你需要根据角色人格、事件、情绪状态、关系状态、记忆和行为意图，生成自然、克制、符合人设的中文 NPC 台词。"


def export_sft(samples: List[Dict[str, Any]], path: str) -> str:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for s in samples:
            out = s.get("reviewed_dialogue") or s.get("dialogue") or s.get("sft_output", {}).get("dialogue", "")
            record = {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(s["sft_prompt"], ensure_ascii=False)},
                {"role": "assistant", "content": out}
            ], "metadata": {"id": s.get("id"), "action": s.get("action"), "mood": s.get("derived_mood")}}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(p)


def export_dpo(samples: List[Dict[str, Any]], path: str) -> str:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for s in samples:
            chosen = s.get("reviewed_dialogue") or s.get("dialogue")
            rejected = s.get("rejected_dialogue") or "这不重要，我没有什么想说的。"
            record = {
                "prompt": json.dumps({"system": SYSTEM_PROMPT, "user": s["sft_prompt"]}, ensure_ascii=False),
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {"id": s.get("id"), "action": s.get("action"), "mood": s.get("derived_mood")}
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(p)


def merge_raw_with_overrides(raw: List[Dict[str, Any]], overrides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    override_map = {o.get("id"): o for o in overrides if o.get("id")}
    merged = []
    for s in raw:
        oid = s.get("id")
        if oid in override_map:
            o = override_map[oid]
            s = dict(s)
            if o.get("reviewed_dialogue"):
                s["reviewed_dialogue"] = o["reviewed_dialogue"]
            if o.get("chosen"):
                s["dialogue"] = o["chosen"]
            if o.get("rejected"):
                s["rejected_dialogue"] = o["rejected"]
            s["human_reviewed"] = True
        merged.append(s)
    return merged
