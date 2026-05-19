import json
import re
from pathlib import Path
from typing import Dict, Any, List

from src.llm.client import LLMClient
from src.character.labels import zh_emotion, zh_action


TEACHER_CANDIDATES_PATH = "data/processed/teacher_candidates.jsonl"


def build_teacher_prompt(sample: Dict[str, Any], num_candidates: int = 3) -> str:
    prompt = sample.get("sft_prompt", {})
    return f"""
你正在为 SoulNPC 生成角色台词候选。

角色与场景：
- 角色：{prompt.get('character', '艾拉')}
- 人格：{prompt.get('personality', '')}
- 场景：{prompt.get('scene', '')}

当前事件：
- 玩家事件：{prompt.get('player_event_zh', prompt.get('player_event', ''))}
- 玩家台词：{prompt.get('player_text', '')}

角色状态：
- 事件前状态：{json.dumps(prompt.get('state_before', {}), ensure_ascii=False)}
- 事件后状态：{json.dumps(sample.get('state_after', {}), ensure_ascii=False)}
- 当前情绪：{zh_emotion(sample.get('emotion', ''))}
- 行为意图：{zh_action(sample.get('action', ''))}

请生成 {num_candidates} 个中文 NPC 台词候选。
要求：
1. 每个候选只写一句或两句，不要长篇独白。
2. 保持角色“谨慎、敏感、外冷内热、不喜欢空头承诺”的人格。
3. 必须符合当前情绪和行为意图。
4. 不要写成客服语气，不要过度煽情，不要突然原谅或突然极端。
5. 只输出编号列表，格式如下：
1. ...
2. ...
3. ...
""".strip()


def parse_candidates(text: str, fallback: str, max_candidates: int = 3) -> List[str]:
    lines = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        raw = re.sub(r"^[-*]\s*", "", raw)
        raw = re.sub(r"^\d+[\.、)）]\s*", "", raw)
        raw = raw.strip(' \"“”')
        if raw:
            lines.append(raw)
    # 如果模型输出一整段，兜底按中文句号粗切。
    if len(lines) <= 1 and len(text) > 20:
        parts = [p.strip(' \"“”') for p in re.split(r"(?<=[。！？!?])", text) if p.strip()]
        lines = parts or lines
    dedup = []
    for item in lines:
        if item and item not in dedup:
            dedup.append(item)
    if fallback and fallback not in dedup:
        dedup.insert(0, fallback)
    return dedup[:max_candidates]


def generate_candidates_for_sample(sample: Dict[str, Any], client: LLMClient, num_candidates: int = 3) -> Dict[str, Any]:
    prompt = build_teacher_prompt(sample, num_candidates=num_candidates)
    raw = client.generate(prompt, temperature=0.85)
    candidates = parse_candidates(raw, fallback=sample.get("dialogue", ""), max_candidates=num_candidates)
    while len(candidates) < num_candidates:
        candidates.append(sample.get("dialogue", "我需要一点时间想清楚。"))
    return {
        "event_type": sample.get("event_type", ""),
        "event_type_zh": sample.get("event_type_zh", ""),
        "emotion": sample.get("emotion", ""),
        "action": sample.get("action", ""),
        "player_text": sample.get("player_text", ""),
        "sample": sample,
        "teacher_raw": raw,
        "candidates": candidates,
    }


def batch_generate_teacher_candidates(
    samples: List[Dict[str, Any]],
    output_path: str = TEACHER_CANDIDATES_PATH,
    limit: int = 50,
    num_candidates: int = 3,
) -> str:
    client = LLMClient()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = samples[: max(0, int(limit))]
    with open(path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(selected):
            record = generate_candidates_for_sample(sample, client=client, num_candidates=num_candidates)
            record["sample_id"] = i
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)


def load_teacher_candidates(path: str = TEACHER_CANDIDATES_PATH) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
