import json
import re
from typing import List, Dict, Any
import requests

from src.modeling.config import load_model_config
from src.character.core import choose_rule_dialogue


def _extract_candidates(text: str, k: int) -> List[str]:
    text = text.strip()
    # Try JSON list first
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()][:k]
        if isinstance(obj, dict):
            arr = obj.get('candidates') or obj.get('候选') or obj.get('responses')
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()][:k]
    except Exception:
        pass
    # split numbered lines
    lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r'^[ABCabc123]\s*[\.、:：）\)]\s*', '', line)
        line = line.strip('"“” ')
        if line and len(line) > 2:
            lines.append(line)
    if lines:
        return lines[:k]
    return [text[:300]] if text else []


def mock_candidates(sample: Dict[str, Any], k: int = 3) -> List[str]:
    base = sample.get('dialogue') or choose_rule_dialogue(sample.get('action', 'stay_observant'), sample.get('state_after', {}), sample.get('sample_id',''))
    event = sample.get('event_type_zh', '这件事')
    variants = [
        base,
        f"关于{event}，我不会立刻下结论。但我会记住你这次的选择。",
        f"我听见了。只是这件事对我来说不算小，我需要一点时间判断你真正的意思。",
        f"如果你希望我相信你，就不要只给我一句解释。让我看看你接下来会怎么做。",
    ]
    return variants[:k]


def build_teacher_prompt(sample: Dict[str, Any], k: int) -> str:
    payload = {
        '角色': '艾拉',
        '人格': '谨慎、敏感、外冷内热、观察力强，不喜欢空头承诺，但会记住真诚的帮助。',
        '场景': '边境星港附近的一间旧酒馆。艾拉是一名情报中介，长期隐藏自己的过去。',
        '玩家事件': sample.get('event_type_zh'),
        '玩家台词': sample.get('player_text'),
        '事件前状态': sample.get('state_before'),
        '事件后状态': sample.get('state_after'),
        '情绪': sample.get('emotion'),
        '行为意图': sample.get('action'),
    }
    return f"""
请为游戏角色“艾拉”生成 {k} 个候选中文 NPC 台词。
要求：
- 台词自然、克制、符合“谨慎、外冷内热、记忆力强”的人格。
- 必须符合当前情绪和行为意图。
- 不要写成客服语气，不要过度煽情。
- 只输出 JSON 数组，例如 ["候选1", "候选2", "候选3"]。

上下文：
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def call_teacher_model(sample: Dict[str, Any], k: int = 3) -> List[str]:
    cfg = load_model_config()
    if cfg.get('provider') == 'mock':
        return mock_candidates(sample, k)
    api_key = cfg.get('api_key')
    base_url = cfg.get('base_url')
    model = cfg.get('model')
    if not api_key or not base_url or not model:
        raise RuntimeError('真实 API 模式需要填写 API Key、Base URL 和模型名称。')
    url = base_url.rstrip('/') + '/chat/completions'
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    prompt = build_teacher_prompt(sample, k)
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': '你是游戏角色对话数据生成器，只输出符合要求的中文候选台词。'},
            {'role': 'user', 'content': prompt},
        ],
        'temperature': 0.8,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    content = data['choices'][0]['message']['content']
    cands = _extract_candidates(content, k)
    if len(cands) < k:
        cands += mock_candidates(sample, k - len(cands))
    return cands[:k]
