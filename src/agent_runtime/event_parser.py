from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple

from src.character.event_model import EVENT_PRIMITIVES

# 轻量规则版事件解析器。
# 未来可由 Event Parser 小模型或 LoRA JSON 模型替换。
# 设计目标：让用户能直接输入自然语言事件，而不是手动选择固定事件标签。
KEYWORD_RULES: Dict[str, List[Tuple[str, float]]] = {
    "help": [("帮", 0.45), ("帮助", 0.75), ("处理", 0.45), ("解决", 0.55), ("找到了", 0.55), ("带来", 0.35)],
    "promise_kept": [("做到", 0.55), ("履行", 0.80), ("守约", 0.85), ("按时", 0.50), ("完成承诺", 0.85)],
    "promise_broken": [("没做到", 0.90), ("没有做到", 0.90), ("失约", 0.90), ("没去", 0.75), ("忘了", 0.55), ("没有回来", 0.80), ("迟到", 0.45), ("答应", 0.25)],
    "gift": [("礼物", 0.80), ("送", 0.55), ("给你", 0.35), ("带给", 0.45)],
    "lie": [("骗", 0.90), ("撒谎", 0.95), ("隐瞒", 0.70), ("没说实话", 0.85), ("编", 0.45)],
    "return_after_absence": [("回来", 0.60), ("很久", 0.45), ("离开", 0.55), ("消失", 0.70), ("好久", 0.55), ("昨天", 0.20), ("今天", 0.12)],
    "personal_question": [("过去", 0.55), ("为什么", 0.30), ("私人", 0.75), ("不愿回答", 0.65), ("追问", 0.70), ("秘密", 0.35)],
    "protect": [("保护", 0.90), ("挡", 0.55), ("救", 0.75), ("伤害你", 0.65), ("站到我身后", 0.70)],
    "ignore_request": [("没时间", 0.70), ("以后再说", 0.70), ("不管", 0.80), ("无视", 0.85), ("不重要", 0.45)],
    "apology": [("抱歉", 0.75), ("对不起", 0.85), ("道歉", 0.85), ("补救", 0.45), ("是我错", 0.90), ("解释", 0.35)],
    "quest_success": [("完成", 0.70), ("确认", 0.45), ("送到", 0.65), ("办好了", 0.75), ("成功", 0.75)],
    "quest_failed": [("失败", 0.80), ("没能", 0.65), ("被抢走", 0.80), ("变糟", 0.70), ("丢了", 0.55)],
    "threat": [("危险", 0.75), ("追杀", 0.90), ("袭击", 0.85), ("威胁", 0.85), ("受伤", 0.55), ("麻烦", 0.35)],
    "comfort": [("别怕", 0.65), ("安慰", 0.80), ("没事", 0.30), ("陪", 0.35), ("关心", 0.45)],
    "respect_boundary": [("不追问", 0.80), ("尊重", 0.70), ("以后再说", 0.35), ("不逼你", 0.85), ("等你愿意", 0.75)],
}

INTENSIFIERS = {
    "非常": 1.25,
    "特别": 1.20,
    "真的": 1.12,
    "一直": 1.15,
    "反复": 1.25,
    "又": 1.10,
    "只是": 0.85,
    "可能": 0.75,
    "好像": 0.70,
}

NEGATION_PATTERNS = ["没有骗", "没骗", "不是故意", "不是想骗"]


def _intensity_multiplier(text: str) -> float:
    mul = 1.0
    for word, factor in INTENSIFIERS.items():
        if word in text:
            mul *= factor
    return max(0.55, min(1.7, mul))


def parse_event_primitives(text: str, top_k: int = 6) -> Dict[str, float]:
    """把自然语言事件描述解析为事件原语权重。

    当前是可解释的规则版，优点是稳定、可展示、无需训练；后续可换为：
    1) API teacher parser；2) 本地 LoRA JSON parser；3) 小型分类/回归模型。
    """
    text = (text or "").strip()
    if not text:
        return {"return_after_absence": 0.2}

    scores: Dict[str, float] = {k: 0.0 for k in EVENT_PRIMITIVES}
    lower = text.lower()
    mul = _intensity_multiplier(text)

    for primitive, rules in KEYWORD_RULES.items():
        for kw, weight in rules:
            if kw in text or kw.lower() in lower:
                scores[primitive] += weight * mul

    # 简单否定修正：例如“我没有骗你”不应强烈触发 lie。
    if any(pat in text for pat in NEGATION_PATTERNS):
        scores["lie"] *= 0.35
        scores["apology"] += 0.10

    # “没做到/没有做到”是 promise_broken，不应同时强烈触发 promise_kept。
    if "没做到" in text or "没有做到" in text or "没能做到" in text:
        scores["promise_kept"] *= 0.10
        scores["promise_broken"] += 0.40 * mul

    # 组合修正：答应+没做到 是失约；回来+抱歉 是失约后的修复。
    if "答应" in text and ("没" in text or "没有" in text or "迟" in text):
        scores["promise_broken"] += 0.45 * mul
    if ("回来" in text or "回来了" in text) and ("抱歉" in text or "对不起" in text or "解释" in text):
        scores["return_after_absence"] += 0.35
        scores["apology"] += 0.35
    if ("危险" in text or "麻烦" in text) and ("告诉" in text or "解释" in text):
        scores["threat"] += 0.30
        # 告诉危险经历也属于轻微自我披露，但当前 primitive 中没有 self_disclosure，先用 comfort/return 轻度表达。
        scores["return_after_absence"] += 0.10

    # 截断和筛选。
    result = {k: round(max(0.0, min(1.5, v)), 3) for k, v in scores.items() if v > 0.08}
    if not result:
        result = {"return_after_absence": 0.25, "curiosity": 0.15} if "curiosity" in EVENT_PRIMITIVES else {"return_after_absence": 0.25}
    ranked = sorted(result.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return dict(ranked)


def explain_primitives(primitives: Dict[str, float]) -> str:
    lines = []
    for k, v in sorted(primitives.items(), key=lambda x: x[1], reverse=True):
        zh = EVENT_PRIMITIVES.get(k, k)
        lines.append(f"- {zh} / {k}: {v:.2f}")
    return "\n".join(lines) if lines else "未识别到明确事件原语。"
