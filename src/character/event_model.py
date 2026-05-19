from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import random


# 小的抽象事件原语。复杂事件由多个 primitive + weight 组合而成。
EVENT_PRIMITIVES = {
    "help": "玩家提供帮助",
    "promise_kept": "玩家履行承诺",
    "promise_broken": "玩家违背承诺",
    "gift": "玩家赠送礼物",
    "lie": "玩家疑似撒谎",
    "return_after_absence": "玩家长时间离开后回来",
    "personal_question": "玩家追问私人问题",
    "protect": "玩家保护角色",
    "ignore_request": "玩家无视请求",
    "apology": "玩家表达歉意",
    "quest_success": "玩家完成委托",
    "quest_failed": "玩家未能完成委托",
    "threat": "玩家带来威胁或风险",
    "comfort": "玩家安慰角色",
    "respect_boundary": "玩家尊重边界",
}

# 每个原语对情感、关系的基础影响。未来可由模型学习替代。
PRIMITIVE_EFFECTS = {
    "help": {"affect": {"valence": 0.20, "safety": 0.08, "stress": -0.06}, "rel": {"trust": 0.12, "intimacy": 0.04}},
    "promise_kept": {"affect": {"valence": 0.18, "safety": 0.08}, "rel": {"trust": 0.16, "commitment": 0.08, "conflict": -0.05}},
    "promise_broken": {"affect": {"valence": -0.25, "safety": -0.10, "stress": 0.14, "arousal": 0.10}, "rel": {"trust": -0.22, "conflict": 0.13, "boundary": 0.08}},
    "gift": {"affect": {"valence": 0.12, "curiosity": 0.05}, "rel": {"intimacy": 0.08, "trust": 0.04}},
    "lie": {"affect": {"valence": -0.18, "safety": -0.12, "stress": 0.12}, "rel": {"trust": -0.20, "conflict": 0.16, "boundary": 0.10}},
    "return_after_absence": {"affect": {"curiosity": 0.15, "arousal": 0.08, "valence": -0.04}, "rel": {"boundary": 0.06}},
    "personal_question": {"affect": {"safety": -0.08, "stress": 0.08}, "rel": {"boundary": 0.10}},
    "protect": {"affect": {"valence": 0.22, "safety": 0.18, "stress": -0.10}, "rel": {"trust": 0.18, "intimacy": 0.10, "commitment": 0.08}},
    "ignore_request": {"affect": {"valence": -0.16, "stress": 0.08}, "rel": {"trust": -0.12, "conflict": 0.07, "boundary": 0.05}},
    "apology": {"affect": {"valence": 0.06, "stress": -0.04}, "rel": {"trust": 0.05, "conflict": -0.05}},
    "quest_success": {"affect": {"valence": 0.18, "safety": 0.06}, "rel": {"trust": 0.12, "commitment": 0.06}},
    "quest_failed": {"affect": {"valence": -0.12, "stress": 0.10}, "rel": {"trust": -0.06, "conflict": 0.04}},
    "threat": {"affect": {"valence": -0.20, "safety": -0.20, "stress": 0.20, "arousal": 0.12}, "rel": {"conflict": 0.14, "boundary": 0.10}},
    "comfort": {"affect": {"valence": 0.12, "stress": -0.08, "safety": 0.06}, "rel": {"intimacy": 0.06, "trust": 0.04}},
    "respect_boundary": {"affect": {"safety": 0.10, "stress": -0.04}, "rel": {"trust": 0.10, "boundary": -0.04}},
}

PRESET_COMPLEX_EVENTS = {
    "玩家帮助了角色": {"help": 0.9, "respect_boundary": 0.2},
    "玩家违背了承诺": {"promise_broken": 1.0, "apology": 0.1},
    "玩家送给角色礼物": {"gift": 0.9, "comfort": 0.2},
    "玩家疑似撒谎": {"lie": 1.0},
    "玩家长时间离开后回来": {"return_after_absence": 1.0, "promise_broken": 0.3},
    "玩家追问角色的私人问题": {"personal_question": 1.0},
    "玩家保护了角色": {"protect": 1.0, "help": 0.5},
    "玩家无视角色请求": {"ignore_request": 1.0},
    "玩家向角色道歉": {"apology": 1.0},
    "玩家完成了角色委托": {"quest_success": 1.0, "promise_kept": 0.4},
    "玩家未能完成角色委托": {"quest_failed": 1.0},
}

PLAYER_TEXT_BY_EVENT = {
    "玩家帮助了角色": ["我找到了你需要的药。", "外面的麻烦我已经处理好了，你暂时安全了。"],
    "玩家违背了承诺": ["抱歉，我昨天说会回来，但我没有做到。", "我不是故意失约的，只是事情比我想的复杂。"],
    "玩家送给角色礼物": ["我觉得你可能会喜欢这个。", "路过集市时看到这个，突然想起了你。"],
    "玩家疑似撒谎": ["我没有对你隐瞒什么。", "我已经把知道的都告诉你了。"],
    "玩家长时间离开后回来": ["我知道我离开了很久。", "我回来，是因为我还想再见你一次。"],
    "玩家追问角色的私人问题": ["你为什么从来不提自己的过去？", "你来到这里之前，到底是什么人？"],
    "玩家保护了角色": ["站到我身后，我来处理。", "我不会让他们伤害你。"],
    "玩家无视角色请求": ["我现在没时间管这个。", "以后再说吧。"],
    "玩家向角色道歉": ["对不起，我当时应该听你的。", "是我错了，我想补救。"],
    "玩家完成了角色委托": ["你交代的事我完成了。", "那条线索是真的，我已经确认过。"],
    "玩家未能完成角色委托": ["我失败了，包裹被抢走了。", "对不起，事情变糟了。"],
}


def compose_effect(primitives: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    affect: Dict[str, float] = {}
    rel: Dict[str, float] = {}
    for p, weight in primitives.items():
        eff = PRIMITIVE_EFFECTS.get(p, {})
        for k, v in eff.get("affect", {}).items():
            affect[k] = affect.get(k, 0.0) + v * weight
        for k, v in eff.get("rel", {}).items():
            rel[k] = rel.get(k, 0.0) + v * weight
    return {"affect": affect, "relationship": rel}


def random_complex_event(rng: random.Random) -> tuple[str, Dict[str, float], str]:
    event_name = rng.choice(list(PRESET_COMPLEX_EVENTS.keys()))
    primitives = dict(PRESET_COMPLEX_EVENTS[event_name])
    # 加入轻微噪声，模拟复杂事件的不同强度。
    for k in primitives:
        primitives[k] = max(0.05, min(1.5, primitives[k] * rng.uniform(0.75, 1.25)))
    player_text = rng.choice(PLAYER_TEXT_BY_EVENT[event_name])
    return event_name, primitives, player_text
