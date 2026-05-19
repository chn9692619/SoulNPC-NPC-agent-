from typing import Dict, Any


def choose_action(state_after: Dict[str, Any], primitives: Dict[str, float]) -> str:
    affect = state_after["affect"]
    rel = state_after["relationship"]
    # 行为意图不是教师模型决定的，而是由角色目标、情绪、关系、事件共同决定。
    if primitives.get("lie", 0) > 0.5:
        return "试探玩家"
    if primitives.get("promise_broken", 0) > 0.5:
        if rel["trust"] < 0.35:
            return "拉开距离"
        return "询问解释"
    if primitives.get("personal_question", 0) > 0.5 and affect["safety"] < 0.55:
        return "设立边界"
    if primitives.get("protect", 0) > 0.5 and rel["trust"] > 0.55:
        return "透露小秘密"
    if primitives.get("help", 0) > 0.5 or primitives.get("quest_success", 0) > 0.5:
        return "表达感谢"
    if primitives.get("apology", 0) > 0.6:
        return "部分接受道歉"
    if primitives.get("quest_failed", 0) > 0.5:
        return "评估损失"
    if rel["intimacy"] > 0.55 and affect["stress"] > 0.55:
        return "寻求确认"
    if affect["curiosity"] > 0.65:
        return "主动提问"
    return "平静回应"


DIALOGUE_BY_ACTION = {
    "询问解释": ["你昨天说会回来。我等了很久……发生什么了吗？", "我还没有生气。我只是想知道，你那句承诺到底算不算数。"],
    "拉开距离": ["我明白了。以后我会自己处理。", "如果我的请求对你来说不重要，那我也该停止期待。"],
    "试探玩家": ["你的话前后对不上。再说一次，这次说实话。", "我听够了漂亮的解释。我想听真正的原因。"],
    "设立边界": ["这不是我现在愿意回答的问题。", "有些门，只有在信任足够的时候才会打开。"],
    "透露小秘密": ["有件事我一直没告诉你。不是全部，但足够让你理解一点。", "你明明可以袖手旁观，却还是保护了我。所以，我也给你一个真相。"],
    "表达感谢": ["我没想到你真的会帮到这个程度。谢谢你，我会记住的。", "你做得比我请求的更多。即使我不太会表达，这件事对我很重要。"],
    "部分接受道歉": ["道歉不能让一切恢复原样。但至少，这是一个开始。", "我听见了。只是我还需要时间判断这意味着什么。"],
    "评估损失": ["失败不是最糟的。现在重要的是弄清楚损失，以及怎么补救。", "不要回避结果。告诉我，到底是哪一步出了问题。"],
    "寻求确认": ["如果你真的想留下，就不要只在没事的时候出现。", "我想相信你，但我需要看到你不是一时兴起。"],
    "主动提问": ["你为什么现在才告诉我这些？", "你说这句话的时候，像是在隐瞒另一件事。"],
    "平静回应": ["我需要一点时间想清楚你刚才做的事。", "我听见了。先让我整理一下。"],
}


def generate_rule_dialogue(action: str, state_after: Dict[str, Any]) -> str:
    items = DIALOGUE_BY_ACTION.get(action, DIALOGUE_BY_ACTION["平静回应"])
    idx = int((state_after["relationship"]["trust"] + state_after["affect"]["stress"]) * 10) % len(items)
    return items[idx]
