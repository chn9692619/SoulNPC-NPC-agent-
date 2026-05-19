STATE_LABELS = {
    "mood": "当前心情",
    "trust": "信任度",
    "affection": "亲近度",
    "stress": "压力值",
    "curiosity": "好奇心",
    "distance": "距离感",
    "current_goal": "当前目标",
    "relationship_stage": "关系阶段",
}

EMOTION_LABELS = {
    "calm": "平静",
    "neutral": "中性",
    "relieved": "松了一口气",
    "disappointed": "失望",
    "softened": "态度软化",
    "suspicious": "怀疑",
    "guarded": "保持戒备",
    "uneasy": "不安",
    "moved": "被触动",
    "hurt": "受伤",
    "cautiously_receptive": "谨慎地接受",
    "grateful": "感激",
    "concerned": "担忧",
}

ACTION_LABELS = {
    "offer_warm_thanks": "温和表达感谢",
    "ask_for_explanation": "询问解释",
    "respond_shyly": "略显害羞地回应",
    "test_player": "试探玩家",
    "ask_why_returned": "询问为何回来",
    "set_boundary": "设定边界",
    "reveal_small_secret": "透露小秘密",
    "withdraw": "退回距离",
    "accept_partially": "部分接受",
    "offer_new_lead": "提供新线索",
    "assess_damage": "评估损失",
}

GOAL_LABELS = {
    "observe_player": "观察玩家",
    "protect_self": "保护自己",
    "evaluate_player_reliability": "评估玩家是否可靠",
    "share_controlled_information": "有限分享情报",
    "learn_more_about_player": "进一步了解玩家",
    "maintain_safe_distance": "保持安全距离",
}

RELATIONSHIP_STAGE_LABELS = {
    "stranger": "陌生人",
    "cautious_ally": "谨慎盟友",
    "trusted_partner": "可信伙伴",
    "distrusted": "不信任对象",
}

IMPORTANCE_LABELS = {
    "low": "低",
    "medium": "中",
    "high": "高",
}


def zh_emotion(value: str) -> str:
    return EMOTION_LABELS.get(value, value)


def zh_action(value: str) -> str:
    return ACTION_LABELS.get(value, value)


def zh_goal(value: str) -> str:
    return GOAL_LABELS.get(value, value)


def zh_relationship(value: str) -> str:
    return RELATIONSHIP_STAGE_LABELS.get(value, value)


def zh_importance(value: str) -> str:
    return IMPORTANCE_LABELS.get(value, value)


def zh_state_value(key: str, value):
    if key == "mood":
        return zh_emotion(str(value))
    if key == "current_goal":
        return zh_goal(str(value))
    if key == "relationship_stage":
        return zh_relationship(str(value))
    if isinstance(value, float):
        return f"{value:.2f}"
    return value


def format_state_zh(state: dict) -> str:
    lines = []
    for key, value in state.items():
        label = STATE_LABELS.get(key, key)
        lines.append(f"{label}: {zh_state_value(key, value)}")
    return "\n".join(lines)
