from typing import List, Dict, Any

from .schemas import MemoryItem


EVENT_ZH = {
    "player_helped_npc": "玩家帮助了角色",
    "player_broke_promise": "玩家违背了承诺",
    "player_gave_gift": "玩家送给角色礼物",
    "player_lied": "玩家疑似撒谎",
    "player_returned_after_long_absence": "玩家长时间离开后回来",
    "player_asked_personal_question": "玩家追问角色的私人问题",
    "player_protected_npc": "玩家保护了角色",
    "player_ignored_npc": "玩家无视了角色请求",
    "player_apologized": "玩家向角色道歉",
    "player_completed_quest": "玩家完成了角色委托",
    "player_failed_quest": "玩家未能完成角色委托",
}

EMOTION_ZH = {
    "relieved": "安心",
    "disappointed": "失望",
    "softened": "态度软化",
    "suspicious": "怀疑",
    "guarded": "戒备",
    "uneasy": "不安",
    "moved": "被触动",
    "hurt": "受伤",
    "cautiously_receptive": "谨慎接受",
    "grateful": "感激",
    "concerned": "担忧",
}


class MemoryBank:
    def __init__(self, max_items: int = 20):
        self.max_items = max_items
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]

    def recent_text(self, n: int = 5) -> str:
        if not self.items:
            return "暂无重要记忆。"
        selected = self.items[-n:]
        return "\n".join([f"- {m.summary}" for m in selected])

    def to_list(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.items]


def summarize_event(event_type: str, player_text: str, emotion: str) -> str:
    event = EVENT_ZH.get(event_type, event_type)
    emotion_text = EMOTION_ZH.get(emotion, emotion)
    if player_text:
        return f"{event}。玩家表达：『{player_text}』。角色产生了{emotion_text}的情绪。"
    return f"{event}。角色产生了{emotion_text}的情绪。"
