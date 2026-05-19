from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, Any
import math


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class AffectVector:
    """连续情感向量。替代单一离散 mood。

    valence: 愉悦-不愉悦
    arousal: 激活度/情绪强度
    dominance: 掌控感
    safety: 安全感
    stress: 压力水平
    curiosity: 好奇/探索倾向
    """
    valence: float = 0.50
    arousal: float = 0.35
    dominance: float = 0.45
    safety: float = 0.55
    stress: float = 0.25
    curiosity: float = 0.45

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def update(self, deltas: Dict[str, float], momentum: float = 0.85) -> None:
        for key, delta in deltas.items():
            if hasattr(self, key):
                old = getattr(self, key)
                # mood 是慢变量，因此 delta 不全量写入，而是带 momentum 平滑。
                new = clip01(old * momentum + (old + delta) * (1 - momentum))
                setattr(self, key, new)

    def discrete_label(self) -> str:
        """给 UI 和训练标签用的粗粒度情绪名。连续向量仍然是核心状态。"""
        if self.stress > 0.68 and self.valence < 0.42:
            return "紧张"
        if self.valence < 0.34 and self.arousal > 0.55:
            return "受伤"
        if self.valence < 0.42 and self.safety < 0.45:
            return "警惕"
        if self.valence < 0.45:
            return "失望"
        if self.valence > 0.68 and self.safety > 0.55:
            return "安心"
        if self.curiosity > 0.66:
            return "好奇"
        return "平静"


@dataclass
class RelationshipVector:
    """非线性关系向量。关系阶段只是派生摘要，不是线性主轴。"""
    trust: float = 0.50
    intimacy: float = 0.25
    dependence: float = 0.20
    conflict: float = 0.10
    boundary: float = 0.65
    commitment: float = 0.20

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def update(self, deltas: Dict[str, float]) -> None:
        for key, delta in deltas.items():
            if hasattr(self, key):
                setattr(self, key, clip01(getattr(self, key) + delta))

    def stage(self) -> str:
        # 关系不是线性阶段，这里仅用于 UI 摘要。
        score = 0.35*self.trust + 0.25*self.intimacy + 0.20*self.commitment - 0.20*self.conflict
        if self.conflict > 0.65 and self.trust < 0.35:
            return "紧张对立"
        if score < 0.25:
            return "疏离观察"
        if score < 0.45:
            return "谨慎接触"
        if score < 0.65:
            return "逐步信任"
        if score < 0.80:
            return "亲密同盟"
        return "深度羁绊"


@dataclass
class CharacterState:
    name: str = "艾拉"
    personality: str = "谨慎、敏感、外冷内热、观察力强，不喜欢空头承诺，但会记住真诚的帮助"
    scene: str = "边境星港附近的一间旧酒馆。艾拉是一名情报中介，长期隐藏自己的过去。"
    affect: AffectVector = field(default_factory=AffectVector)
    relationship: RelationshipVector = field(default_factory=RelationshipVector)
    current_goal: str = "观察玩家是否可靠"
    long_term_goal: str = "保护自己掌握的秘密，并判断玩家是否值得信任"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "personality": self.personality,
            "scene": self.scene,
            "affect": self.affect.to_dict(),
            "relationship": self.relationship.to_dict(),
            "derived_mood": self.affect.discrete_label(),
            "relationship_stage": self.relationship.stage(),
            "current_goal": self.current_goal,
            "long_term_goal": self.long_term_goal,
        }
