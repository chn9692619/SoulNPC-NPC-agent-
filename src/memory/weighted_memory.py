from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datetime import datetime
import random
import math


@dataclass
class MemoryItem:
    text: str
    event: str
    importance: float
    emotional_salience: float
    relation_impact: float
    pinned: bool = False
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")

    def score(self, recency_rank: int = 0) -> float:
        # 简化版权重：未来可以用训练出来的 scorer 替代。
        recency = math.exp(-0.15 * recency_rank)
        return 0.45*self.importance + 0.25*self.emotional_salience + 0.20*self.relation_impact + 0.10*recency

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WeightedMemoryStore:
    def __init__(self, max_items: int = 64, rng_seed: int = 42):
        self.items: List[MemoryItem] = []
        self.max_items = max_items
        self.rng = random.Random(rng_seed)

    def add(self, item: MemoryItem):
        self.items.append(item)
        self.apply_stochastic_forgetting()

    def retrieve(self, top_k: int = 5) -> List[MemoryItem]:
        ranked = []
        for i, item in enumerate(reversed(self.items)):
            ranked.append((item.score(recency_rank=i), item))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in ranked[:top_k]]

    def apply_stochastic_forgetting(self, threshold: float = 0.34):
        if len(self.items) <= self.max_items:
            return
        keep = []
        for idx, item in enumerate(self.items):
            if item.pinned:
                keep.append(item)
                continue
            s = item.score(recency_rank=len(self.items)-idx)
            if s >= threshold:
                keep.append(item)
            else:
                # 低权重记忆以概率遗忘，不是硬删除。
                keep_prob = max(0.05, s / threshold)
                if self.rng.random() < keep_prob:
                    keep.append(item)
        # 如果还太多，保留分数最高的。
        if len(keep) > self.max_items:
            scored = [(m.score(i), m) for i, m in enumerate(reversed(keep))]
            scored.sort(key=lambda x: x[0], reverse=True)
            keep = [m for _, m in scored[:self.max_items]]
        self.items = keep

    def to_context(self, top_k: int = 5) -> str:
        top = self.retrieve(top_k=top_k)
        if not top:
            return "暂无关键记忆。"
        lines = []
        for m in top:
            lines.append(f"- {m.text}（重要度 {m.importance:.2f}，情绪显著性 {m.emotional_salience:.2f}）")
        return "\n".join(lines)
