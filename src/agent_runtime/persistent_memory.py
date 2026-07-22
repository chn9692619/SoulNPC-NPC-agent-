from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _tokenize(text: str) -> set[str]:
    text = text or ""
    # 对中文不做重分词，使用字符 bigram + 关键词混合，避免引入额外依赖。
    chars = [c for c in text if not c.isspace()]
    grams = set(chars)
    grams.update("".join(chars[i:i+2]) for i in range(max(0, len(chars)-1)))
    return grams


@dataclass
class RuntimeMemory:
    memory_id: str
    content: str
    event_primitives: Dict[str, float]
    importance: float = 0.5
    emotional_salience: float = 0.5
    relation_impact: float = 0.5
    pinned: bool = False
    created_at: str = ""
    last_accessed_at: str = ""
    access_count: int = 0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = _now()
        if not self.last_accessed_at:
            self.last_accessed_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PersistentMemoryRAG:
    """轻量 RAG 记忆库。

    这不是普通知识库，而是角色的内生记忆：
    - 语义相关性决定当前是否被想起；
    - importance/emotional_salience/relation_impact 决定长期保留价值；
    - pinned 记忆永不随机遗忘；
    - 低权重记忆可 dropout，模拟人类记忆弱化。
    """

    def __init__(self, path: str | Path, max_items: int = 200, seed: int = 42):
        self.path = Path(path)
        self.max_items = max_items
        self.rng = random.Random(seed)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: List[RuntimeMemory] = []
        self.load()

    def load(self):
        self.items = []
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append(RuntimeMemory(**obj))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for item in self.items:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    def reset(self):
        self.items = []
        if self.path.exists():
            self.path.unlink()

    def add(self, content: str, event_primitives: Dict[str, float], importance: float, emotional_salience: float, relation_impact: float, pinned: bool = False) -> RuntimeMemory:
        mem = RuntimeMemory(
            memory_id=f"mem_{len(self.items)+1:06d}",
            content=content,
            event_primitives=event_primitives,
            importance=float(max(0, min(1, importance))),
            emotional_salience=float(max(0, min(1, emotional_salience))),
            relation_impact=float(max(0, min(1, relation_impact))),
            pinned=bool(pinned),
        )
        self.items.append(mem)
        self.apply_dropout()
        self.save()
        return mem

    def _score(self, item: RuntimeMemory, query: str, primitives: Dict[str, float], recency_rank: int) -> float:
        q_tokens = _tokenize(query)
        m_tokens = _tokenize(item.content)
        semantic = len(q_tokens & m_tokens) / max(1, len(q_tokens | m_tokens))
        event_overlap = 0.0
        for k, v in primitives.items():
            event_overlap += min(float(v), float(item.event_primitives.get(k, 0.0)))
        event_overlap = min(1.0, event_overlap)
        recency = math.exp(-0.18 * recency_rank)
        pinned_bonus = 0.25 if item.pinned else 0.0
        return (
            0.30 * semantic
            + 0.20 * event_overlap
            + 0.18 * item.importance
            + 0.14 * item.emotional_salience
            + 0.10 * item.relation_impact
            + 0.08 * recency
            + pinned_bonus
        )

    def retrieve(self, query: str, primitives: Dict[str, float], top_k: int = 5) -> List[Tuple[float, RuntimeMemory]]:
        scored: List[Tuple[float, RuntimeMemory]] = []
        for rank, item in enumerate(reversed(self.items)):
            score = self._score(item, query, primitives, rank)
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = scored[:top_k]
        now = _now()
        for _, item in out:
            item.access_count += 1
            item.last_accessed_at = now
        self.save()
        return out

    def apply_dropout(self, threshold: float = 0.30):
        if len(self.items) <= self.max_items:
            return
        kept: List[RuntimeMemory] = []
        for item in self.items:
            if item.pinned:
                kept.append(item)
                continue
            base = 0.45 * item.importance + 0.30 * item.emotional_salience + 0.25 * item.relation_impact
            if base >= threshold:
                kept.append(item)
            else:
                if self.rng.random() < max(0.05, base / max(threshold, 1e-6)):
                    kept.append(item)
        if len(kept) > self.max_items:
            kept.sort(key=lambda m: (m.pinned, m.importance + m.emotional_salience + m.relation_impact, m.created_at), reverse=True)
            kept = kept[:self.max_items]
        self.items = kept

    def context_text(self, retrieved: List[Tuple[float, RuntimeMemory]]) -> str:
        if not retrieved:
            return "暂无被检索到的关键记忆。"
        lines = []
        for score, item in retrieved:
            pin = "，永久" if item.pinned else ""
            lines.append(
                f"- [{item.memory_id}] {item.content}（检索分 {score:.2f}，重要度 {item.importance:.2f}，情绪显著性 {item.emotional_salience:.2f}，关系影响 {item.relation_impact:.2f}{pin}）"
            )
        return "\n".join(lines)
