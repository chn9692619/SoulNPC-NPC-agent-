# SoulNPC 架构说明

## 1. 情绪机制

早期版本使用 `mood` 离散标签。v1.0 改为连续情感向量：

- valence: 愉悦度
- arousal: 激活度
- dominance: 掌控感
- safety: 安全感
- stress: 压力
- curiosity: 好奇心

离散 mood 只作为 UI 摘要和训练标签派生，不再作为核心状态。

## 2. 关系机制

关系不是线性阶段，而是关系向量：

- trust
- intimacy
- dependence
- conflict
- boundary
- commitment

关系阶段是由向量派生出的可读摘要，不是核心模型假设。

## 3. 事件机制

复杂事件由小的抽象事件原语与权重组合而成，例如：

```json
{
  "promise_broken": 1.0,
  "apology": 0.2
}
```

后续可以训练事件解析器，把自然语言玩家行为映射到这些原语和权重。

## 4. 记忆机制

记忆带权重：

- importance
- emotional_salience
- relation_impact
- recency

低权重记忆可以随机遗忘，高权重或 pinned 记忆保留。后续可训练 Memory Scorer。

## 5. 行为意图基准

行为意图不由教师模型随意决定，而由以下因素共同决定：

- 当前情感状态
- 关系向量
- 玩家事件原语
- 角色目标
- 安全边界

教师模型主要负责生成候选台词，不应作为唯一行为标签来源。

## 6. 世界模型路线

长期版本可引入 World Model：

- 世界状态
- NPC 自身目标
- 其他 NPC 状态
- 时间推进 tick
- 无玩家介入时的自行动

MVP 阶段先不做完整世界模型，只保留架构接口。
