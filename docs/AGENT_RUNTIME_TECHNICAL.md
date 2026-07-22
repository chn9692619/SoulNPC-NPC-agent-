# SoulNPC v2.0 Agent Runtime Technical Notes

## 1. Why v2.0 exists

The earlier SoulNPC workbench can generate data, export SFT/DPO datasets, and launch LoRA SFT training on AutoDL. However, a training pipeline alone does not show the final agent behavior clearly. Version 2.0 adds an Agent Runtime page that demonstrates the runtime chain:

```text
Natural language player event
-> Event Parser
-> Weighted Memory RAG
-> Cognitive-affective state update
-> Action Policy
-> Dialogue Prompt
-> Rule response / Cloud LoRA response
-> Memory write-back
```

## 2. Model decomposition

SoulNPC is intentionally not designed as one end-to-end black-box model. It is a layered agent:

- Event Parser: maps natural language events to abstract event primitives and weights.
- Memory RAG: retrieves weighted character memories based on semantic and event relevance.
- State Engine: updates continuous affect vectors and relationship vectors.
- Action Policy: predicts the character's behavioral intention.
- Dialogue Model: a Qwen-style Transformer decoder fine-tuned with LoRA to generate NPC dialogue.
- Critic / DPO stage: future module for preferring more consistent character outputs.

This decomposition improves interpretability, controllability, and data efficiency.

## 3. Current Event Parser

The current Event Parser is a rule-based parser. It recognizes patterns such as promise breaking, apology, threat, gift, protection, lying, and personal questions. It outputs a dictionary:

```json
{
  "promise_broken": 1.0,
  "apology": 0.5,
  "threat": 0.4
}
```

This is a bootstrap implementation. Later it can be replaced by:

1. API teacher parser;
2. a small supervised classifier/regressor;
3. a LoRA-tuned JSON policy model.

## 4. Memory RAG

SoulNPC uses a weighted memory store. Each memory has:

- content;
- event primitives;
- importance;
- emotional salience;
- relationship impact;
- pinned flag;
- access count and timestamps.

Retrieval score currently combines:

```text
semantic overlap
+ event primitive overlap
+ importance
+ emotional salience
+ relationship impact
+ recency
+ pinned bonus
```

Low-weight memories can be dropped when the store is too large, while pinned or high-weight memories stay. This is a lightweight approximation of long-term character memory.

## 5. Continuous emotion and relationship vectors

SoulNPC does not treat `mood` as the only state. The underlying affect state is continuous:

- valence;
- arousal;
- dominance;
- safety;
- stress;
- curiosity.

The relationship state is also multi-dimensional:

- trust;
- intimacy;
- dependence;
- conflict;
- boundary;
- commitment.

Discrete mood and relationship stage are only UI summaries.

## 6. LoRA SFT role

LoRA SFT currently trains the Dialogue Model. It does not train the whole agent. The base model is a Transformer decoder model such as Qwen2.5-1.5B-Instruct. LoRA freezes most base model weights and trains a small low-rank adapter.

Training objective:

```text
input: character profile + event primitives + memory + affect/relationship state + action intention
output: character-consistent NPC dialogue
```

## 7. Why this is different from prompt-only NPCs

Prompt-only NPCs rely on a large model to improvise everything. SoulNPC makes character state explicit:

- events are parsed;
- memories are retrieved;
- emotions and relationships are updated;
- actions are selected;
- the dialogue model only verbalizes the selected character reaction.

This is designed for believable game characters with long-term consistency.
