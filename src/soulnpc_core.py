import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple

import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "generated_events.jsonl"
CANDIDATE_PATH = ROOT / "data" / "processed" / "teacher_candidates.jsonl"
OVERRIDE_PATH = ROOT / "data" / "processed" / "reviewed_overrides.jsonl"
PREFERENCE_PATH = ROOT / "data" / "preference" / "preference_pairs.jsonl"
FINAL_SFT_PATH = ROOT / "data" / "export" / "final_train_sft.jsonl"
FINAL_DPO_PATH = ROOT / "data" / "export" / "final_train_dpo.jsonl"
MODEL_CONFIG_PATH = ROOT / "config" / "model_config.json"
OUTPUT_DIR = ROOT / "outputs" / "final_baseline"

for p in [RAW_PATH.parent, CANDIDATE_PATH.parent, OVERRIDE_PATH.parent, PREFERENCE_PATH.parent, FINAL_SFT_PATH.parent, MODEL_CONFIG_PATH.parent, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

EVENTS = {
    "player_helped_npc": "玩家帮助了角色",
    "player_broke_promise": "玩家违背了承诺",
    "player_gave_gift": "玩家送给角色礼物",
    "player_lied": "玩家疑似撒谎",
    "player_returned_after_long_absence": "玩家长时间离开后回来",
    "player_asked_personal_question": "玩家追问私人问题",
    "player_protected_npc": "玩家保护了角色",
    "player_ignored_npc": "玩家无视了角色请求",
    "player_apologized": "玩家向角色道歉",
    "player_completed_quest": "玩家完成了角色委托",
    "player_failed_quest": "玩家未能完成角色委托",
}

EVENT_OPTIONS = [f"{zh}（{key}）" for key, zh in EVENTS.items()]
EVENT_OPTION_TO_KEY = {f"{zh}（{key}）": key for key, zh in EVENTS.items()}

EVENT_TEXTS = {
    "player_helped_npc": ["我找到了你需要的药。", "外面的麻烦我处理好了。", "你之前提到的人，我确认过了。"],
    "player_broke_promise": ["抱歉，我昨天说会回来，但没有做到。", "我知道我答应过你，但临时出了点事。", "我不是故意失约的。"],
    "player_gave_gift": ["我觉得你可能会喜欢这个。", "不是什么贵重东西，只是想送给你。", "路过集市时看到这个，突然想起了你。"],
    "player_lied": ["我没有对你隐瞒什么。", "我已经把知道的都告诉你了。", "你是不是想太多了？我没有骗你。"],
    "player_returned_after_long_absence": ["我知道我离开了很久。", "我回来，是因为我还想再见你一次。", "这段时间发生了很多事。"],
    "player_asked_personal_question": ["你为什么从来不提自己的过去？", "你来到这里之前，到底是什么人？", "你总是在回避这个问题，为什么？"],
    "player_protected_npc": ["站到我身后，我来处理。", "我不会让他们伤害你。", "别怕，这次换我保护你。"],
    "player_ignored_npc": ["我现在没时间管这个。", "以后再说吧。", "这不是我现在优先要处理的事。"],
    "player_apologized": ["对不起，我当时应该听你的。", "是我错了，我想补救。", "我知道道歉可能太晚，但我还是想说出来。"],
    "player_completed_quest": ["你交代的事我完成了。", "包裹已经送到了，没有人跟踪我。", "那条线索是真的，我确认过。"],
    "player_failed_quest": ["我失败了，包裹被抢走了。", "我没能及时完成任务。", "对不起，事情变糟了。"],
}

EMOTION_ZH = {
    "calm": "平静",
    "grateful": "感激",
    "disappointed": "失望",
    "touched": "被打动",
    "suspicious": "怀疑",
    "guarded": "戒备",
    "relieved": "安心",
    "hurt": "受伤",
    "curious": "好奇",
    "worried": "担忧",
}

ACTION_ZH = {
    "offer_warm_thanks": "真诚感谢",
    "ask_for_explanation": "询问解释",
    "respond_shyly": "含蓄回应",
    "test_player": "试探玩家",
    "ask_why_returned": "询问为何回来",
    "set_boundary": "设定边界",
    "reveal_small_secret": "透露小秘密",
    "withdraw": "后退回避",
    "accept_partially": "部分接受",
    "offer_new_lead": "提供新线索",
    "assess_damage": "评估损失",
}

RULES = {
    "player_helped_npc": ("grateful", "offer_warm_thanks", 0.12, 0.08, -0.05),
    "player_broke_promise": ("disappointed", "ask_for_explanation", -0.18, -0.05, 0.10),
    "player_gave_gift": ("touched", "respond_shyly", 0.05, 0.12, -0.03),
    "player_lied": ("suspicious", "test_player", -0.22, -0.08, 0.14),
    "player_returned_after_long_absence": ("guarded", "ask_why_returned", -0.08, -0.02, 0.08),
    "player_asked_personal_question": ("guarded", "set_boundary", -0.03, 0.00, 0.07),
    "player_protected_npc": ("relieved", "reveal_small_secret", 0.18, 0.10, -0.08),
    "player_ignored_npc": ("hurt", "withdraw", -0.15, -0.07, 0.09),
    "player_apologized": ("calm", "accept_partially", 0.08, 0.04, -0.04),
    "player_completed_quest": ("grateful", "offer_new_lead", 0.14, 0.07, -0.05),
    "player_failed_quest": ("worried", "assess_damage", -0.05, -0.02, 0.12),
}

DIALOGUE_TEMPLATES = {
    "offer_warm_thanks": ["我没想到你真的会帮到这个程度。谢谢你，我会记住的。", "这次我欠你一个人情。别笑，我不是随便说这种话的人。"],
    "ask_for_explanation": ["你昨天说会回来。我等了很久……发生什么了吗？", "如果你只是忘了，也请直接告诉我。比起借口，我更想听实话。"],
    "respond_shyly": ["你没必要特意带这个给我。不过……我会收好的。", "我不太习惯收礼物。但如果是你给的，我会认真保管。"],
    "test_player": ["你的话前后对不上。再说一次，这次说实话。", "我听够了漂亮的解释。我想听真正的原因。"],
    "ask_why_returned": ["你消失了这么久，现在又像什么都没发生一样回来。为什么？", "回来不是问题。问题是，你为什么现在才回来。"],
    "set_boundary": ["这不是我现在愿意回答的问题。", "有些门，只有在信任足够的时候才会打开。"],
    "reveal_small_secret": ["有件事我一直没告诉你。不是全部，但足够让你理解一点。", "你明明可以袖手旁观，却还是保护了我。所以，我也给你一个真相。"],
    "withdraw": ["我明白了。那我不会继续麻烦你。", "如果我的请求对你来说不重要，那我也该停止期待。"],
    "accept_partially": ["道歉不能让一切恢复原样。但至少，这是一个开始。", "我接受你的道歉，但信任不是一句话就能修好的。"],
    "offer_new_lead": ["你完成了你的部分。那我也会履行我的承诺。有条线索，你应该知道。", "既然你守约，我也不会继续隐瞒。这是新的线索。"],
    "assess_damage": ["失败不是最糟的。现在重要的是弄清楚损失，以及怎么补救。", "不要回避结果。告诉我，到底是哪一步出了问题。"],
}

def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))

def random_state(rng: random.Random) -> Dict[str, float]:
    return {
        "mood": "平静",
        "trust": round(rng.uniform(0.20, 0.80), 3),
        "affection": round(rng.uniform(0.10, 0.70), 3),
        "stress": round(rng.uniform(0.05, 0.60), 3),
        "curiosity": round(rng.uniform(0.10, 0.80), 3),
        "distance": round(rng.uniform(0.20, 0.85), 3),
    }

def apply_rule(event_type: str, state: Dict[str, Any]) -> Dict[str, Any]:
    emotion, action, trust_delta, affection_delta, stress_delta = RULES[event_type]
    after = dict(state)
    after["trust"] = round(clamp(after["trust"] + trust_delta), 3)
    after["affection"] = round(clamp(after["affection"] + affection_delta), 3)
    after["stress"] = round(clamp(after["stress"] + stress_delta), 3)
    after["mood"] = EMOTION_ZH[emotion]
    return {
        "emotion": emotion,
        "emotion_zh": EMOTION_ZH[emotion],
        "action": action,
        "action_zh": ACTION_ZH[action],
        "state_after": after,
        "deltas": {"trust": trust_delta, "affection": affection_delta, "stress": stress_delta},
    }

def build_sample(i: int, event_type: str, state: Dict[str, Any], player_text: str, rng: random.Random) -> Dict[str, Any]:
    rule = apply_rule(event_type, state)
    dialogue = rng.choice(DIALOGUE_TEMPLATES[rule["action"]])
    return {
        "sample_id": f"sample_{i:06d}",
        "character_name": "艾拉",
        "character_profile": "谨慎、敏感、外冷内热、观察力强，不喜欢空头承诺，但会记住真诚的帮助。",
        "scene": "边境星港附近的一间旧酒馆。艾拉是一名情报中介，长期隐藏自己的过去。",
        "event_type": event_type,
        "event_type_zh": EVENTS[event_type],
        "player_text": player_text,
        "state_before": state,
        "state_after": rule["state_after"],
        "emotion": rule["emotion"],
        "emotion_zh": rule["emotion_zh"],
        "action": rule["action"],
        "action_zh": rule["action_zh"],
        "deltas": rule["deltas"],
        "dialogue": dialogue,
    }

def generate_samples(n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    samples = []
    event_keys = list(EVENTS.keys())
    for i in range(n):
        event = rng.choice(event_keys)
        state = random_state(rng)
        player_text = rng.choice(EVENT_TEXTS[event])
        samples.append(build_sample(i, event, state, player_text, rng))
    return samples

def write_jsonl(records: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def append_jsonl(record: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def save_model_config(provider: str, api_key: str, base_url: str, model: str) -> str:
    payload = {"provider": provider, "api_key": api_key, "base_url": base_url, "model": model}
    MODEL_CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(MODEL_CONFIG_PATH)

def load_model_config() -> Dict[str, str]:
    if MODEL_CONFIG_PATH.exists():
        return json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    return {"provider": "mock", "api_key": "", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"}

def mock_candidates(sample: Dict[str, Any], k: int = 3) -> List[str]:
    base = sample["dialogue"]
    variants = [
        base,
        f"我听见了。只是这件事不会立刻从我心里消失。{sample['player_text'][:12]}……你最好认真一点。",
        f"我需要一点时间判断你这句话的分量。现在，我更想看你接下来怎么做。",
        f"别急着解释。你说过什么、做过什么，我都记得。",
    ]
    return variants[:k]

def call_teacher_model(sample: Dict[str, Any], k: int, config: Dict[str, str]) -> Tuple[List[str], str]:
    provider = config.get("provider", "mock")
    if provider == "mock" or not config.get("api_key"):
        return mock_candidates(sample, k), "Mock 本地候选"
    prompt = f"""
你是游戏角色台词生成器。请根据角色设定和状态，生成 {k} 个中文 NPC 回复候选。每个候选单独一行，不要编号。
角色：{sample['character_name']}
性格：{sample['character_profile']}
场景：{sample['scene']}
玩家事件：{sample['event_type_zh']}
玩家台词：{sample['player_text']}
当前情绪：{sample['emotion_zh']}
行为意图：{sample['action_zh']}
要求：自然、克制、符合记忆和情绪逻辑，不要客服腔。
""".strip()
    try:
        url = config.get("base_url", "").rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}
        body = {
            "model": config.get("model", "qwen-plus"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        resp = requests.post(url, headers=headers, json=body, timeout=45)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        lines = [x.strip(" -0123456789.、") for x in text.splitlines() if x.strip()]
        lines = [x for x in lines if len(x) >= 4]
        if len(lines) < k:
            lines += mock_candidates(sample, k - len(lines))
        return lines[:k], "真实教师模型候选"
    except Exception as exc:
        return mock_candidates(sample, k), f"教师模型失败，已使用 Mock 兜底：{exc}"

def generate_raw_dataset(n: int, seed: int) -> Dict[str, Any]:
    samples = generate_samples(n, seed)
    write_jsonl(samples, RAW_PATH)
    return {"count": len(samples), "path": str(RAW_PATH)}

def generate_teacher_candidates(limit: int, k: int, use_mock: bool = True, strict_real: bool = False) -> Dict[str, Any]:
    raw = read_jsonl(RAW_PATH)
    if not raw:
        return {"ok": False, "message": "缺少自动生成样本，请先生成基础训练样本。"}
    config = load_model_config()
    if use_mock:
        config["provider"] = "mock"
    else:
        provider = config.get("provider", "mock")
        if provider == "mock":
            return {"ok": False, "message": "你选择了真实 API 模式，但模型设置仍为 mock。请先到『模型设置』选择 qwen 或 deepseek 并保存。"}
        if not config.get("api_key"):
            return {"ok": False, "message": "你选择了真实 API 模式，但 API Key 为空。请先到『模型设置』填写并保存。"}
        if not config.get("base_url") or not config.get("model"):
            return {"ok": False, "message": "你选择了真实 API 模式，但 Base URL 或模型名称为空。请先到『模型设置』补全并保存。"}
    selected = raw[: min(limit, len(raw))]
    out = []
    logs = []
    for idx, sample in enumerate(selected, start=1):
        cands, msg = call_teacher_model(sample, k, config)
        if strict_real and ("失败" in msg or msg.startswith("Mock")):
            return {"ok": False, "message": f"真实教师模型调用失败，已停止。\n位置：{idx}/{len(selected)} {sample['sample_id']}\n原因：{msg}"}
        rec = {"sample_id": sample["sample_id"], "candidates": cands, "source": msg}
        out.append(rec)
        logs.append(f"{idx}/{len(selected)} {sample['sample_id']}：{msg}")
    write_jsonl(out, CANDIDATE_PATH)
    return {"ok": True, "count": len(out), "path": str(CANDIDATE_PATH), "logs": logs[-5:]}

def get_candidate_map() -> Dict[str, Dict[str, Any]]:
    return {r["sample_id"]: r for r in read_jsonl(CANDIDATE_PATH)}

def save_review(sample_id: str, chosen_text: str, rejected_text: str, final_text: str, note: str = "") -> Dict[str, Any]:
    rec = {
        "sample_id": sample_id,
        "final_text": final_text.strip() or chosen_text.strip(),
        "chosen": chosen_text.strip() or final_text.strip(),
        "rejected": rejected_text.strip(),
        "note": note,
    }
    append_jsonl(rec, OVERRIDE_PATH)
    if rec["chosen"] and rec["rejected"]:
        append_jsonl({"sample_id": sample_id, "prompt": sample_id, "chosen": rec["chosen"], "rejected": rec["rejected"], "note": note}, PREFERENCE_PATH)
    return rec

def latest_overrides() -> Dict[str, Dict[str, Any]]:
    overrides = {}
    for rec in read_jsonl(OVERRIDE_PATH):
        overrides[rec["sample_id"]] = rec
    return overrides

def make_rejected(sample: Dict[str, Any]) -> str:
    if sample["emotion"] in {"disappointed", "suspicious", "hurt", "guarded"}:
        return "没关系，我完全不在意。我们继续像以前一样吧。"
    if sample["emotion"] in {"grateful", "relieved", "touched"}:
        return "随便吧，这种事对我没有任何意义。"
    return "我没有任何感觉，也不需要回应。"

def build_prompt_text(sample: Dict[str, Any]) -> str:
    return (
        f"角色：{sample['character_name']}\n"
        f"性格：{sample['character_profile']}\n"
        f"场景：{sample['scene']}\n"
        f"玩家事件：{sample['event_type_zh']}\n"
        f"玩家台词：{sample['player_text']}\n"
        f"当前情绪：{sample['emotion_zh']}\n"
        f"行为意图：{sample['action_zh']}\n"
        "请生成一句符合角色人格、情绪和记忆逻辑的中文台词。"
    )

def export_final_training_data() -> Dict[str, Any]:
    raw = read_jsonl(RAW_PATH)
    if not raw:
        return {"ok": False, "message": "缺少自动生成样本，无法导出最终训练数据。"}
    overrides = latest_overrides()
    prefs = read_jsonl(PREFERENCE_PATH)
    sft = []
    dpo = []
    final_records = []
    for sample in raw:
        override = overrides.get(sample["sample_id"])
        final_dialogue = override["final_text"] if override else sample["dialogue"]
        final_sample = dict(sample)
        final_sample["final_dialogue"] = final_dialogue
        final_sample["used_human_override"] = bool(override)
        final_records.append(final_sample)
        sft.append({
            "messages": [
                {"role": "system", "content": "你是可信游戏角色智能体，需要根据角色状态生成自然、克制、符合情绪和记忆逻辑的中文台词。"},
                {"role": "user", "content": build_prompt_text(sample)},
                {"role": "assistant", "content": final_dialogue},
            ],
            "metadata": {"sample_id": sample["sample_id"], "event_type": sample["event_type"], "emotion": sample["emotion"], "action": sample["action"], "used_human_override": bool(override)},
        })
        dpo.append({
            "prompt": build_prompt_text(sample),
            "chosen": override["chosen"] if override else final_dialogue,
            "rejected": override["rejected"] if override and override.get("rejected") else make_rejected(sample),
            "metadata": {"sample_id": sample["sample_id"], "auto_weak_pair": not bool(override and override.get("rejected"))},
        })
    write_jsonl(sft, FINAL_SFT_PATH)
    write_jsonl(dpo, FINAL_DPO_PATH)
    return {
        "ok": True,
        "raw_count": len(raw),
        "override_count": len(overrides),
        "final_sft_count": len(sft),
        "final_dpo_count": len(dpo),
        "preference_count": len(prefs),
        "final_records": final_records,
        "sft_path": str(FINAL_SFT_PATH),
        "dpo_path": str(FINAL_DPO_PATH),
    }

def feature_text(sample: Dict[str, Any]) -> str:
    return " ".join([
        sample.get("event_type_zh", ""), sample.get("player_text", ""), sample.get("emotion_zh", ""), sample.get("action_zh", ""),
        json.dumps(sample.get("state_before", {}), ensure_ascii=False), sample.get("final_dialogue", sample.get("dialogue", "")),
    ])

def train_final_baseline() -> Dict[str, Any]:
    export = export_final_training_data()
    if not export.get("ok"):
        return export
    records = export["final_records"]
    if len(records) < 20:
        return {"ok": False, "message": "训练样本少于 20 条，请先生成更多自动样本。"}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X = [feature_text(r) for r in records]
    y_emo = [r["emotion"] for r in records]
    y_act = [r["action"] for r in records]
    summaries = []
    for name, y in [("emotion", y_emo), ("action", y_act)]:
        labels = sorted(set(y))
        stratify = y if min(y.count(c) for c in set(y)) >= 2 and len(set(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
        clf = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=6000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        report = classification_report(y_test, pred, labels=labels, zero_division=0)
        cm = confusion_matrix(y_test, pred, labels=labels).tolist()
        (OUTPUT_DIR / f"{name}_report.txt").write_text(report, encoding="utf-8")
        (OUTPUT_DIR / f"{name}_confusion_matrix.json").write_text(json.dumps({"labels": labels, "matrix": cm}, ensure_ascii=False, indent=2), encoding="utf-8")
        joblib.dump(clf, OUTPUT_DIR / f"{name}_classifier.joblib")
        summaries.append(f"{name} 分类报告：\n{report[:1200]}")
    return {
        "ok": True,
        "raw_count": export["raw_count"],
        "override_count": export["override_count"],
        "final_sft_count": export["final_sft_count"],
        "final_dpo_count": export["final_dpo_count"],
        "preference_count": export["preference_count"],
        "sft_path": export["sft_path"],
        "dpo_path": export["dpo_path"],
        "output_dir": str(OUTPUT_DIR),
        "summary": "\n\n".join(summaries),
    }

def count_jsonl(path: Path) -> int:
    return len(read_jsonl(path))

def project_status() -> Dict[str, int]:
    return {
        "raw": count_jsonl(RAW_PATH),
        "candidates": count_jsonl(CANDIDATE_PATH),
        "overrides": count_jsonl(OVERRIDE_PATH),
        "preferences": count_jsonl(PREFERENCE_PATH),
        "final_sft": count_jsonl(FINAL_SFT_PATH),
        "final_dpo": count_jsonl(FINAL_DPO_PATH),
        "reports": len(list(OUTPUT_DIR.glob("*.txt"))) if OUTPUT_DIR.exists() else 0,
    }

def clear_cache(level: str) -> Dict[str, Any]:
    targets = []
    if level == "仅清理教师候选与临时数据":
        targets = [CANDIDATE_PATH]
    elif level == "清理人工审阅与偏好数据":
        targets = [OVERRIDE_PATH, PREFERENCE_PATH]
    elif level == "清理自动生成数据":
        targets = [RAW_PATH]
    elif level == "清理训练导出与输出":
        targets = [FINAL_SFT_PATH, FINAL_DPO_PATH, OUTPUT_DIR]
    elif level == "全量重置数据与输出":
        targets = [RAW_PATH, CANDIDATE_PATH, OVERRIDE_PATH, PREFERENCE_PATH, FINAL_SFT_PATH, FINAL_DPO_PATH, OUTPUT_DIR]
    removed = []
    for t in targets:
        if t.is_dir() and t.exists():
            shutil.rmtree(t)
            removed.append(str(t))
        elif t.exists():
            t.unlink()
            removed.append(str(t))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return {"removed": removed, "count": len(removed)}

def run_inference(event_option: str, player_text: str) -> Dict[str, Any]:
    event = EVENT_OPTION_TO_KEY.get(event_option, "player_broke_promise")
    rng = random.Random()
    state = random_state(rng)
    sample = build_sample(0, event, state, player_text or rng.choice(EVENT_TEXTS[event]), rng)
    return sample
