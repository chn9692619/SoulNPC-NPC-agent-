from typing import Dict, Any


DIALOGUE_TEMPLATES = {
    "ask_for_explanation": [
        "你昨天说会回来。我等了很久……发生什么了吗？",
        "我还没有生气。我只是想知道，你那句承诺到底算不算数。",
        "如果你只是忘了，也请直接告诉我。比起借口，我更想听实话。",
    ],
    "offer_warm_thanks": [
        "我没想到你真的会帮到这个程度。谢谢你，我会记住的。",
        "你做得比我请求的更多。即使我不太会表达，这件事对我很重要。",
        "这次我欠你一个人情。别笑，我不是随便说这种话的人。",
    ],
    "respond_shyly": [
        "你没必要特意带这个给我。不过……我会收好的。",
        "用礼物让我放松警惕，这招有点狡猾。可惜，好像确实有用。",
        "我不太习惯收礼物。但如果是你给的，我会认真保管。",
    ],
    "test_player": [
        "你的话前后对不上。再说一次，这次说实话。",
        "我听够了漂亮的解释。我想听真正的原因。",
        "你以为我没有发现吗？我只是想看看你会不会主动承认。",
    ],
    "ask_why_returned": [
        "你消失了这么久，现在又像什么都没发生一样回来。为什么？",
        "我以为你已经忘了这里。看来并不是。",
        "回来不是问题。问题是，你为什么现在才回来。",
    ],
    "set_boundary": [
        "这不是我现在愿意回答的问题。",
        "有些门，只有在信任足够的时候才会打开。",
        "别再追问了。至少现在不要。",
    ],
    "reveal_small_secret": [
        "有件事我一直没告诉你。不是全部，但足够让你理解一点。",
        "你明明可以袖手旁观，却还是保护了我。所以，我也给你一个真相。",
        "我不习惯把秘密交给别人。但这次，我愿意告诉你一小部分。",
    ],
    "withdraw": [
        "我明白了。那我不会继续麻烦你。",
        "如果我的请求对你来说不重要，那我也该停止期待。",
        "没关系。以后我会自己处理。",
    ],
    "accept_partially": [
        "道歉不能让一切恢复原样。但至少，这是一个开始。",
        "我听见了。只是我还需要时间判断这意味着什么。",
        "我接受你的道歉，但信任不是一句话就能修好的。",
    ],
    "offer_new_lead": [
        "你完成了你的部分。那我也会履行我的承诺。有条线索，你应该知道。",
        "你证明了自己。下一步，我可以把更重要的情报告诉你。",
        "既然你守约，我也不会继续隐瞒。这是新的线索。",
    ],
    "assess_damage": [
        "失败不是最糟的。现在重要的是弄清楚损失，以及怎么补救。",
        "不要回避结果。告诉我，到底是哪一步出了问题。",
        "东西丢了，时间也浪费了。但如果你还想补救，就先把情况说清楚。",
    ],
}


def generate_rule_dialogue(action: str, state_after: Dict[str, Any], player_text: str = "") -> str:
    templates = DIALOGUE_TEMPLATES.get(action, ["我需要一点时间想清楚你刚才做的事。"])
    index = int((state_after.get("trust", 0.5) + state_after.get("stress", 0.2)) * 10) % len(templates)
    return templates[index]


def build_dialogue_prompt(character_profile: Dict[str, Any], sample: Dict[str, Any]) -> str:
    return f"""
角色设定：
{character_profile}

事件前状态：
{sample.get('state_before')}

事件后状态：
{sample.get('state_after')}

玩家事件：
{sample.get('event_type')}

玩家台词：
{sample.get('player_text', '')}

当前情绪：
{sample.get('emotion')}

行为意图：
{sample.get('action')}

请为这个角色写一句自然、符合人格和情绪状态的中文台词。
""".strip()
