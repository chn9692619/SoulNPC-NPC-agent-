from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List
import random

from src.utils.labels import EVENT_ZH, MOOD_ZH, ACTION_ZH, RELATIONSHIP_ZH


@dataclass
class CharacterState:
    mood: str = 'calm'
    trust: float = 0.50
    affection: float = 0.30
    stress: float = 0.20
    curiosity: float = 0.40
    distance: float = 0.60
    current_goal: str = 'observe_player'
    relationship_stage: str = 'stranger'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'CharacterState':
        base = CharacterState()
        for k, v in d.items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


EVENT_RULES = {
    'player_helped_npc': dict(mood='grateful', action='offer_warm_thanks', trust=0.10, affection=0.06, stress=-0.06, distance=-0.05, importance='medium'),
    'player_broke_promise': dict(mood='disappointed', action='ask_for_explanation', trust=-0.18, affection=-0.05, stress=0.10, distance=0.08, importance='high'),
    'player_gave_gift': dict(mood='moved', action='respond_shyly', trust=0.04, affection=0.10, stress=-0.03, distance=-0.03, importance='medium'),
    'player_lied': dict(mood='suspicious', action='test_player', trust=-0.22, affection=-0.06, stress=0.12, distance=0.12, importance='high'),
    'player_returned_after_long_absence': dict(mood='guarded', action='ask_why_returned', trust=-0.05, affection=-0.02, stress=0.06, curiosity=0.10, importance='medium'),
    'player_asked_personal_question': dict(mood='uneasy', action='set_boundary', trust=-0.03, stress=0.08, distance=0.07, importance='medium'),
    'player_protected_npc': dict(mood='relieved', action='reveal_small_secret', trust=0.18, affection=0.08, stress=-0.12, distance=-0.07, importance='high'),
    'player_ignored_npc': dict(mood='hurt', action='withdraw', trust=-0.12, affection=-0.08, stress=0.08, distance=0.10, importance='high'),
    'player_apologized': dict(mood='calm', action='accept_partially', trust=0.06, affection=0.02, stress=-0.04, distance=-0.02, importance='medium'),
    'player_completed_quest': dict(mood='grateful', action='offer_new_lead', trust=0.14, affection=0.05, stress=-0.08, distance=-0.04, importance='high'),
    'player_failed_quest': dict(mood='disappointed', action='assess_damage', trust=-0.08, affection=-0.03, stress=0.12, distance=0.04, importance='medium'),
}

PLAYER_TEXTS = {
    'player_helped_npc': ['我找到了你需要的药。', '外面的麻烦我已经处理好了，你暂时安全了。', '你之前说的那个人，我已经帮你确认过了。'],
    'player_broke_promise': ['抱歉，我昨天说会回来，但我没有做到。', '我知道我答应过你，但临时出了点事。', '我不是故意失约的，只是事情比我想的复杂。'],
    'player_gave_gift': ['我觉得你可能会喜欢这个。', '不是什么贵重东西，只是想送给你。', '路过集市时看到这个，突然想起了你。'],
    'player_lied': ['我没有对你隐瞒什么。', '我已经把知道的都告诉你了。', '你是不是想太多了？我没有骗你。'],
    'player_returned_after_long_absence': ['我知道我离开了很久。', '我回来，是因为我还想再见你一次。', '这段时间发生了很多事，我不知道该怎么解释。'],
    'player_asked_personal_question': ['你为什么从来不提自己的过去？', '你来到这里之前，到底是什么人？', '你总是在回避这个问题，为什么？'],
    'player_protected_npc': ['站到我身后，我来处理。', '我不会让他们伤害你。', '别怕，这次换我保护你。'],
    'player_ignored_npc': ['我现在没时间管这个。', '以后再说吧。', '这不是我现在优先要处理的事。'],
    'player_apologized': ['对不起，我当时应该听你的。', '是我错了，我想补救。', '我知道现在道歉可能太晚，但我还是想说出来。'],
    'player_completed_quest': ['你交代的事我完成了。', '包裹已经送到了，没有人跟踪我。', '那条线索是真的，我已经确认过。'],
    'player_failed_quest': ['我失败了，包裹被抢走了。', '我没能及时完成任务。', '对不起，事情变糟了。'],
}

DIALOGUE_TEMPLATES = {
    'ask_for_explanation': ['你昨天说会回来。我等了很久……发生什么了吗？', '我还没有生气。我只是想知道，你那句承诺到底算不算数。', '比起借口，我更想听实话。'],
    'offer_warm_thanks': ['我没想到你真的会帮到这个程度。谢谢你，我会记住的。', '你做得比我请求的更多。即使我不太会表达，这件事对我很重要。', '这次我欠你一个人情。别笑，我不是随便说这种话的人。'],
    'respond_shyly': ['你没必要特意带这个给我。不过……我会收好的。', '用礼物让我放松警惕，这招有点狡猾。可惜，好像确实有用。', '我不太习惯收礼物。但如果是你给的，我会认真保管。'],
    'test_player': ['你的话前后对不上。再说一次，这次说实话。', '我听够了漂亮的解释。我想听真正的原因。', '你以为我没有发现吗？我只是想看看你会不会主动承认。'],
    'ask_why_returned': ['你消失了这么久，现在又像什么都没发生一样回来。为什么？', '我以为你已经忘了这里。看来并不是。', '回来不是问题。问题是，你为什么现在才回来。'],
    'set_boundary': ['这不是我现在愿意回答的问题。', '有些门，只有在信任足够的时候才会打开。', '别再追问了。至少现在不要。'],
    'reveal_small_secret': ['有件事我一直没告诉你。不是全部，但足够让你理解一点。', '你明明可以袖手旁观，却还是保护了我。所以，我也给你一个真相。', '我不习惯把秘密交给别人。但这次，我愿意告诉你一小部分。'],
    'withdraw': ['我明白了。那我不会继续麻烦你。', '如果我的请求对你来说不重要，那我也该停止期待。', '没关系。以后我会自己处理。'],
    'accept_partially': ['道歉不能让一切恢复原样。但至少，这是一个开始。', '我听见了。只是我还需要时间判断这意味着什么。', '我接受你的道歉，但信任不是一句话就能修好的。'],
    'offer_new_lead': ['你完成了你的部分。那我也会履行我的承诺。有条线索，你应该知道。', '你证明了自己。下一步，我可以把更重要的情报告诉你。', '既然你守约，我也不会继续隐瞒。这是新的线索。'],
    'assess_damage': ['失败不是最糟的。现在重要的是弄清楚损失，以及怎么补救。', '不要回避结果。告诉我，到底是哪一步出了问题。', '东西丢了，时间也浪费了。但如果你还想补救，就先把情况说清楚。'],
    'stay_observant': ['我需要一点时间想清楚你刚才做的事。']
}


def update_relationship_stage(state: CharacterState):
    if state.trust > 0.78 and state.affection > 0.62:
        state.relationship_stage = 'close_companion'
    elif state.trust > 0.65:
        state.relationship_stage = 'trusted_ally'
    elif state.trust > 0.45:
        state.relationship_stage = 'cautious_ally'
    elif state.distance > 0.75 or state.trust < 0.25:
        state.relationship_stage = 'distant'
    else:
        state.relationship_stage = 'stranger'


def apply_event(state: CharacterState, event_type: str) -> Dict[str, Any]:
    rule = EVENT_RULES.get(event_type, dict(mood='neutral', action='stay_observant', importance='low'))
    before = state.to_dict()
    after_state = CharacterState.from_dict(before)
    after_state.mood = rule.get('mood', after_state.mood)
    for k in ['trust', 'affection', 'stress', 'curiosity', 'distance']:
        if k in rule:
            setattr(after_state, k, clamp(getattr(after_state, k) + rule[k]))
    if rule.get('action') in ['reveal_small_secret', 'offer_new_lead']:
        after_state.current_goal = 'share_carefully'
    elif rule.get('action') in ['test_player', 'ask_for_explanation']:
        after_state.current_goal = 'test_player_reliability'
    elif rule.get('action') in ['withdraw', 'set_boundary']:
        after_state.current_goal = 'protect_self'
    else:
        after_state.current_goal = 'observe_player'
    update_relationship_stage(after_state)
    return {
        'state_before': before,
        'state_after': after_state.to_dict(),
        'emotion': after_state.mood,
        'action': rule.get('action', 'stay_observant'),
        'importance': rule.get('importance', 'low'),
        'deltas': {k: rule.get(k, 0.0) for k in ['trust', 'affection', 'stress', 'curiosity', 'distance']}
    }


def choose_rule_dialogue(action: str, state_after: Dict[str, Any], sample_id: str = '') -> str:
    opts = DIALOGUE_TEMPLATES.get(action, DIALOGUE_TEMPLATES['stay_observant'])
    seed = abs(hash(str(state_after) + sample_id))
    return opts[seed % len(opts)]


def random_state(rng: random.Random) -> CharacterState:
    return CharacterState(
        trust=rng.uniform(0.2, 0.8), affection=rng.uniform(0.1, 0.7),
        stress=rng.uniform(0.05, 0.6), curiosity=rng.uniform(0.1, 0.8), distance=rng.uniform(0.2, 0.85)
    )


def event_options_cn():
    return [f'{v}（{k}）' for k, v in EVENT_ZH.items()]


def parse_event_from_cn(label: str) -> str:
    if '（' in label and label.endswith('）'):
        return label.split('（')[-1].strip('）')
    for k, v in EVENT_ZH.items():
        if label == v:
            return k
    return label


def state_cards_html(state: Dict[str, Any]) -> str:
    items = [
        ('当前心情', MOOD_ZH.get(state.get('mood'), state.get('mood'))),
        ('信任度', f"{state.get('trust', 0):.2f}"),
        ('亲近度', f"{state.get('affection', 0):.2f}"),
        ('压力值', f"{state.get('stress', 0):.2f}"),
        ('距离感', f"{state.get('distance', 0):.2f}"),
        ('关系阶段', RELATIONSHIP_ZH.get(state.get('relationship_stage'), state.get('relationship_stage'))),
    ]
    cards = ''.join([f"<div class='mini-card'><div class='mini-title'>{a}</div><div class='mini-value'>{b}</div></div>" for a,b in items])
    return f"<div class='mini-grid'>{cards}</div>"


@dataclass
class SoulNPC:
    state: CharacterState = field(default_factory=CharacterState)
    memories: List[str] = field(default_factory=list)

    def interact(self, event_type: str, player_text: str) -> Dict[str, Any]:
        result = apply_event(self.state, event_type)
        self.state = CharacterState.from_dict(result['state_after'])
        dialogue = choose_rule_dialogue(result['action'], result['state_after'])
        memory = f"玩家事件：{EVENT_ZH.get(event_type, event_type)}；玩家说：{player_text}；艾拉的反应：{MOOD_ZH.get(result['emotion'], result['emotion'])} / {ACTION_ZH.get(result['action'], result['action'])}。"
        self.memories.append(memory)
        self.memories = self.memories[-8:]
        return {**result, 'dialogue': dialogue, 'memory': memory, 'memories': list(self.memories)}
