from typing import Dict, Any, List, Tuple
from src.utils.io import read_jsonl, write_jsonl

CANDIDATE_PATH = 'data/processed/teacher_candidates.jsonl'
OVERRIDE_PATH = 'data/processed/reviewed_overrides.jsonl'
PREF_PATH = 'data/preference/preference_pairs.jsonl'


def load_candidate(index: int) -> Dict[str, Any]:
    rows = read_jsonl(CANDIDATE_PATH)
    if not rows:
        raise RuntimeError('没有教师候选数据。请先在数据工厂生成候选回复。')
    index = max(0, min(index, len(rows)-1))
    return rows[index]


def candidate_count() -> int:
    return len(read_jsonl(CANDIDATE_PATH))


def save_review(index: int, choice: str, edited_text: str = '', note: str = '') -> Dict[str, Any]:
    cand = load_candidate(index)
    candidates = cand.get('candidates', [])
    choice_map = {'A': 0, 'B': 1, 'C': 2}
    chosen_idx = choice_map.get(choice, 0)
    chosen = edited_text.strip() or (candidates[chosen_idx] if chosen_idx < len(candidates) else (candidates[0] if candidates else ''))
    rejected = None
    for i, c in enumerate(candidates):
        if i != chosen_idx:
            rejected = c
            break
    if not rejected:
        rejected = '我没有任何感觉，也不需要回应。'

    overrides = read_jsonl(OVERRIDE_PATH)
    overrides = [x for x in overrides if x.get('sample_id') != cand.get('sample_id')]
    overrides.append({
        'sample_id': cand.get('sample_id'),
        'chosen_text': chosen,
        'choice': choice,
        'note': note,
        'source': cand.get('source', ''),
    })
    write_jsonl(overrides, OVERRIDE_PATH)

    prefs = read_jsonl(PREF_PATH)
    prefs = [x for x in prefs if x.get('metadata', {}).get('sample_id') != cand.get('sample_id')]
    prefs.append({
        'prompt': str(cand.get('sample', {}).get('sft_prompt', '')),
        'chosen': chosen,
        'rejected': rejected,
        'metadata': {
            'sample_id': cand.get('sample_id'),
            'choice': choice,
            'event_type': cand.get('event_type'),
            'event_type_zh': cand.get('event_type_zh'),
        }
    })
    write_jsonl(prefs, PREF_PATH)
    return {'sample_id': cand.get('sample_id'), 'chosen': chosen, 'rejected': rejected}
