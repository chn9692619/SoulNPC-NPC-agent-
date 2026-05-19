from typing import Dict, Any, List
from src.utils.io import read_jsonl, write_jsonl
from src.modeling.teacher_client import call_teacher_model, mock_candidates

RAW_PATH = 'data/raw/generated_events.jsonl'
CANDIDATE_PATH = 'data/processed/teacher_candidates.jsonl'


def generate_teacher_candidates(limit: int = 5, k: int = 3, use_real_api: bool = False) -> Dict[str, Any]:
    raw = read_jsonl(RAW_PATH)
    if not raw:
        raise RuntimeError('缺少基础样本，请先在数据工厂生成基础训练样本。')
    selected = raw[:max(0, min(limit, len(raw)))]
    rows = []
    logs = []
    for idx, sample in enumerate(selected, start=1):
        if use_real_api:
            cands = call_teacher_model(sample, k)
            source = '真实教师模型候选'
        else:
            cands = mock_candidates(sample, k)
            source = 'Mock 本地候选'
        row = {
            'sample_id': sample['sample_id'],
            'source': source,
            'event_type': sample['event_type'],
            'event_type_zh': sample.get('event_type_zh', ''),
            'player_text': sample.get('player_text', ''),
            'sample': sample,
            'candidates': cands,
        }
        rows.append(row)
        logs.append(f"{idx}/{len(selected)} {sample['sample_id']}：{source}")
    write_jsonl(rows, CANDIDATE_PATH)
    return {'count': len(rows), 'path': CANDIDATE_PATH, 'logs': '\n'.join(logs)}
