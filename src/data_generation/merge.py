from typing import Dict, Any, List
from src.utils.io import read_jsonl, write_jsonl
from src.data_generation.generator import export_sft, export_dpo

RAW_PATH = 'data/raw/generated_events.jsonl'
OVERRIDE_PATH = 'data/processed/reviewed_overrides.jsonl'
PREF_PATH = 'data/preference/preference_pairs.jsonl'
FINAL_SFT = 'data/export/final_train_sft.jsonl'
FINAL_DPO = 'data/export/final_train_dpo.jsonl'


def load_overrides() -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(OVERRIDE_PATH)
    return {r.get('sample_id'): r for r in rows if r.get('sample_id')}


def build_final_rows() -> List[Dict[str, Any]]:
    raw = read_jsonl(RAW_PATH)
    overrides = load_overrides()
    final = []
    for s in raw:
        sid = s.get('sample_id')
        if sid in overrides:
            ov = overrides[sid]
            s = dict(s)
            chosen = ov.get('chosen_text') or ov.get('edited_text') or s.get('dialogue')
            s['dialogue'] = chosen
            s['sft_output'] = dict(s.get('sft_output', {}))
            s['sft_output']['dialogue'] = chosen
            s['reviewed'] = True
            s['review_note'] = ov.get('note', '')
        final.append(s)
    return final


def export_final_training_files() -> Dict[str, Any]:
    raw = read_jsonl(RAW_PATH)
    overrides = read_jsonl(OVERRIDE_PATH)
    prefs = read_jsonl(PREF_PATH)
    final_rows = build_final_rows()
    sft_path = export_sft(final_rows, FINAL_SFT)
    if prefs:
        dpo_path = write_jsonl(prefs, FINAL_DPO)
        dpo_count = len(prefs)
        dpo_source = '人工偏好数据'
    else:
        dpo_path = export_dpo(final_rows, FINAL_DPO)
        dpo_count = len(final_rows)
        dpo_source = '自动构造偏好数据'
    return {
        'raw_count': len(raw),
        'override_count': len(overrides),
        'final_sft_count': len(final_rows),
        'final_dpo_count': dpo_count,
        'dpo_source': dpo_source,
        'sft_path': sft_path,
        'dpo_path': dpo_path,
    }
