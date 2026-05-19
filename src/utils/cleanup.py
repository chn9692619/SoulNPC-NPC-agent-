from pathlib import Path
import shutil

TARGETS = {
    '仅清理教师候选与临时数据': ['data/processed/teacher_candidates.jsonl'],
    '清理人工审阅与偏好数据': ['data/processed/reviewed_overrides.jsonl', 'data/preference/preference_pairs.jsonl'],
    '清理自动生成数据': ['data/raw/generated_events.jsonl'],
    '清理训练导出与输出': ['data/export', 'outputs'],
    '全量重置数据与输出': ['data/raw', 'data/processed', 'data/preference', 'data/export', 'outputs'],
}


def clean(mode: str, confirm: str) -> str:
    if confirm.strip() != '确认清理':
        return '未执行：请输入“确认清理”后再点击清理。'
    paths = TARGETS.get(mode, [])
    removed = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
            removed.append(str(path))
        elif path.exists():
            path.unlink()
            removed.append(str(path))
    return '清理完成：\n' + ('\n'.join(removed) if removed else '没有找到需要清理的文件。')
