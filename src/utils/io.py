import json
from pathlib import Path
from typing import Iterable, Dict, Any, List


def ensure_parent(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> str:
    p = Path(path)
    ensure_parent(p)
    with open(p, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return str(p)


def read_json(path: str | Path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(obj, path: str | Path) -> str:
    p = Path(path)
    ensure_parent(p)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(p)


def count_jsonl(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                n += 1
    return n
