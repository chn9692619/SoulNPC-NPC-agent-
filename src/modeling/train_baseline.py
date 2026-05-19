from pathlib import Path
from typing import Dict, Any, List
import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

from src.utils.io import read_jsonl, write_jsonl
from src.data_generation.merge import build_final_rows, export_final_training_files


def row_to_text(s: Dict[str, Any]) -> str:
    return '\n'.join([
        s.get('event_type_zh', ''),
        s.get('event_type', ''),
        s.get('player_text', ''),
        json.dumps(s.get('state_before', {}), ensure_ascii=False),
        json.dumps(s.get('deltas', {}), ensure_ascii=False),
        s.get('dialogue', ''),
    ])


def train_classifier(rows: List[Dict[str, Any]], label_key: str, output_dir: Path) -> Dict[str, Any]:
    X = [row_to_text(r) for r in rows]
    y = [r.get(label_key, 'unknown') for r in rows]
    labels = sorted(set(y))
    if len(labels) < 2:
        raise RuntimeError(f'{label_key} 类别少于 2，无法训练分类器。')
    stratify = y if min(y.count(c) for c in labels) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, zero_division=0)
    acc = sum(p == t for p, t in zip(pred, y_test)) / max(1, len(y_test))

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f'{label_key}_report.txt'
    model_path = output_dir / f'{label_key}_classifier.joblib'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    joblib.dump(model, model_path)

    # confusion matrix
    try:
        cm = confusion_matrix(y_test, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.8), max(5, len(labels)*0.7)))
        im = ax.imshow(cm)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center')
        fig.tight_layout()
        cm_path = output_dir / f'{label_key}_confusion_matrix.png'
        fig.savefig(cm_path, dpi=160)
        plt.close(fig)
    except Exception:
        cm_path = ''

    return {
        'label_key': label_key,
        'accuracy': acc,
        'classes': labels,
        'report_path': str(report_path),
        'model_path': str(model_path),
        'confusion_matrix_path': str(cm_path),
        'report': report,
        'test_size': len(y_test),
        'train_size': len(y_train),
    }


def train_final_baseline(output_dir: str = 'outputs/final_baseline') -> Dict[str, Any]:
    merge_info = export_final_training_files()
    rows = build_final_rows()
    if len(rows) < 10:
        raise RuntimeError('最终训练样本少于 10 条，请先生成更多基础数据。')
    out = Path(output_dir)
    emotion = train_classifier(rows, 'emotion', out)
    action = train_classifier(rows, 'action', out)
    summary = {
        'merge_info': merge_info,
        'emotion': emotion,
        'action': action,
        'output_dir': str(out),
    }
    with open(out / 'training_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
