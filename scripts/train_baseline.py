import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib


def load_rows(path):
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def features(s):
    st = s['state_before']
    aff = st['affect']; rel = st['relationship']
    x = {f'affect_{k}':v for k,v in aff.items()}
    x.update({f'rel_{k}':v for k,v in rel.items()})
    for p,w in s.get('event_primitives',{}).items(): x[f'event_{p}']=w
    return x


def train_one(rows, label_key, out_dir):
    X = [features(r) for r in rows]
    y = [r[label_key] for r in rows]
    if len(set(y)) < 2:
        return f'{label_key}: 类别数不足，无法训练。'
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    pipe=Pipeline([('vec',DictVectorizer()),('clf',LogisticRegression(max_iter=1000,class_weight='balanced'))])
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte)
    acc=accuracy_score(yte,pred)
    rep=classification_report(yte,pred,zero_division=0)
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    joblib.dump(pipe, Path(out_dir)/f'{label_key}_classifier.joblib')
    with open(Path(out_dir)/f'{label_key}_report.txt','w',encoding='utf-8') as f:
        f.write(f'Accuracy: {acc:.4f}\n\n{rep}')
    return f'{label_key} Accuracy: {acc:.4f}\n{rep[:1200]}'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='data/raw/generated_events.jsonl')
    parser.add_argument('--out',default='outputs/baseline')
    args=parser.parse_args()
    rows=load_rows(args.input)
    print(f'Loaded {len(rows)} rows')
    print(train_one(rows,'derived_mood',args.out))
    print(train_one(rows,'action',args.out))

if __name__=='__main__': main()
