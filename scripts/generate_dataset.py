import argparse
from src.data_generation.generator import generate_samples, write_jsonl
from src.data_generation.exporters import export_sft, export_dpo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    samples = generate_samples(args.n, args.seed)
    raw = write_jsonl(samples, 'data/raw/generated_events.jsonl')
    sft = export_sft(samples, 'data/export/final_train_sft.jsonl')
    dpo = export_dpo(samples, 'data/export/final_train_dpo.jsonl')
    print(f'生成完成: {len(samples)}')
    print(f'原始数据: {raw}')
    print(f'SFT: {sft}')
    print(f'DPO: {dpo}')

if __name__ == '__main__':
    main()
