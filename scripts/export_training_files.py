import argparse

from src.data_generation.generator import read_jsonl
from src.data_generation.exporters import export_sft, export_preference_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/generated_events.jsonl")
    parser.add_argument("--sft_output", type=str, default="data/export/train_sft.jsonl")
    parser.add_argument("--pref_output", type=str, default="data/export/train_dpo.jsonl")
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    print(f"Loaded samples: {len(samples)}")
    print(export_sft(samples, args.sft_output))
    print(export_preference_pairs(samples, args.pref_output))


if __name__ == "__main__":
    main()
