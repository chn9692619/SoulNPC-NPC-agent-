import argparse

from src.data_generation.generator import read_jsonl
from src.data_generation.teacher import batch_generate_teacher_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/generated_events.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/teacher_candidates.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--num_candidates", type=int, default=3)
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    path = batch_generate_teacher_candidates(
        samples=samples,
        output_path=args.output,
        limit=args.limit,
        num_candidates=args.num_candidates,
    )
    print(f"Teacher candidates saved: {path}")


if __name__ == "__main__":
    main()
