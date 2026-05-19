"""SoulNPC DPO LoRA 训练脚本模板。

建议在已有 SFT adapter 基础上继续训练。此脚本是可运行骨架，具体参数请根据显存调整。
"""
import argparse, json
from scripts.model_utils import resolve_base_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--train_file', default='data/export/final_train_dpo.jsonl')
    parser.add_argument('--output_dir', default='outputs/lora_dpo')
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model
    try:
        from trl import DPOTrainer
    except Exception as e:
        raise RuntimeError('请先安装 trl: pip install trl') from e

    tokenizer = AutoTokenizer.from_pretrained(resolve_base_model_path({'base_model': args.base_model, 'model_source': 'modelscope'}), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(resolve_base_model_path({'base_model': args.base_model, 'model_source': 'modelscope'}), trust_remote_code=True, device_map='auto')
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type='CAUSAL_LM'))
    ds = load_dataset('json', data_files=args.train_file, split='train')

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        fp16=True,
        report_to='none'
    )
    trainer = DPOTrainer(model=model, args=train_args, train_dataset=ds, tokenizer=tokenizer)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print('DPO LoRA training finished.')

if __name__ == '__main__':
    main()
