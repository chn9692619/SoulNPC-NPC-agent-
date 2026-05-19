"""SoulNPC SFT LoRA 训练脚本模板。

建议在 AutoDL / Linux CUDA 环境运行。
本脚本假设数据格式为 data/export/final_train_sft.jsonl 中的 messages 格式。
"""
import argparse, json
from pathlib import Path
from scripts.model_utils import resolve_base_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default_training_config.json')
    args = parser.parse_args()
    cfg = json.load(open(args.config, 'r', encoding='utf-8'))

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    try:
        from transformers import BitsAndBytesConfig
        import torch
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if cfg.get('use_4bit', True) else None
    except Exception:
        bnb = None

    model_name = resolve_base_model_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='auto', quantization_config=bnb)
    if cfg.get('use_4bit', True):
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=cfg.get('lora_r',8), lora_alpha=cfg.get('lora_alpha',16), lora_dropout=cfg.get('lora_dropout',0.05),
        target_modules=cfg.get('target_modules'), task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, lora)
    ds = load_dataset('json', data_files=cfg['train_file'], split='train')

    def format_messages(ex):
        text = tokenizer.apply_chat_template(ex['messages'], tokenize=False, add_generation_prompt=False)
        tok = tokenizer(text, truncation=True, max_length=cfg.get('max_seq_length',1024), padding=False)
        tok['labels'] = tok['input_ids'].copy()
        return tok

    ds = ds.map(format_messages, remove_columns=ds.column_names)
    args_train = TrainingArguments(
        output_dir=cfg.get('output_dir','outputs/lora_sft'),
        per_device_train_batch_size=cfg.get('per_device_train_batch_size',1),
        gradient_accumulation_steps=cfg.get('gradient_accumulation_steps',8),
        learning_rate=cfg.get('learning_rate',2e-4),
        num_train_epochs=cfg.get('num_train_epochs',1),
        logging_steps=10,
        save_steps=200,
        fp16=True,
        report_to='none'
    )
    from transformers import DataCollatorForLanguageModeling, Trainer
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args_train, train_dataset=ds, data_collator=collator)
    trainer.train()
    model.save_pretrained(cfg.get('output_dir','outputs/lora_sft'))
    tokenizer.save_pretrained(cfg.get('output_dir','outputs/lora_sft'))
    print('SFT LoRA training finished.')

if __name__ == '__main__':
    main()
