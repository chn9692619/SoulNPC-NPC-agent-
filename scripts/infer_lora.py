"""SoulNPC LoRA 推理脚本。
在 AutoDL / Linux CUDA 环境运行，用于加载基础模型 + LoRA adapter 生成 NPC 台词。
"""
import argparse, json
from pathlib import Path
from scripts.model_utils import resolve_base_model_path


def build_prompt(user_text: str) -> list:
    system = (
        "你是 SoulNPC 中的可信游戏角色台词生成模型。"
        "你需要根据角色人格、情绪状态、关系状态、记忆和行为意图，"
        "生成自然、克制、符合人设的中文 NPC 台词。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default_training_config.json')
    parser.add_argument('--adapter', default=None)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max_new_tokens', type=int, default=180)
    args = parser.parse_args()

    cfg = json.load(open(args.config, 'r', encoding='utf-8'))
    base_model = resolve_base_model_path(cfg)
    adapter = args.adapter or cfg.get('output_dir', 'outputs/lora_sft')

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    messages = build_prompt(args.prompt)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    new_tokens = out[0][inputs['input_ids'].shape[-1]:]
    print(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


if __name__ == '__main__':
    main()
