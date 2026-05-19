"""SoulNPC LoRA 云端推理服务。
在 AutoDL 上运行，默认监听 0.0.0.0:6006，可通过 AutoDL 端口映射访问。
"""
import argparse, json
from scripts.model_utils import resolve_base_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default_training_config.json')
    parser.add_argument('--adapter', default=None)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=6006)
    args = parser.parse_args()

    cfg = json.load(open(args.config, 'r', encoding='utf-8'))
    base_model = resolve_base_model_path(cfg)
    adapter = args.adapter or cfg.get('output_dir', 'outputs/lora_sft')

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch
    import gradio as gr

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    system = (
        "你是 SoulNPC 中的可信游戏角色台词生成模型。"
        "你需要根据角色人格、情绪状态、关系状态、记忆和行为意图，"
        "生成自然、克制、符合人设的中文 NPC 台词。"
    )

    def generate(prompt, max_new_tokens=180, temperature=0.7):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=0.9,
            )
        new_tokens = out[0][inputs['input_ids'].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    demo = gr.Interface(
        fn=generate,
        inputs=[gr.Textbox(lines=10, label='SoulNPC 推理输入'), gr.Slider(32, 512, value=180, step=16, label='最大生成长度'), gr.Slider(0.1, 1.2, value=0.7, step=0.1, label='temperature')],
        outputs=gr.Textbox(lines=8, label='NPC 台词输出'),
        title='SoulNPC LoRA 云端推理服务',
    )
    demo.launch(server_name=args.host, server_port=args.port, show_error=True)


if __name__ == '__main__':
    main()
