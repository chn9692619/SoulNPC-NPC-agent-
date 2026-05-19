from pathlib import Path
from typing import Dict, Any
from src.utils.io import read_json, write_json

CONFIG_PATH = Path('config/model_config.json')
DEFAULT_CONFIG = {
    'provider': 'mock',
    'api_key': '',
    'base_url': '',
    'model': 'mock',
}

PRESETS = {
    'Mock 本地模拟': {'provider': 'mock', 'base_url': '', 'model': 'mock'},
    'DeepSeek': {'provider': 'deepseek', 'base_url': 'https://api.deepseek.com', 'model': 'deepseek-chat'},
    '通义千问 / 阿里百炼': {'provider': 'qwen', 'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'model': 'qwen-plus'},
}


def load_model_config() -> Dict[str, Any]:
    cfg = read_json(CONFIG_PATH, default=None)
    if not cfg:
        write_json(DEFAULT_CONFIG, CONFIG_PATH)
        return dict(DEFAULT_CONFIG)
    out = dict(DEFAULT_CONFIG)
    out.update(cfg)
    return out


def save_model_config(provider_label: str, api_key: str, base_url: str, model: str) -> Dict[str, Any]:
    preset = PRESETS.get(provider_label, PRESETS['Mock 本地模拟'])
    cfg = {
        'provider': preset['provider'],
        'api_key': api_key or '',
        'base_url': base_url or preset['base_url'],
        'model': model or preset['model'],
    }
    write_json(cfg, CONFIG_PATH)
    return cfg


def current_mode_text() -> str:
    cfg = load_model_config()
    if cfg.get('provider') == 'mock':
        return '当前模式：Mock 本地模拟，不会调用真实 API。'
    return f"当前模式：真实 API。provider={cfg.get('provider')}，model={cfg.get('model')}，base_url={cfg.get('base_url')}"
