"""Model loading helpers for SoulNPC cloud LoRA training.

The training scripts can load a base model from:
1. a local directory;
2. ModelScope (recommended on AutoDL/mainland China networks);
3. Hugging Face / optional mirror endpoint.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any


def resolve_base_model_path(cfg: Dict[str, Any]) -> str:
    """Return a local path or hub model id that Transformers can load.

    Config keys:
    - base_model: repo id, e.g. Qwen/Qwen2.5-1.5B-Instruct
    - model_source: local | modelscope | huggingface
    - local_model_path: direct local model directory, optional
    - model_cache_dir: cache directory for downloaded model, optional
    - hf_endpoint: optional Hugging Face endpoint/mirror
    """
    base_model = cfg.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")
    source = str(cfg.get("model_source", "modelscope")).lower().strip()
    local_model_path = str(cfg.get("local_model_path", "")).strip()
    cache_dir = str(cfg.get("model_cache_dir", "")).strip()

    if local_model_path:
        p = Path(local_model_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"local_model_path does not exist: {p}")
        print(f"[SoulNPC] Loading base model from local path: {p}", flush=True)
        return str(p)

    if source == "local":
        p = Path(base_model).expanduser()
        if not p.exists():
            raise FileNotFoundError(
                "model_source=local requires base_model to be an existing local directory. "
                f"Got: {base_model}"
            )
        print(f"[SoulNPC] Loading base model from local base_model path: {p}", flush=True)
        return str(p)

    if source == "modelscope":
        try:
            from modelscope import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "model_source=modelscope requires package 'modelscope'. "
                "Run: python -m pip install modelscope"
            ) from exc
        kwargs = {}
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = cache_dir
        print(f"[SoulNPC] Downloading/loading model from ModelScope: {base_model}", flush=True)
        model_dir = snapshot_download(base_model, **kwargs)
        print(f"[SoulNPC] ModelScope local model dir: {model_dir}", flush=True)
        return model_dir

    # Hugging Face path. Optional mirror endpoint can be set by config.
    hf_endpoint = str(cfg.get("hf_endpoint", "")).strip()
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"[SoulNPC] HF_ENDPOINT={hf_endpoint}", flush=True)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    print(f"[SoulNPC] Loading base model from Hugging Face id/path: {base_model}", flush=True)
    return base_model
