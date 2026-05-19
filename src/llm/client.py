import os
import requests
from dotenv import load_dotenv


load_dotenv()


class LLMClient:
    def __init__(self):
        load_dotenv(override=True)
        self.provider = os.getenv("LLM_PROVIDER", "mock")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "")
        self.model = os.getenv("LLM_MODEL", "")

    def refresh(self):
        load_dotenv(override=True)
        self.provider = os.getenv("LLM_PROVIDER", "mock")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "")
        self.model = os.getenv("LLM_MODEL", "")

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        self.refresh()
        if self.provider == "mock":
            return self._mock_response(prompt)
        if self.provider == "openai_compatible":
            return self._call_openai_compatible(prompt, system_prompt=system_prompt, temperature=temperature)
        raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")

    def _mock_response(self, prompt: str) -> str:
        # 用于无 API 环境下跑通产品线。真实候选生成请在工作台里配置模型接口。
        if "候选" in prompt or "台词" in prompt:
            return (
                "1. 你来晚了。我确实等了很久，但我更想知道，你为什么没有回来。\n"
                "2. 我不会说自己完全不在意。只是现在，我还愿意听你解释一次。\n"
                "3. 你至少回来了。这不代表我已经原谅你，但说明这件事还有继续谈的余地。"
            )
        return "我需要一点时间判断，你这次说的话有多少是真的。"

    def _call_openai_compatible(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        if not self.api_key or not self.base_url or not self.model:
            raise ValueError("LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL must be set.")

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or (
                        "你是 SoulNPC 的教师模型，负责生成可信游戏角色的中文台词。"
                        "你必须保持人格、情绪、记忆和行为意图的一致性。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=80)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
