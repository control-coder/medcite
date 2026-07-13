"""DeepSeek LLM 客户端封装。

使用 httpx 直接调用 OpenAI 兼容 API（DeepSeek），
不依赖 openai 库（减少依赖）。

环境变量:
    DEEPSEEK_API_KEY: API 密钥
    DEEPSEEK_BASE_URL: API 地址（默认 https://api.deepseek.com/v1）
"""

from __future__ import annotations

import os

import httpx


class LLMClient:
    """DeepSeek LLM 客户端。

    用法:
        client = LLMClient()
        response = client.chat("What is the diagnosis?")
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "deepseek-v4-flash-free",
        timeout: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        seed: int = 42,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        )
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    @property
    def is_configured(self) -> bool:
        """是否已配置 API key。"""
        return bool(self.api_key)

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """调用 LLM 生成回复。

        Args:
            prompt: 用户提示词。
            system_prompt: 系统提示词（可选）。
            temperature: 温度（默认 0，可复现）。
            max_tokens: 最大生成 token 数。

        Returns:
            LLM 生成的文本。

        Raises:
            ValueError: API key 未配置。
            httpx.HTTPStatusError: API 调用失败。
        """
        if not self.is_configured:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. Configure in .env or pass api_key."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                "seed": self.seed,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
