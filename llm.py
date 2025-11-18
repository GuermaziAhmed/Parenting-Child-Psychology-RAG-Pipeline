"""LLM client wrapper for OpenRouter via openai library.
Gracefully degrades if API key missing or authentication fails.
"""
from __future__ import annotations
import os
from typing import List
from openai import OpenAI
from config import OPENROUTER_ENV_VAR, LLM_MODEL

FALLBACK_ANSWER = (
    "[LLM unavailable] Unable to retrieve a model-generated answer right now. "
    "Please verify your OPENROUTER_API_KEY or internet connectivity."
)

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv(OPENROUTER_ENV_VAR)
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self._client = None
        if self.api_key:
            try:
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                print(f"Failed to init OpenAI client: {e}")
                self._client = None
        else:
            print(f"Environment variable {OPENROUTER_ENV_VAR} not set; using fallback responses.")

    def chat(self, prompt: str) -> str:
        if not self._client:
            return FALLBACK_ANSWER
        try:
            resp = self._client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            # Newer openai python may return .choices list with message
            choice = resp.choices[0]
            content = getattr(choice.message, "content", None) or getattr(choice, "text", "")
            return content.strip()
        except Exception as e:
            print(f"LLM request failed: {e}")
            return FALLBACK_ANSWER
