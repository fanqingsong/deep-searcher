import os
from typing import Dict, List
from deepsearcher.llm.base import BaseLLM, ChatResponse


class OpenAI(BaseLLM):
    def __init__(self, model: str = "gpt-4o", **kwargs):
        from openai import OpenAI as OpenAI_
        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI_(api_key=api_key, **kwargs)


    def chat(self, messages: List[Dict]) -> ChatResponse:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return ChatResponse(content=completion.choices[0].message.content, total_tokens=completion.usage.total_tokens)

