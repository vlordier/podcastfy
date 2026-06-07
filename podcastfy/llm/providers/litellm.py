"""LiteLLM provider implementation."""

import os
from typing import Optional
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..base import LLMProvider


class LiteLLM(LLMProvider):
    def __init__(self, api_key: str, model: str, **kwargs):
        self.model_name = model
        self.llm = ChatLiteLLM(
            model=model,
            temperature=kwargs.get("temperature", 1.0),
            api_key=api_key or os.environ.get(kwargs.get("api_key_label", "OPENAI_API_KEY"), ""),
        )

    def generate(
        self,
        prompt: str,
        images: Optional[list[str]] = None,
        config_conversation: Optional[dict] = None,
    ) -> str:
        prompt_template = ChatPromptTemplate.from_messages([
            ("human", prompt),
        ])
        chain = prompt_template | self.llm | StrOutputParser()
        return chain.invoke({})