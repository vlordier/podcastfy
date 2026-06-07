"""Gemini LLM provider implementation."""
import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..base import LLMProvider
from podcastfy.utils.constants import (
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_MAX_OUTPUT_TOKENS,
)


class GeminiLLM(LLMProvider):
    def __init__(self, api_key: str, model: str, **kwargs):
        self.model_name = model
        common_params = {
            "temperature": kwargs.get("temperature", 1.0),
            "presence_penalty": DEFAULT_PRESENCE_PENALTY,
            "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
        }
        self.llm = ChatGoogleGenerativeAI(
            api_key=api_key or os.environ.get("GEMINI_API_KEY", ""),
            model=model,
            max_output_tokens=kwargs.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
            **common_params,
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