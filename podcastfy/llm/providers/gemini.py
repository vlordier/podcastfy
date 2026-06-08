"""Gemini LLM provider implementation."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from podcastfy.utils.constants import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_PRESENCE_PENALTY,
)

from ..base import LLMProvider


class GeminiLLM(LLMProvider):
    def __init__(self, api_key: str, model: str, **kwargs):
        self.model_name = model
        common_params = {
            "temperature": kwargs.get("temperature", 1.0),
            "presence_penalty": DEFAULT_PRESENCE_PENALTY,
            "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
        }
        self.llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model=model,
            max_output_tokens=kwargs.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
            **common_params,
        )

    def generate(
        self,
        prompt: str,
        images: list[str] | None = None,
        config_conversation: dict | None = None,
    ) -> str:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("human", prompt),
            ]
        )
        chain = prompt_template | self.llm | StrOutputParser()
        return chain.invoke({})
