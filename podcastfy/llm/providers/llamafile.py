"""Llamafile LLM provider implementation."""

from typing import Optional
from langchain_community.llms.llamafile import Llamafile
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..base import LLMProvider


class LlamafileLLM(LLMProvider):
    def __init__(self, api_key: str = "", model: str = "local", **kwargs):
        self.model_name = model
        self.llm = Llamafile()

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