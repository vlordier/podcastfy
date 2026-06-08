"""LiteLLM provider implementation."""

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..base import LLMProvider


class LiteLLM(LLMProvider):
    def __init__(self, api_key: str, model: str, **kwargs):
        self.model_name = model
        self.llm = ChatLiteLLM(
            model=model,
            temperature=kwargs.get("temperature", 1.0),
            api_key=api_key,
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
