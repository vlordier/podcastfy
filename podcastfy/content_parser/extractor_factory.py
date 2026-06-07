"""Factory for creating content extractors."""

from typing import ClassVar, Optional, Type
from .extractor_base import ContentExtractor


class ExtractorFactory:
    _extractors: ClassVar[list[Type[ContentExtractor]]] = []

    @classmethod
    def register(cls, extractor_cls: Type[ContentExtractor]) -> None:
        cls._extractors.append(extractor_cls)

    @classmethod
    def create(cls, source: str) -> Optional[ContentExtractor]:
        for ex_cls in cls._extractors:
            if ex_cls.can_handle(source):
                return ex_cls()
        return None