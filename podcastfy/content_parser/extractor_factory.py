"""Factory for creating content extractors."""

from typing import ClassVar

from .extractor_base import ContentExtractor


class ExtractorFactory:
    _extractors: ClassVar[list[type[ContentExtractor]]] = []

    @classmethod
    def register(cls, extractor_cls: type[ContentExtractor]) -> None:
        cls._extractors.append(extractor_cls)

    @classmethod
    def create(cls, source: str) -> ContentExtractor | None:
        for ex_cls in cls._extractors:
            if ex_cls.can_handle(source):
                return ex_cls()
        return None
