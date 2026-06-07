"""Abstract base class for content extractors."""

from abc import ABC, abstractmethod


class ContentExtractor(ABC):
    @classmethod
    @abstractmethod
    def can_handle(cls, source: str) -> bool:
        ...

    @abstractmethod
    def extract(self, source: str) -> str:
        ...