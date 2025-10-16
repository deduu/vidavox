from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional


class BaseRetriever(ABC):
    """Abstract base class for retrievers (BM25, FAISS, Hybrid, etc.)."""

    @abstractmethod
    def add_documents(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        """Add multiple documents synchronously."""
        pass

    @abstractmethod
    async def add_documents_async(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        """Add multiple documents asynchronously."""
        pass

    @abstractmethod
    def remove_documents(self, doc_ids: List[str]) -> List[str]:
        """Remove multiple documents synchronously."""
        pass

    @abstractmethod
    async def async_remove_documents(self, doc_ids: List[str]) -> List[str]:
        """Remove multiple documents asynchronously."""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5, **kwargs) -> List[Tuple[str, str, float]]:
        """Search synchronously."""
        pass

    @abstractmethod
    async def async_search(self, query: str, k: int = 5, **kwargs) -> List[Tuple[str, str, float]]:
        """Search asynchronously."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist retriever index/state."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load retriever index/state."""
        pass

    @abstractmethod
    def clear_documents(self) -> None:
        """Clear all stored data."""
        pass

    @abstractmethod
    async def async_clear_documents(self) -> None:
        """Asynchronously clear all stored data."""
        pass
