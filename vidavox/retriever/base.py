from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional


class BaseRetriever(ABC):
    """Abstract base class for retrievers (BM25, FAISS, Hybrid, etc.)."""

    # ─── Core CRUD ──────────────────────────────────────────────
    @abstractmethod
    def add_documents(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        pass

    @abstractmethod
    async def add_documents_async(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        pass

    @abstractmethod
    def remove_documents(self, doc_ids: List[str]) -> List[str]:
        pass

    @abstractmethod
    async def async_remove_documents(self, doc_ids: List[str]) -> List[str]:
        pass

    # ─── Core Search ─────────────────────────────────────────────
    @abstractmethod
    def search(self, query: str, k: int = 5, **kwargs) -> List[Tuple[str, str, float]]:
        pass

    @abstractmethod
    async def async_search(self, query: str, k: int = 5, **kwargs) -> List[Tuple[str, str, float]]:
        pass

    # ─── Optional Batch Search ──────────────────────────────────
    def search_batch(
        self,
        queries: List[str],
        k: int = 5,
        **kwargs
    ) -> List[List[Tuple[str, str, float]]]:
        """
        Optional: Batch synchronous search.
        Subclasses should override if supported.
        Default implementation loops over .search().
        """
        return [self.search(q, k, **kwargs) for q in queries]

    async def async_search_batch(
        self,
        queries: List[str],
        k: int = 5,
        **kwargs
    ) -> List[List[Tuple[str, str, float]]]:
        """
        Optional: Batch asynchronous search.
        Subclasses should override if they can parallelize.
        """
        return [await self.async_search(q, k, **kwargs) for q in queries]

    # ─── Persistence ────────────────────────────────────────────
    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def clear_documents(self) -> None:
        pass

    @abstractmethod
    async def async_clear_documents(self) -> None:
        pass
