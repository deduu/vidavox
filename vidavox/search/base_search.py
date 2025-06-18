from abc import ABC, abstractmethod
from typing import List, Tuple

class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Tuple[str, float]]:
        """
        Searches synchronously for the given query and returns a list of tuples (document_id, score).
        """
        pass

    @abstractmethod
    async def async_search(self, query: str, **kwargs) -> List[Tuple[str, float]]:
        """
        Searches asynchronously for the given query and returns a list of tuples (document_id, score).
        """
        pass
    
    @abstractmethod
    def add_documents(self, docs: List[Tuple[str, str]]) -> None:
        """
        Adds documents synchronously to the provider.
        Each document is a tuple (doc_id, document_text).
        """
        pass

    @abstractmethod
    async def async_add_documents(self, docs: List[Tuple[str, str]]) -> None:
        """
        Asynchronously adds documents to the provider.
        """
        pass
    
    @abstractmethod
    def clear_documents(self) -> None:
        """
        Synchronously clears all documents and resets the provider state.
        """
        pass

    @abstractmethod
    async def async_clear_documents(self) -> None:
        """
        Asynchronously clears all documents and resets the provider state.
        """
        pass
