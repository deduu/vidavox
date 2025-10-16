from typing import List, Tuple, Dict, Any
from vidavox.retriever.base import BaseRetriever
from vidavox.search.bm25_search import BM25_search


class BM25Retriever(BaseRetriever):
    """Adapter class that wraps BM25_search into a unified retriever interface."""

    def __init__(self, remove_stopwords: bool = True, perform_lemmatization: bool = False):
        self.model = BM25_search(remove_stopwords, perform_lemmatization)

    def add_documents(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        self.model.add_documents(docs_with_ids)

    async def add_documents_async(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        await self.model.add_documents_async(docs_with_ids)

    def remove_documents(self, doc_ids: List[str]) -> List[str]:
        return self.model.remove_documents(doc_ids)

    async def async_remove_documents(self, doc_ids: List[str]) -> List[str]:
        return await self.model.async_remove_documents(doc_ids)

    def search(self, query: str, k: int = 5, **kwargs) -> List[Tuple[str, str, float]]:
        return self.model.get_top_n_docs(query, n=k, **kwargs)

    async def async_search(self, query: str, k: int = 5, **kwargs) -> List[Tuple[str, str, float]]:
        return await self.model.async_get_top_n_docs(query, n=k, **kwargs)

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model.load(path)

    def clear_documents(self) -> None:
        self.model.clear_documents()

    async def async_clear_documents(self) -> None:
        await self.model.async_clear_documents()
