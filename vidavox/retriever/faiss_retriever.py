from typing import List, Tuple, Dict, Any
from vidavox.retriever.base import BaseRetriever
from vidavox.search.faiss_search import FAISS_search


class FAISSRetriever(BaseRetriever):
    """Adapter class that wraps FAISS_search into a unified retriever interface."""

    def __init__(self, embedding_model: Any = None):
        self.model = FAISS_search(embedding_model)

    def add_documents(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        self.model.add_documents(docs_with_ids)

    async def add_documents_async(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        await self.model.add_documents_async(docs_with_ids)

    def remove_documents(self, doc_ids: List[str]) -> List[str]:
        return self.model.remove_documents(doc_ids)

    async def async_remove_documents(self, doc_ids: List[str]) -> List[str]:
        return await self.model.async_remove_documents(doc_ids)

    def search(self, query: str, k: int = 5, **kwargs) -> List[Tuple[str, str, float]]:
        distances, ids = self.model.search(query, k)
        results = []
        for d, doc_id in zip(distances, ids):
            text = self.model.get_document(doc_id)
            results.append((doc_id, text, float(d)))
        return results

    async def async_search(self, query: str, k: int = 5, **kwargs) -> List[Tuple[str, str, float]]:
        distances, ids = await self.model.async_search(query, k)
        results = []
        for d, doc_id in zip(distances, ids):
            text = await self.model.async_get_document(doc_id)
            results.append((doc_id, text, float(d)))
        return results

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model.load(path)

    def clear_documents(self) -> None:
        self.model.clear_documents()

    async def async_clear_documents(self) -> None:
        await self.model.async_clear_documents()
