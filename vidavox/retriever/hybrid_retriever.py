from typing import List, Tuple, Optional, Dict
from vidavox.retriever.base import BaseRetriever
from vidavox.search.hybrid_search import Hybrid_search, SearchMode


class HybridRetriever(BaseRetriever):
    """
    Adapter to integrate Hybrid_search with the BaseRetriever interface.
    """

    def __init__(
        self,
        bm25_search,
        faiss_search,
        reranker_model_name: str = "BAAI/bge-reranker-v2-gemma",
        initial_bm25_weight: float = 0.5,
        search_mode: SearchMode = SearchMode.HYBRID,
    ):
        self.model = Hybrid_search(
            bm25_search=bm25_search,
            faiss_search=faiss_search,
            reranker_model_name=reranker_model_name,
            initial_bm25_weight=initial_bm25_weight,
            search_mode=search_mode,
        )

    # ─── Core CRUD wrappers ──────────────────────────────────────────────
    def add_documents(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        """Add to both BM25 and FAISS indices."""
        self.model.bm25_search.add_documents(docs_with_ids)
        self.model.faiss_search.add_documents(docs_with_ids)

    async def add_documents_async(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        await self.model.bm25_search.add_documents_async(docs_with_ids)
        await self.model.faiss_search.add_documents_async(docs_with_ids)

    def remove_documents(self, doc_ids: List[str]) -> List[str]:
        result = self.model.remove_documents(doc_ids)
        # Combine unique removed IDs from both
        return sorted(set(result["bm25_removed"]) | set(result["faiss_removed"]))

    async def async_remove_documents(self, doc_ids: List[str]) -> List[str]:
        result = await self.model.async_remove_documents(doc_ids)
        return sorted(set(result["bm25_removed"]) | set(result["faiss_removed"]))

    # ─── Unified search entrypoints ─────────────────────────────────────
    def search(
        self,
        query: str,
        k: int = 5,
        **kwargs,
    ) -> List[Tuple[str, str, float]]:
        """
        Returns list of (doc_id, text, score)
        """
        hybrid_results = self.model.advanced_search(
            query,
            keywords=kwargs.get("keywords"),
            top_n=k,
            threshold=kwargs.get("threshold", 0.5),
            prefixes=kwargs.get("prefixes"),
            include_doc_ids=kwargs.get("include_doc_ids"),
            exclude_doc_ids=kwargs.get("exclude_doc_ids"),
        )

        results = []
        for doc_id, score in hybrid_results:
            text = self.model.bm25_search.doc_dict.get(
                doc_id, {}).get("text", "")
            results.append((doc_id, text, float(score)))
        return results

    async def async_search(
        self,
        query: str,
        k: int = 5,
        **kwargs,
    ) -> List[Tuple[str, str, float]]:
        hybrid_results = await self.model.advanced_search_async(
            query,
            keywords=kwargs.get("keywords"),
            top_n=k,
            threshold=kwargs.get("threshold", 0.5),
            prefixes=kwargs.get("prefixes"),
            include_doc_ids=kwargs.get("include_doc_ids"),
            exclude_doc_ids=kwargs.get("exclude_doc_ids"),
        )
        results = []
        for doc_id, score in hybrid_results:
            text = self.model.bm25_search.doc_dict.get(
                doc_id, {}).get("text", "")
            results.append((doc_id, text, float(score)))
        return results

    # ─── Persistence (delegated) ────────────────────────────────────────
    def save(self, path: str) -> None:
        self.model.bm25_search.save(f"{path}/bm25")
        self.model.faiss_search.save(f"{path}/faiss")

    def load(self, path: str) -> None:
        self.model.bm25_search.load(f"{path}/bm25")
        self.model.faiss_search.load(f"{path}/faiss")

    def clear_documents(self) -> None:
        self.model.bm25_search.clear_documents()
        self.model.faiss_search.clear_documents()

    async def async_clear_documents(self) -> None:
        await self.model.bm25_search.async_clear_documents()
        await self.model.faiss_search.async_clear_documents()
