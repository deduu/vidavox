from typing import Literal, Optional
from vidavox.retriever.bm25_retriever import BM25Retriever
from vidavox.retriever.faiss_retriever import FAISSRetriever
from vidavox.retriever.hybrid_retriever import HybridRetriever
from vidavox.search.hybrid_search import SearchMode

RetrieverType = Literal["bm25", "faiss", "hybrid"]


class RetrieverFactory:
    """Central factory for initializing modular retrievers."""

    @staticmethod
    def create(
        kind: RetrieverType = "hybrid",
        embedding_model: Optional[str] = None,
        bm25: Optional[BM25Retriever] = None,
        faiss: Optional[FAISSRetriever] = None,
        **kwargs,
    ):
        if kind == "bm25":
            return BM25Retriever()
        elif kind == "faiss":
            return FAISSRetriever(embedding_model)
        elif kind == "hybrid":
            # reuse or create base retrievers
            bm25 = bm25 or BM25Retriever()
            faiss = faiss or FAISSRetriever(embedding_model)
            return HybridRetriever(
                bm25.model,
                faiss.model,
                search_mode=kwargs.get("search_mode", SearchMode.HYBRID),
            )
        else:
            raise ValueError(f"Unknown retriever type: {kind}")
