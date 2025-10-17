import logging
from vidavox.retriever.schema.data import SearchKind, EngineConfig
from vidavox.retrieval_system.utils.process_failure import ProcessFailure

logger = logging.getLogger(__name__)


class RetrieverComponents:
    """Manages retriever and search components with validation."""

    def __init__(self, retriever, config: EngineConfig):
        self.retriever = retriever
        self.config = config
        self._validate_and_initialize()

    def _validate_and_initialize(self):
        """Validate retriever and extract components."""
        try:
            if not hasattr(self.retriever, "model"):
                raise ValueError("Retriever missing 'model' attribute")

            # --- base references ---
            model = self.retriever.model
            self.bm25_wrapper = getattr(model, "bm25_search", None)
            self.faiss_wrapper = getattr(model, "faiss_search", None)

            # --- handle single-engine retrievers gracefully ---
            # If retriever *is* FAISSRetriever, its model *is already* a FAISS_search
            from vidavox.search.faiss_search import FAISS_search
            from vidavox.search.bm25_search import BM25_search

            if isinstance(model, FAISS_search):
                self.faiss_wrapper = model
            elif isinstance(model, BM25_search):
                self.bm25_wrapper = model

            self.hybrid_wrapper = model

            print(f"self.bm25_wrapper: {self.bm25_wrapper}")
            print(f"self.faiss_wrapper: {self.faiss_wrapper}")
            print(f"self.hybrid_wrapper: {self.hybrid_wrapper}")

            # --- validation ---
            if self.config.search_kind == SearchKind.BM25 and not self.bm25_wrapper:
                raise ValueError("BM25 search not available in retriever")

            if self.config.search_kind == SearchKind.FAISS and not self.faiss_wrapper:
                raise ValueError("FAISS search not available in retriever")

            if self.config.search_kind == SearchKind.HYBRID:
                if not self.bm25_wrapper or not self.faiss_wrapper:
                    raise ValueError(
                        "Hybrid search requires both BM25 and FAISS")

            logger.info(
                f"Retriever initialized: {self.config.search_kind.value}")

        except Exception as e:
            retriever_type = type(self.retriever).__name__
            model_type = type(getattr(self.retriever, "model", None)).__name__
            logger.error(
                f"RetrieverComponents initialization failed:\n"
                f"  Retriever type : {retriever_type}\n"
                f"  Model type     : {model_type}\n"
                f"  Search kind    : {self.config.search_kind}\n"
                f"  BM25 available : {hasattr(self.retriever.model, 'bm25_search')}\n"
                f"  FAISS available: {hasattr(self.retriever.model, 'faiss_search')}\n"
                f"  Error          : {e}",
                exc_info=True
            )
            raise ValueError(
                f"Failed to initialize RetrieverComponents: {e}") from e
