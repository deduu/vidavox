import logging
from pathlib import Path


from vidavox.retriever.components.retriever import RetrieverComponents
from vidavox.retriever.schema.data import EngineConfig
from vidavox.retriever.managers.doc_manager import DocumentManager
from vidavox.retriever.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class IndexManager:
    """Handles loading/saving indices with validation and error recovery."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        components: RetrieverComponents,
        doc_manager: DocumentManager,
        token_counter: TokenCounter,
    ):
        self.config = config
        self.components = components
        self.doc_manager = doc_manager
        self.token_counter = token_counter

    def save(self, directory: Path) -> bool:
        """Save indices to disk with comprehensive error handling."""
        try:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)

            self._save_metadata(directory)
            self._save_documents(directory, self.doc_manager)
            self._save_search_indices(directory, self.components)

            logger.info(f"✓ Indices saved to {directory}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to save indices: {e}", exc_info=True)
            return False

    def load(self, directory: Path) -> bool:
        """Load indices from disk with validation."""
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Index directory not found: {directory}")
            return False

        try:
            self._validate_metadata(directory)
            self._load_search_indices(directory, self.components)
            self._load_documents(directory, self.components,
                                 self.doc_manager, self.token_counter)
            logger.info(
                f"✓ Loaded {len(self.doc_manager.documents)} documents from {directory}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load indices: {e}", exc_info=True)
            return False

    def _save_metadata(self, directory: Path):
        """Save engine metadata."""
        import json
        meta = {
            "embedding_model": self.config.embedding_model,
            "search_kind": self.config.search_kind.value,
            "version": "2.0",
        }
        (directory / "meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8"
        )

    def _save_documents(self, directory: Path, doc_manager: DocumentManager):
        """Save documents with compression."""
        import pickle
        import gzip

        docs_data = {
            doc_id: (doc.text, doc.meta_data)
            for doc_id, doc in doc_manager.documents.items()
        }

        compressed = gzip.compress(
            pickle.dumps(docs_data, protocol=pickle.HIGHEST_PROTOCOL)
        )
        (directory / "docs.pkl.gz").write_bytes(compressed)
        logger.debug(f"Saved {len(docs_data)} documents")

    def _save_search_indices(self, directory: Path, components: RetrieverComponents):
        """Save search indices."""
        if components.faiss_wrapper:
            components.faiss_wrapper.save(directory)
        if components.bm25_wrapper:
            components.bm25_wrapper.save(directory)

    def _validate_metadata(self, directory: Path):
        """Validate metadata matches current config."""
        import json

        meta_file = directory / "meta.json"
        if not meta_file.exists():
            raise ValueError("Missing meta.json in index directory")

        meta = json.loads(meta_file.read_text(encoding="utf-8"))

        if meta.get("embedding_model") != self.config.embedding_model:
            logger.warning(
                f"Model mismatch: saved={meta.get('embedding_model')}, "
                f"current={self.config.embedding_model}"
            )

    def _load_search_indices(self, directory: Path, components: RetrieverComponents):
        """Load search indices."""
        if components.faiss_wrapper:
            components.faiss_wrapper.load(directory)
        if components.bm25_wrapper:
            components.bm25_wrapper.load(directory)

    def _load_documents(
        self,
        directory: Path,
        components: RetrieverComponents,
        doc_manager: DocumentManager,
        token_counter: TokenCounter,
    ):
        """Load documents and populate managers with backward compatibility."""
        import pickle
        import gzip

        docs_file = directory / "docs.pkl.gz"
        if not docs_file.exists():
            logger.warning("No documents file found")
            return

        raw_data = pickle.loads(gzip.decompress(docs_file.read_bytes()))

        # Handle both old (2-tuple) and new (3-tuple) formats
        for doc_id, payload in raw_data.items():
            # Unpack payload - supports both 2-tuple and 3-tuple formats
            if len(payload) == 3:
                # New format: (text, meta, owner_id)
                text, meta, user_id = payload
                logger.debug(
                    f"Loaded doc {doc_id} with explicit owner: {user_id}")
            elif len(payload) == 2:
                # Old format: (text, meta) - extract owner from metadata
                text, meta = payload
                user_id = meta.get("owner_id") if meta else None
                logger.debug(
                    f"Loaded doc {doc_id} with metadata owner: {user_id}")
            else:
                logger.warning(
                    f"Invalid payload format for {doc_id}: "
                    f"expected 2 or 3 elements, got {len(payload)}"
                )
                continue

            # Populate doc manager
            doc_manager.add_document(doc_id, text, meta, user_id=user_id)

            # Populate token counter
            token_counter.add_document(doc_id, text, user_id=user_id)

            # Populate FAISS doc_dict if needed
            if components.faiss_wrapper and hasattr(components.faiss_wrapper, 'doc_dict'):
                components.faiss_wrapper.doc_dict[doc_id] = text
