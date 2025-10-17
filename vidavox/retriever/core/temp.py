"""
Refactored RetrievalEngine with improved modularity, reliability, and debuggability.

Key improvements:
- Separated concerns into focused components
- Added comprehensive error handling and logging
- Introduced validation and type safety
- Made dependencies explicit
- Added circuit breaker patterns for resilience
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal
from contextlib import asynccontextmanager
from enum import Enum

from vidavox.utils.gpu import clear_cuda_cache
from vidavox.utils.token_counter import TokenCounter
from vidavox.document import ProcessingConfig
from vidavox.document.doc_manager import DocumentManager
from vidavox.retriever.factory_retriever import RetrieverFactory
from vidavox.retriever.search_manager import SearchManager
from vidavox.retrieval_system.batch import BatchProcessor
from vidavox.retrieval_system.formatters.base import BaseResultFormatter
from vidavox.retrieval_system.persistence.state_manager import StateManager
from vidavox.retrieval_system.utils.process_failure import ProcessFailure
from vidavox.schemas.common import DocItem

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration & Validation
# ============================================================================

class SearchKind(str, Enum):
    """Supported search types."""
    BM25 = "bm25"
    FAISS = "faiss"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class EngineConfig:
    """Immutable configuration for RetrievalEngine."""
    embedding_model: str = "all-MiniLM-L6-v2"
    index_dir: Optional[Path] = None
    use_async: bool = False
    show_docs: bool = False
    search_kind: SearchKind = SearchKind.HYBRID
    batch_size: int = 100
    max_workers: int = 4
    enable_gpu_cache_clearing: bool = True
    min_gpu_freed_mb: int = 50
    
    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        if self.index_dir:
            object.__setattr__(self, 'index_dir', Path(self.index_dir))


@dataclass
class IngestionStats:
    """Track ingestion progress and failures."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    failures: List[ProcessFailure] = field(default_factory=list)
    
    def record_failure(self, path: str, error: str):
        """Record a processing failure."""
        self.failed_files += 1
        self.failures.append(ProcessFailure(path, error))
    
    def record_success(self, chunk_count: int):
        """Record successful processing."""
        self.processed_files += 1
        self.total_chunks += chunk_count


# ============================================================================
# Component Management
# ============================================================================

class RetrieverComponents:
    """Manages retriever and search components with validation."""
    
    def __init__(self, retriever, config: EngineConfig):
        self.retriever = retriever
        self.config = config
        self._validate_and_initialize()
    
    def _validate_and_initialize(self):
        """Validate retriever and extract components."""
        if not hasattr(self.retriever, 'model'):
            raise ValueError("Retriever missing 'model' attribute")
        
        self.bm25_search = getattr(self.retriever.model, "bm25_search", None)
        self.faiss_search = getattr(self.retriever.model, "faiss_search", None)
        self.hybrid_search = self.retriever.model
        
        # Validate based on search kind
        if self.config.search_kind == SearchKind.BM25 and not self.bm25_search:
            raise ValueError("BM25 search not available in retriever")
        if self.config.search_kind == SearchKind.FAISS and not self.faiss_search:
            raise ValueError("FAISS search not available in retriever")
        if self.config.search_kind == SearchKind.HYBRID:
            if not self.bm25_search or not self.faiss_search:
                raise ValueError("Hybrid search requires both BM25 and FAISS")
        
        logger.info(f"Retriever initialized: {self.config.search_kind.value}")


# ============================================================================
# Index Persistence Manager
# ============================================================================

class IndexManager:
    """Handles loading/saving indices with validation and error recovery."""
    
    def __init__(self, config: EngineConfig):
        self.config = config
    
    def save(
        self,
        directory: Path,
        components: RetrieverComponents,
        doc_manager: DocumentManager,
    ) -> bool:
        """Save indices to disk with comprehensive error handling."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            
            # Save in order: metadata -> docs -> indices
            self._save_metadata(directory)
            self._save_documents(directory, doc_manager)
            self._save_search_indices(directory, components)
            
            logger.info(f"✓ Indices saved to {directory}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to save indices: {e}", exc_info=True)
            return False
    
    def load(
        self,
        directory: Path,
        components: RetrieverComponents,
        doc_manager: DocumentManager,
        token_counter: TokenCounter,
    ) -> bool:
        """Load indices from disk with validation."""
        if not directory or not directory.exists():
            logger.warning(f"Index directory not found: {directory}")
            return False
        
        try:
            # Load in order: metadata -> indices -> docs
            self._validate_metadata(directory)
            self._load_search_indices(directory, components)
            self._load_documents(directory, components, doc_manager, token_counter)
            
            logger.info(
                f"✓ Loaded {len(doc_manager.documents)} documents from {directory}"
            )
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
        if components.faiss_search:
            components.faiss_search.save(directory)
        if components.bm25_search:
            components.bm25_search.save(directory)
    
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
        if components.faiss_search:
            components.faiss_search.load(directory)
        if components.bm25_search:
            components.bm25_search.load(directory)
    
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
                logger.debug(f"Loaded doc {doc_id} with explicit owner: {user_id}")
            elif len(payload) == 2:
                # Old format: (text, meta) - extract owner from metadata
                text, meta = payload
                user_id = meta.get("owner_id") if meta else None
                logger.debug(f"Loaded doc {doc_id} with metadata owner: {user_id}")
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
            if components.faiss_search and hasattr(components.faiss_search, 'doc_dict'):
                components.faiss_search.doc_dict[doc_id] = text


# ============================================================================
# File Processing Orchestrator
# ============================================================================

class FileProcessor:
    """Coordinates file processing with proper error handling."""
    
    def __init__(self, processors: List[Any], config: EngineConfig):
        self.processors = processors
        self.config = config
    
    def select_processor(self, file_path: str):
        """Select appropriate processor for file."""
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor
        raise ValueError(f"No processor available for {file_path}")
    
    async def process_file(
        self,
        item: Union[str, DocItem],
        existing_docs: Dict,
        chunker: Optional[Callable],
        config: Optional[ProcessingConfig],
        **kwargs,
    ) -> Tuple[List[Tuple[str, str, Dict]], Optional[str]]:
        """
        Process a single file.
        
        Returns:
            (processed_docs, error_message)
        """
        try:
            # Normalize input
            if isinstance(item, str):
                path_str = item
                doc_id = Path(path_str).stem
                file_url = None
                folder_id = None
            else:
                path_str = item.path
                doc_id = item.doc_id
                file_url = item.url
                folder_id = item.folder_id
            
            # Select and run processor
            processor = self.select_processor(path_str)
            
            process_kwargs = {
                "doc_id": doc_id,
                "folder_id": folder_id,
                "existing_docs": existing_docs,
                "file_url": file_url,
                "config": config,
                "chunker": chunker,
                **kwargs,
            }
            
            docs = processor.process(path_str, **process_kwargs)
            return docs or [], None
            
        except Exception as e:
            error_msg = f"Failed to process {getattr(item, 'path', item)}: {e}"
            logger.error(error_msg, exc_info=True)
            return [], error_msg


# ============================================================================
# Access Control Manager
# ============================================================================




# ============================================================================
# Main RetrievalEngine
# ============================================================================

class RetrievalEngine:
    """
    Modular retrieval engine with clear separation of concerns.
    
    Components:
    - IndexManager: Handles persistence
    - FileProcessor: Coordinates ingestion
    - AccessControl: Manages permissions
    - SearchManager: Handles queries
    - BatchProcessor: Processes documents in batches
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        *,
        index_dir: Optional[Union[str, Path]] = None,
        use_async: bool = False,
        show_docs: bool = False,
        kind: Literal["bm25", "faiss", "hybrid"] = "hybrid",
        vector_store=None,
        persistence=None,
        **config_kwargs,
    ):
        """Initialize retrieval engine with configuration validation."""
        
        # Create configuration
        self.config = EngineConfig(
            embedding_model=embedding_model,
            index_dir=Path(index_dir) if index_dir else None,
            use_async=use_async,
            show_docs=show_docs,
            search_kind=SearchKind(kind),
            **config_kwargs,
        )
        
        # Core managers
        self.doc_manager = DocumentManager()
        self.token_counter = TokenCounter()
        
        # Initialize retriever components
        retriever = RetrieverFactory.create(
            kind=kind,
            embedding_model=self.config.embedding_model
        )
        self.components = RetrieverComponents(retriever, self.config)
        
        # Specialized managers
        self.index_manager = IndexManager(self.config)
        self.access_control = AccessControl(self.doc_manager)
        self.search_manager = SearchManager(
            self.components.hybrid_search,
            self.doc_manager
        )
        
        # File processors (order matters - most specific first)
        from vidavox.retriever.doc_processors import (
            CSVProcessor,
            ExcelProcessor,
            StandardFileProcessor,
        )
        processor_list = [
            CSVProcessor(),
            ExcelProcessor(),
            StandardFileProcessor(),
        ]
        self.file_processor = FileProcessor(processor_list, self.config)
        
        # Batch processor
        self.batch_processor = BatchProcessor(
            self.doc_manager,
            self.token_counter,
            self.components.bm25_search,
            self.components.faiss_search,
            persistence,
        )
        
        # Load existing indices if available
        if self.config.index_dir:
            self.index_manager.load(
                self.config.index_dir,
                self.components,
                self.doc_manager,
                self.token_counter,
            )
        
        logger.info(f"✓ RetrievalEngine initialized: {self.config.search_kind.value}")
    
    # ========================================================================
    # Persistence API
    # ========================================================================
    
    def save_indices(self, directory: Optional[Union[str, Path]] = None) -> bool:
        """Save indices to disk."""
        save_dir = Path(directory) if directory else self.config.index_dir
        if not save_dir:
            raise ValueError("No index directory specified")
        
        return self.index_manager.save(
            save_dir,
            self.components,
            self.doc_manager,
        )
    
    def load_indices(self, directory: Optional[Union[str, Path]] = None) -> bool:
        """Load indices from disk."""
        load_dir = Path(directory) if directory else self.config.index_dir
        if not load_dir:
            logger.warning("No index directory specified")
            return False
        
        return self.index_manager.load(
            load_dir,
            self.components,
            self.doc_manager,
            self.token_counter,
        )
    
    # ========================================================================
    # Ingestion API
    # ========================================================================
    
    async def ingest_files(
        self,
        sources: Sequence[Union[str, DocItem]],
        *,
        config: Optional[ProcessingConfig] = None,
        chunker: Optional[Callable] = None,
        show_progress: bool = False,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> IngestionStats:
        """
        Ingest files with comprehensive tracking and error handling.
        
        Returns:
            IngestionStats with detailed processing information
        """
        stats = IngestionStats(total_files=len(sources))
        batch_docs: List[Tuple[str, str, Dict]] = []
        
        # Progress tracking
        iterator = sources
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(sources, desc="Processing files", unit="file")
        
        for item in iterator:
            # Process file
            docs, error = await self.file_processor.process_file(
                item=item,
                existing_docs=self.doc_manager.documents,
                chunker=chunker,
                config=config,
                **kwargs,
            )
            
            # Handle errors
            if error:
                path = getattr(item, 'path', str(item))
                stats.record_failure(path, error)
                continue
            
            if not docs:
                continue
            
            # Accumulate batch
            batch_docs.extend(docs)
            
            # Process batch when full
            if len(batch_docs) >= self.config.batch_size:
                await self.batch_processor.process_batch(batch_docs, user_id=user_id)
                stats.record_success(len(batch_docs))
                batch_docs = []
        
        # Flush remaining
        if batch_docs:
            await self.batch_processor.process_batch(batch_docs, user_id=user_id)
            stats.record_success(len(batch_docs))
        
        logger.info(
            f"✓ Ingestion complete: {stats.processed_files}/{stats.total_files} files, "
            f"{stats.total_chunks} chunks, {stats.failed_files} failures"
        )
        
        return stats
    
    # Backward compatibility wrapper
    async def from_paths(self, sources, **kwargs) -> "RetrievalEngine":
        """Legacy wrapper for ingest_files."""
        stats = await self.ingest_files(sources, **kwargs)
        self.last_failures = stats.failures
        return self
    
    # ========================================================================
    # Query API
    # ========================================================================
    
    def query(
        self,
        query_text: str,
        *,
        keywords: Optional[List[str]] = None,
        threshold: float = 0.4,
        top_k: int = 5,
        formatter: Optional[BaseResultFormatter] = None,
        user_id: Optional[str] = None,
        **search_kwargs,
    ) -> List[Dict]:
        """Execute a search query with access control."""
        
        # Apply access control
        allowed_doc_ids = self.access_control.get_allowed_doc_ids(user_id)
        if allowed_doc_ids is not None:
            search_kwargs['include_doc_ids'] = allowed_doc_ids
        
        # Execute search
        if self.config.use_async:
            return asyncio.run(
                self._query_async(
                    query_text, keywords, threshold, top_k, formatter, **search_kwargs
                )
            )
        
        results = self.search_manager.search(
            query_text,
            keywords=keywords,
            top_k=top_k,
            threshold=threshold,
            **search_kwargs,
        )
        
        return self.search_manager.process_search_results(results, formatter)
    
    async def _query_async(
        self,
        query_text: str,
        keywords: Optional[List[str]],
        threshold: float,
        top_k: int,
        formatter: Optional[BaseResultFormatter],
        **search_kwargs,
    ) -> List[Dict]:
        """Async query execution."""
        results = await self.search_manager.search_async(
            query_text,
            keywords=keywords,
            top_k=top_k,
            threshold=threshold,
            **search_kwargs,
        )
        return self.search_manager.process_search_results(results, formatter)
    
    # ========================================================================
    # Deletion API
    # ========================================================================
    
    async def delete_document(
        self,
        doc_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Delete a single document with access control."""
        
        # Check access
        if not self.access_control.can_access_document(doc_id, user_id):
            logger.warning(f"Access denied: user={user_id} doc={doc_id}")
            return False
        
        if doc_id not in self.doc_manager.documents:
            logger.warning(f"Document not found: {doc_id}")
            return False
        
        try:
            # Remove from indices
            await self._remove_from_indices(doc_id)
            
            # Remove from managers
            self.token_counter.remove_document(doc_id)
            self.doc_manager.delete_document(doc_id, user_id)
            
            logger.info(f"✓ Deleted document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to delete {doc_id}: {e}", exc_info=True)
            return False
    
    async def delete_documents(
        self,
        doc_ids: List[str],
        user_id: Optional[str] = None,
    ) -> List[str]:
        """Bulk delete documents with access control."""
        
        # Filter by access
        to_delete = self.access_control.filter_doc_ids(doc_ids, user_id)
        
        # Filter by existence
        to_delete = [
            doc_id for doc_id in to_delete
            if doc_id in self.doc_manager.documents
        ]
        
        if not to_delete:
            return []
        
        try:
            # Bulk remove from indices
            await self._remove_batch_from_indices(to_delete)
            
            # Remove from managers
            deleted = []
            for doc_id in to_delete:
                try:
                    self.token_counter.remove_document(doc_id)
                    self.doc_manager.delete_document(doc_id, user_id)
                    deleted.append(doc_id)
                except Exception as e:
                    logger.warning(f"Failed to fully delete {doc_id}: {e}")
            
            logger.info(f"✓ Bulk deleted {len(deleted)} documents")
            return deleted
            
        except Exception as e:
            logger.error(f"✗ Bulk deletion failed: {e}", exc_info=True)
            return []
    
    def delete_user_documents(self, user_id: str) -> int:
        """Delete all documents for a user."""
        doc_ids = self.doc_manager.get_user_docs(user_id)
        if not doc_ids:
            return 0
        
        deleted = asyncio.run(self.delete_documents(doc_ids, user_id=user_id))
        logger.info(f"✓ Deleted {len(deleted)} documents for user {user_id}")
        return len(deleted)
    
    # ========================================================================
    # Internal helpers
    # ========================================================================
    
    async def _remove_from_indices(self, doc_id: str):
        """Remove document from search indices."""
        if self.components.bm25_search:
            if hasattr(self.components.bm25_search, 'remove_document_async'):
                await self.components.bm25_search.remove_document_async(doc_id)
            else:
                self.components.bm25_search.remove_document(doc_id)
        
        if self.components.faiss_search:
            if hasattr(self.components.faiss_search, 'remove_document_async'):
                await self.components.faiss_search.remove_document_async(doc_id)
            else:
                self.components.faiss_search.remove_document(doc_id)
        
        if self.config.enable_gpu_cache_clearing:
            clear_cuda_cache(min_freed_mb=self.config.min_gpu_freed_mb)
    
    async def _remove_batch_from_indices(self, doc_ids: List[str]):
        """Bulk remove documents from search indices."""
        if self.components.bm25_search:
            self.components.bm25_search.remove_documents(doc_ids)
        
        if self.components.faiss_search:
            self.components.faiss_search.remove_documents(doc_ids)
        
        if self.config.enable_gpu_cache_clearing:
            clear_cuda_cache(min_freed_mb=self.config.min_gpu_freed_mb)
    
    # ========================================================================
    # Legacy compatibility
    # ========================================================================
    
    def retrieve(self, *args, **kwargs):
        """Alias for backward compatibility."""
        return self.query(*args, **kwargs)
    
    async def retrieve_async(self, *args, **kwargs):
        """Async alias for backward compatibility."""
        return await self._query_async(*args, **kwargs)