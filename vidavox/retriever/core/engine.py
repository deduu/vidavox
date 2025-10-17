# ---------------- retrieval_system/core/engine.py ----------------
"""Main RetrievalEngine rewritten to orchestrate the new, modular pieces."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal
import pickle
import json
import gzip
from tqdm import tqdm


from starlette.concurrency import run_in_threadpool
from concurrent.futures import ThreadPoolExecutor

from vidavox.utils.gpu import clear_cuda_cache
from vidavox.utils.token_counter import TokenCounter, TokenSnapshot
from vidavox.document import ProcessingConfig
from vidavox.document.doc_manager import DocumentManager
from vidavox.search import BM25_search, FAISS_search, Hybrid_search, SearchMode
from vidavox.search.persistence_search import AsyncPersistence
from vidavox.schemas.common import DocItem  # type: ignore

from vidavox.retriever.factory_retriever import RetrieverFactory

from vidavox.retriever.search_manager import SearchManager
from vidavox.retriever.components.retriever import RetrieverComponents
from vidavox.retriever.managers.index_manager import IndexManager
from vidavox.retriever.managers.doc_manager import DocumentManager
from vidavox.retriever.managers.access_manager import AccessControl
from vidavox.retriever.doc_processors import FileProcessor, BatchProcessor
from vidavox.retriever.schema.data import EngineConfig, IngestionStats, SearchKind
from vidavox.retriever.doc_processors import CSVProcessor, ExcelProcessor, StandardFileProcessor


logger = logging.getLogger(__name__)


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

    # --- construction -----------------------------------------------------------------
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        *,
        index_dir: Optional[Union[str, Path]] = None,
        use_async: bool = False,
        show_docs: bool = False,
        kind: Literal["bm25", "faiss", "hybrid"] = "bm25",
        **config_kwargs,

    ):
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
        print(f"retriever: {retriever}")
        self.components = RetrieverComponents(retriever, self.config)

        # Specialized managers
        self.index_manager = IndexManager(
            config=self.config,
            components=self.components,
            doc_manager=self.doc_manager,
            token_counter=self.token_counter,
        )

        self.access_control = AccessControl(self.doc_manager)

        self.search_manager = SearchManager(
            self.components.retriever,
            self.doc_manager
        )

        processor_list = [
            CSVProcessor(),
            ExcelProcessor(),
            StandardFileProcessor(),
        ]
        self.file_processor = FileProcessor(processor_list, self.config)

        # Batch processor ---------------------------------------------------------------
        self.batch_processor = BatchProcessor(
            self.doc_manager,
            self.token_counter,
            self.components.bm25_wrapper,
            self.components.faiss_wrapper,
        )

        # Load existing indices if available
        if self.config.index_dir:
            self.index_manager.load(
                self.config.index_dir,
                self.components,
                self.doc_manager,
                self.token_counter,
            )

        logger.info(
            f"✓ RetrievalEngine initialized: {self.config.search_kind.value}")

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
        logger.info(f"Search results: {results}")

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
        if self.components.bm25_wrapper:
            if hasattr(self.components.bm25_wrapper, 'remove_document_async'):
                await self.components.bm25_wrapper.remove_document_async(doc_id)
            else:
                self.components.bm25_wrapper.remove_document(doc_id)

        if self.components.faiss_wrapper:
            if hasattr(self.components.faiss_wrapper, 'remove_document_async'):
                await self.components.faiss_wrapper.remove_document_async(doc_id)
            else:
                self.components.faiss_wrapper.remove_document(doc_id)

        if self.config.enable_gpu_cache_clearing:
            clear_cuda_cache(min_freed_mb=self.config.min_gpu_freed_mb)

    async def _remove_batch_from_indices(self, doc_ids: List[str]):
        """Bulk remove documents from search indices."""
        if self.components.bm25_wrapper:
            self.components.bm25_wrapper.remove_documents(doc_ids)

        if self.components.faiss_wrapper:
            self.components.faiss_wrapper.remove_documents(doc_ids)

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
