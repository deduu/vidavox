# batch/batch_processor.py
"""Batch processing logic for documents."""

import logging
import asyncio
import threading
from typing import List, Tuple, Dict, Optional

from starlette.concurrency import run_in_threadpool
from concurrent.futures import ThreadPoolExecutor

from vidavox.utils.gpu import clear_cuda_cache
from vidavox.utils.token_counter import TokenCounter
from vidavox.document.doc_manager import DocumentManager
from vidavox.search.persistence_search import AsyncPersistence

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing of documents."""
    
    def __init__(
        self,
        doc_manager: DocumentManager,
        token_counter: TokenCounter,
        bm25_wrapper,
        faiss_wrapper,
        persistence: Optional[AsyncPersistence] = None
    ):
        self.doc_manager = doc_manager
        self.token_counter = token_counter
        self.bm25_wrapper = bm25_wrapper
        self.faiss_wrapper = faiss_wrapper
        self.persistence = persistence
        self.batch_lock = threading.Lock()
    
    def process_batch(
        self,
        docs: List[Tuple[str, str, Dict]],
        user_id: Optional[str] = "User A"
    ) -> None:
        """Process a batch of documents efficiently."""
        try:
            with self.batch_lock:
                # Extract components for batch processing
                doc_ids, texts, meta_datas = zip(*docs)
                
                # Update document manager
                self.doc_manager.add_documents(docs, user_id=user_id)
                
                # Update token counter
                for doc_id, text in zip(doc_ids, texts):
                    self.token_counter.add_document(doc_id, text, user_id=user_id)
                
                # CPU-bound indexing off the event loop
                def _index():
                    # BM25 is pure-python but fast; FAISS is the expensive bit
                    self.bm25_wrapper.add_documents(list(zip(doc_ids, texts)))
                    inserted = self.faiss_wrapper.add_documents(
                        list(zip(doc_ids, texts)), return_vectors=True
                    )
                    return inserted
                
                # inserted_vectors = asyncio.run(
                #     run_in_threadpool(_index)  # one thread-pool hop
                # )

                # Check if we're in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    if loop:
                        # Run in thread pool without asyncio.run()
                        with ThreadPoolExecutor() as executor:
                            inserted_vectors = executor.submit(_index).result()
                  
                except RuntimeError:
                    # No running event loop, safe to use asyncio.run()
                    inserted_vectors = asyncio.run(run_in_threadpool(_index))
                
                clear_cuda_cache(min_freed_mb=50)
                
                logger.info(f"Added {len(doc_ids)} documents to the engine for {user_id}")
                
                # Handle persistence if available
                if self.persistence:
                    self._handle_persistence(docs, doc_ids, inserted_vectors)
                
                # Update file modification time index if applicable
                self._update_mtime_index(docs)
        
        except Exception as e:
            logger.error(f"Failed to process document batch: {e}")
            raise
    
    def _handle_persistence(
        self,
        docs: List[Tuple[str, str, Dict]],
        doc_ids: List[str],
        inserted_vectors
    ) -> None:
        """Handle persistence operations for the batch."""
        try:
            self.persistence.queue_docs(docs)
            logger.info(f"Persisted {len(doc_ids)} documents to the persistence store.")
            
            if inserted_vectors:  # never None when return_vectors=True
                self.persistence.queue_vectors(inserted_vectors)
                logger.info(f"Persisted {len(inserted_vectors)} FAISS vectors to the persistence store.")
            
            self.persistence.queue_bm25(self.bm25_wrapper.get_multiple_doc_terms(doc_ids))
            self.persistence.queue_token_counts([
                (d, self.token_counter.get_doc_tokens(d)) for d in doc_ids
            ])
            
        except Exception as e:
            logger.error(f"Failed to handle persistence: {e}")
    
    def _update_mtime_index(self, docs: List[Tuple[str, str, Dict]]) -> None:
        """Update file modification time index."""
        if not docs:
            return
            
        try:
            _, _, first_meta = docs[0]
            file_name = first_meta.get("file_name")
            mtime = first_meta.get("modification_time")
            
            if file_name and mtime and hasattr(self, '_file_mtime_index'):
                self._file_mtime_index[file_name] = mtime
                
        except Exception as e:
            logger.warning(f"Failed to update mtime index: {e}")