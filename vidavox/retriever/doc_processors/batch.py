# batch/batch_processor.py
import logging
import asyncio
import threading
from typing import List, Tuple, Dict, Optional, Union
from itertools import zip_longest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from starlette.concurrency import run_in_threadpool
from vidavox.utils.gpu import clear_cuda_cache
from vidavox.utils.token_counter import TokenCounter
from vidavox.document.doc_manager import DocumentManager
from vidavox.search.persistence_search import AsyncPersistence

logger = logging.getLogger(__name__)


# ✅ Define a top-level picklable helper function
def run_index_job(args):
    doc_ids, texts, bm25_wrapper, faiss_wrapper = args
    if bm25_wrapper:
        bm25_wrapper.add_documents(list(zip(doc_ids, texts)))
    if faiss_wrapper:
        return faiss_wrapper.add_documents(list(zip(doc_ids, texts)), return_vectors=True)
    return None


class BatchProcessor:
    """Handles batch processing of documents."""

    def __init__(
        self,
        doc_manager: DocumentManager,
        token_counter: TokenCounter,
        bm25_wrapper=None,
        faiss_wrapper=None,
    ):
        self.doc_manager = doc_manager
        self.token_counter = token_counter
        self.bm25_wrapper = bm25_wrapper
        self.faiss_wrapper = faiss_wrapper
        self.batch_lock = threading.Lock()

        self.mode = (
            "hybrid" if bm25_wrapper and faiss_wrapper
            else "faiss" if faiss_wrapper
            else "bm25" if bm25_wrapper
            else "none"
        )
        logger.info(f"BatchProcessor initialized in {self.mode.upper()} mode")

    async def process_batch(
        self,
        docs: List[Union[Tuple[str, str, Dict], Tuple[str, str, Dict, Optional[str]]]],
        user_id: Optional[str] = "User A",
    ) -> None:
        """Process a batch of documents efficiently."""
        try:
            with self.batch_lock:
                normalised = [(d + (None,)) if len(d) ==
                              3 else d for d in docs]
                doc_ids, texts, meta_datas, folder_ids = zip_longest(
                    *normalised)

                # Update document manager
                self.doc_manager.add_documents(docs, user_id=user_id)

                # Update token counter
                for doc_id, text in zip(doc_ids, texts):
                    self.token_counter.add_document(
                        doc_id, text, user_id=user_id)

                # ✅ CPU-bound indexing offloaded to process pool
                try:
                    loop = asyncio.get_running_loop()
                    args = (doc_ids, texts, self.bm25_wrapper,
                            self.faiss_wrapper)
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        inserted_vectors = await loop.run_in_executor(
                            executor, run_index_job, args
                        )
                except RuntimeError:
                    # fallback for sync context
                    args = (doc_ids, texts, self.bm25_wrapper,
                            self.faiss_wrapper)
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        inserted_vectors = executor.submit(
                            run_index_job, args).result()

                clear_cuda_cache(min_freed_mb=50)
                logger.info(
                    f"Added {len(doc_ids)} documents to the engine for {user_id}")

                # Update file modification time index if applicable
                self._update_mtime_index(docs)

        except Exception as e:
            logger.error(f"Failed to process document batch: {e}")
            raise

    def _update_mtime_index(self, docs: List[Tuple[str, str, Dict]]) -> None:
        """Update file modification time index."""
        if not docs:
            return
        try:
            _, _, first_meta = docs[0]
            file_name = first_meta.get("file_name")
            mtime = first_meta.get("modification_time")
            if file_name and mtime and hasattr(self, "_file_mtime_index"):
                self._file_mtime_index[file_name] = mtime
        except Exception as e:
            logger.warning(f"Failed to update mtime index: {e}")
