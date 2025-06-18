# rag_persistence.py

import asyncio
import threading
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from vidavox.document_store.store import VectorStorePsql

logger = logging.getLogger(__name__)

class AsyncPersistence:
    """
    Fire-and-forget, thread-safe persistence for VectorStorePsql.
    Works from both sync and async callers without ever hopping loops.
    """
    def __init__(self, vector_store: VectorStorePsql):
        self.vs = vector_store

        # 1) pick an event loop: either the one already running,
        #    or spin up a new one in a daemon thread
        try:
            self._loop = asyncio.get_running_loop()
            self._owns_loop = False
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            self._owns_loop = True

            def _start_loop(loop: asyncio.AbstractEventLoop):
                asyncio.set_event_loop(loop)
                loop.run_forever()

            t = threading.Thread(target=_start_loop, args=(self._loop,), daemon=True)
            t.start()

        # 2) create our queue (we'll bind it to self._loop below)
        self._queue: asyncio.Queue = asyncio.Queue()
        # force the queue to dispatch on our chosen loop
        object.__setattr__(self._queue, "_loop", self._loop)

        # 3) schedule the single worker coroutine
        if self._owns_loop:
            # from another thread: use call_soon_threadsafe
            self._loop.call_soon_threadsafe(self._loop.create_task, self._worker())
        else:
            # same thread: just create the task
            self._loop.create_task(self._worker())

    # ──────────────────────────────────────────────────────────────────────────
    # Public API: fire-and-forget “queue” methods
    # ──────────────────────────────────────────────────────────────────────────

    def queue_docs(self, docs: List[Tuple[str, str, Dict[str, Any]]]):
        """[(doc_id, text, meta), …]"""
        # thread-safe enqueue
        self._loop.call_soon_threadsafe(self._queue.put_nowait, ("docs", docs))

    def queue_vectors(self, vectors: List[Tuple[str, np.ndarray]]):
        """[(doc_id, vector), …]"""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, ("vectors", vectors))

    def queue_bm25(self, bm25_map: Dict[str, Dict[str, int]]):
        """{doc_id: {term: freq}}"""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, ("bm25", bm25_map))

    def queue_token_counts(self, counts: List[Tuple[str, int]]):
        """[(doc_id, token_count), …]"""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, ("tokens", counts))

    # ──────────────────────────────────────────────────────────────────────────
    # Internal worker: pulls jobs _serially_ to avoid concurrent DB issues
    # ──────────────────────────────────────────────────────────────────────────

    async def _worker(self):
        while True:
            job_type, payload = await self._queue.get()
            print(f"[WORKER] got job {job_type}, queue left: {self._queue.qsize()}")
            try:
                if job_type == "docs":
                    await self.vs.store_documents_batch(payload)
                elif job_type == "vectors":
                    await self.vs.store_faiss_vectors_batch(payload)
                elif job_type == "bm25":
                    for doc_id, terms in payload.items():
                        await self.vs.store_bm25_terms(doc_id, terms)
                elif job_type == "tokens":
                    await self.vs.store_token_counts_batch(payload)
                else:
                    logger.error(f"Unknown AsyncPersistence job: {job_type}")
            except Exception:
                logger.exception("AsyncPersistence: write failed")
            finally:
                self._queue.task_done()

    # ──────────────────────────────────────────────────────────────────────────
    # Optional utility: wait for everything to finish before shutdown
    # ──────────────────────────────────────────────────────────────────────────

    async def flush(self):
        """Await until queue is fully drained (all pending writes done)."""
        await self._queue.join()
