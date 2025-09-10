# ---------------- retrieval_system/core/engine.py ----------------
"""Main RetrievalEngine rewritten to orchestrate the new, modular pieces."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
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

# Local, relative imports from this package
from ..processors import CSVProcessor, ExcelProcessor, StandardFileProcessor
from ..batch import BatchProcessor
from ..search import SearchManager
from ..formatters.base import BaseResultFormatter
from ..formatters.custom import CustomResultFormatter
from ..core.components import SearchResult  # re‑exported from sibling
from ..persistence.state_manager import StateManager
from ..utils.process_failure import ProcessFailure

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Facade that wires processors, batcher, search and persistence together."""

    # --- construction -----------------------------------------------------------------
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        *,
        use_async: bool = False,
        show_docs: bool = False,
        vector_store=None,  # VectorStorePsql or compatible
        persistence: Optional[AsyncPersistence] = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.use_async = use_async
        self.show_docs = show_docs

        # Managers / counters ----------------------------------------------------------
        self.doc_manager: DocumentManager = DocumentManager()
        self.token_counter: TokenCounter = TokenCounter()

        # Search back‑ends --------------------------------------------------------------
        self.bm25_wrapper = BM25_search()
        self.faiss_wrapper = FAISS_search(self.embedding_model)
        self.hybrid_search = Hybrid_search(
            self.bm25_wrapper, self.faiss_wrapper)
        self.search_mgr = SearchManager(self.hybrid_search, self.doc_manager)

        # Persistence (optional) --------------------------------------------------------
        self.vector_store = vector_store
        self.persistence = persistence or (
            AsyncPersistence(
                vector_store) if vector_store is not None else None
        )

        # Batch processor ---------------------------------------------------------------
        self.batch_processor = BatchProcessor(
            self.doc_manager,
            self.token_counter,
            self.bm25_wrapper,
            self.faiss_wrapper,
            self.persistence,
        )

        # File processors ordered by specificity ---------------------------------------
        self.file_processors = [
            CSVProcessor(),
            ExcelProcessor(),
            StandardFileProcessor(),  # catch‑all LAST!
        ]

    # ---------------------------- persistence API -----------------------------
    def _build_state_dict(self) -> Dict:
        """Gather all in-memory structures *including* folder/file metadata."""
        # Collect folder→meta from each doc’s metadata (if present)
        folders: Dict[str, Dict] = {}
        files: Dict[str, Dict] = {}
        for did, doc in self.doc_manager.documents.items():
            meta = doc.meta_data or {}
            # ① folders – keyed by folder_id
            fid = meta.get("folder_id")
            if fid and fid not in folders:
                folders[fid] = {
                    "id": fid,
                    "name": meta.get("folder_name"),
                    "parent_id": meta.get("parent_folder_id"),
                    "user_id": meta.get("owner_id"),
                }
            # ② files – keyed by file_id – fall back to did prefix
            file_id = meta.get("file_id") or did.split("_chunk", 1)[0]
            if file_id not in files:
                files[file_id] = {
                    "id": file_id,
                    "name": meta.get("file_name"),
                    "folder_id": fid,
                    "user_id": meta.get("owner_id"),
                }

        return {
            "embedding_model": self.embedding_model,
            "documents": {
                did: (doc.text, doc.meta_data)
                for did, doc in self.doc_manager.documents.items()
            },
            "token_snapshot": self.token_counter.snapshot().__dict__,
            "folders": folders,  # <─ NEW
            "files": files,      # <─ NEW
        }

    # ---------------------------------------------------------------------
    #  Index-only persistence  (documents still live in DB via AsyncPersistence)
    # ---------------------------------------------------------------------
    def save_indices(self, dir_: str | Path) -> None:
        dir_ = Path(dir_)
        dir_.mkdir(parents=True, exist_ok=True)

        # Serialize docs INCLUDING owner
        docs_blob = gzip.compress(pickle.dumps(
            {
                doc_id: (doc.text, doc.meta_data)
                for doc_id, doc in self.doc_manager.documents.items()
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        ))
        (dir_ / "docs.pkl.gz").write_bytes(docs_blob)

        # Search indices
        self.faiss_wrapper.save(dir_)
        self.bm25_wrapper.save(dir_)

        # tiny meta (good place for model version)
        (dir_ / "meta.json").write_text(
            json.dumps(
                {"embedding_model": self.embedding_model},
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Indices saved -> %s", dir_)

    def load_indices(self, dir_: str | Path) -> None:
        dir_ = Path(dir_)

        # 1) search indices
        self.faiss_wrapper.load(dir_)
        self.bm25_wrapper.load(dir_)

        # 2) documents
        docs_file = dir_ / "docs.pkl.gz"
        if docs_file.exists():
            raw = pickle.loads(gzip.decompress(docs_file.read_bytes()))
            # populate FAISS.doc_dict only if you need it
            self.faiss_wrapper.doc_dict.update(
                {doc_id: text for doc_id, (text, _meta) in raw.items()}
            )
            for doc_id, payload in raw.items():
                # Accept 2- or 3-tuple
                if len(payload) == 3:
                    text, meta, owner = payload
                else:                       # old file
                    text, meta = payload
                    owner = meta.get("owner_id")     # may be None
                self.doc_manager.add_document(
                    doc_id, text, meta, user_id=owner)
                self.token_counter.add_document(doc_id, text, user_id=owner)

        logger.info(
            "Indices loaded <- %s  (docs=%d, owners=%s)",
            dir_,
            len(self.doc_manager.documents),
            {u: len(s) for u, s in self.doc_manager.user_to_doc_ids.items()},
        )

    # Public

    def save_state(self, path: str | Path, *, fmt: str = "json") -> bool:
        """Persist engine state. *fmt* can be 'json' or 'pickle'."""
        path = Path(path).expanduser().resolve()
        data = self._build_state_dict()
        try:
            if fmt == "json":
                StateManager(path.parent).save(path.stem, data, fmt="json")
            elif fmt == "pickle":
                with path.open("wb") as fp:
                    pickle.dump(data, fp)
            else:
                raise ValueError("fmt must be 'json' or 'pickle'")
            logger.info("Engine state saved to %s (%s)", path, fmt)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save state - %s", exc)
            return False

    def load_state(self, path: str | Path, *, fmt: str | None = None, verbose: bool = False) -> bool:
        """Restore engine from a previously saved JSON / pickle snapshot."""
        path = Path(path).expanduser().resolve()
        if fmt is None:  # auto‑infer from filename
            fmt = "json" if path.suffix.lower() == ".json" else "pickle"
        try:
            if fmt == "json":
                data = StateManager(path.parent).load(
                    path.stem, fmt="json", default=None)
            else:
                with path.open("rb") as fp:
                    data = pickle.load(fp)
            if not data:
                logger.warning("No data found in %s", path)
                return False
            if verbose:
                logger.info("Loaded snapshot keys: %s", list(data.keys()))
            self._restore_from_state(data)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not load state - %s", exc)
            return False

    # --------------------------- internal restore -----------------------------

    def _restore_from_state(self, data: Dict) -> None:

        # Re-add documents & tokens
        for did, (text, meta) in data["documents"].items():
            owner = meta.get("owner_id")
            self.doc_manager.add_document(did, text, meta, owner)
            self.token_counter.add_document(did, text, user_id=owner)

        # Snapshot restore
        snap_dict = data.get("token_snapshot", {})
        if snap_dict:
            snap = TokenSnapshot(**snap_dict)
            self.token_counter.total_tokens = snap.total_tokens
            self.token_counter.doc_tokens = snap.doc_tokens
            self.token_counter.user_total_tokens = snap.user_total_tokens
            self.token_counter.user_doc_tokens = snap.user_doc_tokens

        # Rebuild indices (BM25 + FAISS) --------------------------------------
        batch = [(did, txt) for did, (txt, _) in data["documents"].items()]
        self.bm25_wrapper.clear_documents()
        self.faiss_wrapper.clear_documents()
        if batch:
            def _reindex():
                self.bm25_wrapper.add_documents(batch)
                self.faiss_wrapper.add_documents(batch)
            # asyncio.run(asyncio.to_thread(_reindex))

            try:
                loop = asyncio.get_running_loop()
                # ── We are already inside an event-loop ────────────────────────────────
                with ThreadPoolExecutor() as executor:
                    inserted_vectors = loop.run_in_executor(executor, _reindex)
                    # if you really need the blocking result, add “await” or “.result()”
            except RuntimeError:
                # ── No running loop: do it synchronously or via asyncio.run() ─────────
                asyncio.run(run_in_threadpool(_reindex))
        logger.info("Restored %d docs into indices", len(batch))

    # ----------------------------------------------------------------- private helpers
    def _select_processor(self, file_path: str):
        """Return the first processor that declares it *can* handle `file_path`."""
        for proc in self.file_processors:
            if proc.can_process(file_path):
                return proc
        raise ValueError(f"No processor available for {file_path}")

    def _allowed_ids(self, user_id: str | None) -> list[str] | None:
        """Return the list of doc_ids this user may see (or None == no filter)."""
        if user_id is None:
            return None                     # admin / background tasks
        return self.doc_manager.get_user_docs(user_id)

    # ----------------------------------------------------------------------- ingestion
    async def from_paths(
        self,
        sources: Sequence[Union[str, 'DocItem']],  # noqa: F821 – DocItem comes from caller’s lib
        *,
        config: Optional[ProcessingConfig] = None,
        chunker: Optional[Callable] = None,
        show_progress: bool = False,
        text_col: Optional[str] = None,
        metadata_cols: Optional[List[str]] = None,
        use_recursive: bool = True,
        user_id: Optional[str] = "User A",
        failures: Optional[List[ProcessFailure]] = None,
    ) -> "RetrievalEngine":
        """Ingest the given paths (or DocItems) into the engine, respecting incremental logic."""
        metadata_cols = metadata_cols or []
        existing_docs = self.doc_manager.documents
        batch_docs: List[Tuple[str, str, Dict]] = []
        BATCH_SIZE = 100

        iterator: Iterable = (
            tqdm(sources, desc="Processing files",
                 unit="file") if show_progress else sources
        )

        for item in iterator:
            # ----------------------------------------------------------------- normalise input
            if isinstance(item, str):
                path_str, doc_id, file_url, folder_id = item, None, None, None
            else:  # DocItem assumed – import lazily to avoid hard dep if caller doesn’t use them
                from vidavox.schemas.common import DocItem  # type: ignore

                assert isinstance(item, DocItem)
                path_str, doc_id, file_url, folder_id = item.path, item.doc_id, item.url, item.folder_id

                logger.info(
                    f"Processing item as DocItem: path_str='{path_str}', doc_id='{doc_id}', file_url='{file_url}, folder_id='{folder_id}'")
            # ----------------------------------------------------------------- choose processor
            processor = self._select_processor(path_str)
            try:
                kwargs: Dict = {
                    "doc_id": doc_id or Path(path_str).stem,
                    "folder_id": folder_id,
                    "existing_docs": existing_docs,
                    "file_url": file_url,
                    "config": config,
                    "chunker": chunker,
                    "use_recursive": use_recursive,
                    "text_col": text_col,
                    "metadata_cols": metadata_cols,
                }
                docs_from_file = processor.process(path_str, **kwargs)
            except Exception as e:
                logger.exception("Failed to process %s: %s", path_str, e)
                failures.append(ProcessFailure(path_str, str(e)))
                continue

            if not docs_from_file:
                continue

            batch_docs.extend(docs_from_file)
            if len(batch_docs) >= BATCH_SIZE:
                await self.batch_processor.process_batch(batch_docs, user_id=user_id)
                batch_docs = []

        # flush remainder ------------------------------------------------------------
        if batch_docs:
            await self.batch_processor.process_batch(batch_docs, user_id=user_id)

        self.last_failures = failures

        return self

    def from_directory(
        self,
        directory: str,
        *,
        recursive: bool = True,
        allowed_extensions: Optional[List[str]] = None,
        show_progress: bool = False,
        **kwargs,
    ) -> "RetrievalEngine":
        """Ingest *all* files in `directory`, optionally filtered by ext list."""
        from vidavox.document.file_processor import FileProcessor  # local lib

        files = FileProcessor().collect_files(directory, recursive, allowed_extensions)
        if not files:
            raise ValueError(f"No files found in {directory}")

        logger.info("Found %d files under %s", len(files), directory)
        return self.from_paths(files, show_progress=show_progress, **kwargs)

    # --------------------------------------------------------------------- querying API

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
        """Public convenience - delegates to SearchManager then formats."""
        if self.use_async:
            return asyncio.run(
                self._query_async(query_text, keywords, threshold,
                                  top_k, formatter, user_id, **search_kwargs)
            )
        return self._query_sync(query_text, keywords, threshold, top_k, formatter, user_id, **search_kwargs)

    # -------------------------------
    def _query_sync(
        self,
        query_text: str,
        keywords: Optional[List[str]],
        threshold: float,
        top_k: int,
        formatter: Optional[BaseResultFormatter],
        user_id: Optional[str],
        **search_kwargs,
    ) -> List[Dict]:
        results = self.search_mgr.search(
            query_text,
            keywords=keywords,
            top_k=top_k,
            threshold=threshold,
            user_id=user_id,
            **search_kwargs,
        )
        return self.search_mgr.process_search_results(results, formatter)

    async def _query_async(
        self,
        query_text: str,
        keywords: Optional[List[str]],
        threshold: float,
        top_k: int,
        formatter: Optional[BaseResultFormatter],
        user_id: Optional[str],
        **search_kwargs,
    ) -> List[Dict]:
        results = await self.search_mgr.search_async(
            query_text,
            keywords=keywords,
            top_k=top_k,
            threshold=threshold,
            user_id=user_id,
            **search_kwargs,
        )
        return self.search_mgr.process_search_results(results, formatter)

    # -----------------------------------------------------------------
    #  Single-document deletion helpers
    # -----------------------------------------------------------------
    def delete_document(
        self,
        doc_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Remove **one** document from every internal structure.

        Returns
        -------
        bool
            True if the doc was fully removed, False otherwise.
        """
        # ---------------------------------------------------------------- ownership / existence
        allowed = set(self._allowed_ids(user_id) or [])
        if doc_id not in allowed or doc_id not in self.doc_manager.documents:
            logger.warning(
                "delete_document: %s not found / not owned by %s", doc_id, user_id)
            return False

        try:
            # ---------------------------------------------------------------- indices
            self.bm25_wrapper.remove_document(doc_id)
            self.faiss_wrapper.remove_document(doc_id)
            clear_cuda_cache(min_freed_mb=50)

            # ---------------------------------------------------------------- counters & in-mem store
            self.token_counter.remove_document(doc_id)
            self.doc_manager.delete_document(doc_id, user_id)
            return True

        except Exception as exc:                   # noqa: BLE001
            logger.error("delete_document: %s - %s", doc_id, exc)
            return False

    async def delete_document_async(
        self,
        doc_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Asynchronous version of `delete_document`.
        """
        allowed = set(self._allowed_ids(user_id) or [])
        if doc_id not in allowed or doc_id not in self.doc_manager.documents:
            return False

        try:
            await self.bm25_wrapper.remove_document_async(doc_id)
            await self.faiss_wrapper.remove_document_async(doc_id)
            clear_cuda_cache(min_freed_mb=50)

            self.token_counter.remove_document(doc_id)
            self.doc_manager.delete_document(doc_id, user_id)

            if self.persistence:
                await self.persistence.flush_async()

            return True

        except Exception as exc:                   # noqa: BLE001
            logger.error("delete_document_async: %s - %s", doc_id, exc)
            return False

    # -----------------------------------------------------------------
    #  Bulk deletion
    # -----------------------------------------------------------------

    def delete_documents(
        self,
        doc_ids: List[str],
        user_id: Optional[str] = None,
    ) -> List[str]:
        """
        Remove a batch of documents from every internal structure.
        Returns the list of IDs that were actually deleted.
        """
        # Filter by ownership (multi-tenant safety) and existence
        allowed = set(self._allowed_ids(user_id) or doc_ids)
        to_delete = [
            d for d in doc_ids if d in allowed and d in self.doc_manager.documents]
        if not to_delete:
            return []

        # 1) Purge both search indices in one go
        try:
            self.bm25_wrapper.remove_documents(to_delete)
            self.faiss_wrapper.remove_documents(to_delete)
        except Exception as exc:              # noqa: BLE001
            logger.error("Batch index removal failed – %s", exc)

        clear_cuda_cache(min_freed_mb=50)

        # 2) Remove from token counter & doc manager
        removed: List[str] = []
        for d in to_delete:
            try:
                self.token_counter.remove_document(d)
                self.doc_manager.delete_document(d, user_id)
                removed.append(d)
            except Exception as exc:          # noqa: BLE001
                logger.warning("Could not fully purge %s – %s", d, exc)

        return removed

    def delete_user_documents(self, user_id: str) -> int:
        """
        Delete all documents owned by a given user_id from in-memory state and indices.
        Returns the number of documents deleted.
        """
        doc_ids = self.doc_manager.get_user_docs(user_id)
        if not doc_ids:
            return 0

        deleted_ids = self.delete_documents(doc_ids, user_id=user_id)
        logger.info("Deleted %d documents for user_id=%s",
                    len(deleted_ids), user_id)
        return len(deleted_ids)

    # Expose low‑level APIs for advanced use‑cases -------------------------------------

    def search(self, *args, **kwds):
        return self.search_mgr.search(*args, **kwds)

    async def search_async(self, *args, **kwds):
        return await self.search_mgr.search_async(*args, **kwds)

    # -----------------------------------------------------------------
    # Legacy convenience wrappers – feel free to delete once migrated
    # -----------------------------------------------------------------
    def retrieve(self, *args, **kwargs):
        """Alias for backward compatibility."""
        return self.query(*args, **kwargs)

    async def retrieve_async(self, *args, **kwargs):
        """Async alias for backward compatibility."""
        return await self._query_async(*args, **kwargs)

    def retrieve_best_chunk_per_document(
        self,
        query_text: str,
        keywords=None,
        per_doc_top_n: int = 5,
        threshold: float = 0.53,
        **search_kwargs,
    ):
        raw = self.search_mgr.search_best_chunk_per_document(
            query_text,
            keywords,
            per_doc_top_n=per_doc_top_n,
            threshold=threshold,
            **search_kwargs,
        )
        return self.search_mgr.process_search_results(raw, search_kwargs.get("result_formatter"))

     # ---------------------------- NEW public batch helpers -------------------
    def query_batch(
        self,
        queries: List[str],
        *,
        keywords: Optional[List[List[str]]] = None,
        threshold: float = 0.4,
        top_k: int = 5,
        formatter: Optional[BaseResultFormatter] = None,
        prefixes: Optional[List[str]] = None,
        include_doc_ids: Optional[List[str]] = None,
        exclude_doc_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        group_by_query: bool = False,
    ) -> List[List[Dict]]:
        if self.use_async:
            return asyncio.run(
                self.retrieve_batch_async(
                    queries,
                    keywords,
                    threshold,
                    top_k,
                    formatter,
                    prefixes,
                    include_doc_ids,
                    exclude_doc_ids,
                    user_id=user_id,
                    group_by_query=group_by_query,
                )
            )
        return self.retrieve_batch(
            queries,
            keywords,
            threshold,
            top_k,
            formatter,
            prefixes,
            include_doc_ids,
            exclude_doc_ids,
            user_id=user_id,
            group_by_query=group_by_query,
        )

    # ------------------------- internal batch orchestrators ------------------
    def retrieve_batch(
        self,
        query_text: List[str],
        keywords: Optional[List[List[str]]] = None,
        threshold: float = 0.4,
        top_k: int = 5,
        formatter: Optional[BaseResultFormatter] = None,
        prefixes: Optional[List[str]] = None,
        include_doc_ids: Optional[List[str]] = None,
        exclude_doc_ids: Optional[List[str]] = None,
        *,
        max_results_size: int = 1000,
        user_id: Optional[str] = None,
        group_by_query: bool = False,
    ) -> List[List[Dict]]:
        if user_id is not None:
            include_doc_ids = self._allowed_ids(user_id)
        capped_k = min(top_k, max_results_size)
        raw_batches = self.search_batch(
            query_text,
            keywords,
            capped_k,
            threshold,
            prefixes,
            include_doc_ids,
            exclude_doc_ids,
            user_id,
        )
        per_query = [self.search_mgr.process_search_results(
            b, formatter) for b in raw_batches]
        if group_by_query:
            return per_query
        return self._merge_batches(per_query)

    async def retrieve_batch_async(
        self,
        query_text: List[str],
        keywords: Optional[List[List[str]]] = None,
        threshold: float = 0.4,
        top_k: int = 5,
        formatter: Optional[BaseResultFormatter] = None,
        prefixes: Optional[List[str]] = None,
        include_doc_ids: Optional[List[str]] = None,
        exclude_doc_ids: Optional[List[str]] = None,
        *,
        max_results_size: int = 1000,
        user_id: Optional[str] = None,
        group_by_query: bool = False,
    ) -> List[List[Dict]]:
        if user_id is not None:
            include_doc_ids = self._allowed_ids(user_id)
        capped_k = min(top_k, max_results_size)
        raw_batches = await self.search_batch_async(
            query_text,
            keywords,
            capped_k,
            threshold,
            prefixes,
            include_doc_ids,
            exclude_doc_ids,
            user_id,
        )
        per_query = [self.search_mgr.process_search_results(
            b, formatter) for b in raw_batches]
        if group_by_query:
            return per_query
        return self._merge_batches(per_query)

    # --------------- thin wrappers around SearchManager for callers ----------
    def search_batch(
        self,
        query_text: List[str],
        keywords: Optional[List[List[str]]],
        top_k: int,
        threshold: float,
        prefixes: Optional[List[str]],
        include_doc_ids: Optional[List[str]],
        exclude_doc_ids: Optional[List[str]],
        user_id: Optional[str],
    ):
        return self.search_mgr.search_batch(
            query_text,
            keywords,
            top_k,
            threshold,
            prefixes,
            include_doc_ids,
            exclude_doc_ids,
            user_id,
        )

    async def search_batch_async(
        self,
        query_text: List[str],
        keywords: Optional[List[List[str]]],
        top_k: int,
        threshold: float,
        prefixes: Optional[List[str]],
        include_doc_ids: Optional[List[str]],
        exclude_doc_ids: Optional[List[str]],
        user_id: Optional[str],
    ):
        return await self.search_mgr.search_batch_async(
            query_text,
            keywords,
            top_k,
            threshold,
            prefixes,
            include_doc_ids,
            exclude_doc_ids,
            user_id,
        )

    # ------------------------- util ------------------------------------------
    def _merge_batches(self, batches: List[List[Dict]]):
        return self.search_mgr.merge_batches(batches)
