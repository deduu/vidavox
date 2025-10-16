import threading
import asyncio
from typing import List, Tuple, Optional, Any, Union, Dict, Set, Callable
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
import faiss
import numpy as np
import torch
import pickle
import gzip
import json

from vidavox.utils.script_tracker import log_processing_time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDFilter:
    """
    Base class for ID filtering operations in FAISS searches.
    """

    def get_selector(self) -> Optional[faiss.IDSelector]:
        """Returns a FAISS IDSelector for filtering during search"""
        raise NotImplementedError("Subclasses must implement get_selector")

    def filter_results(
        self, distances: np.ndarray, indices: np.ndarray, id_mapping: Dict[int, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Post-processes search results by filtering based on IDs"""
        raise NotImplementedError("Subclasses must implement filter_results")


class IncludeIDsFilter(IDFilter):
    """Filter that includes only documents with specific IDs"""

    def __init__(self, include_ids: Set[str], reverse_id_map: Dict[str, int]):
        """
        Initialize with IDs to include.

        Args:
            include_ids: Set of document IDs to include
            reverse_id_map: Mapping from doc_id to FAISS integer ID
        """
        self.include_ids = include_ids
        self.include_faiss_ids = set()
        for doc_id in include_ids:
            if doc_id in reverse_id_map:
                self.include_faiss_ids.add(reverse_id_map[doc_id])

    def get_selector(self) -> Optional[faiss.IDSelector]:
        if not self.include_faiss_ids:
            return None

        # Convert to list and ensure it's a numpy array with the right type
        ids_list = list(self.include_faiss_ids)
        if not ids_list:
            return None

        # Create a long array (int64)
        ids_array = np.array(ids_list, dtype="int64")

        # Pass the array directly without size (the size is determined from the array)
        return faiss.IDSelectorBatch(ids_array)

    def filter_results(
        self, distances: np.ndarray, indices: np.ndarray, id_mapping: Dict[int, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.include_ids:
            return np.array([]), np.array([])

        valid_indices = []
        for i, idx in enumerate(indices):
            if idx == -1:  # FAISS uses -1 for invalid/empty results
                continue
            doc_id = id_mapping.get(idx)
            if doc_id in self.include_ids:
                valid_indices.append(i)

        if not valid_indices:
            return np.array([]), np.array([])

        return distances[valid_indices], indices[valid_indices]


class ExcludeIDsFilter(IDFilter):
    """Filter that excludes documents with specific IDs"""

    def __init__(self, exclude_ids: Set[str], reverse_id_map: Dict[str, int]):
        """
        Initialize with IDs to exclude.

        Args:
            exclude_ids: Set of document IDs to exclude
            reverse_id_map: Mapping from doc_id to FAISS integer ID
        """
        self.exclude_ids = exclude_ids
        self.exclude_faiss_ids = set()
        for doc_id in exclude_ids:
            if doc_id in reverse_id_map:
                self.exclude_faiss_ids.add(reverse_id_map[doc_id])

    def get_selector(self) -> Optional[faiss.IDSelector]:
        if not self.exclude_faiss_ids:
            return None

        # Convert to list and ensure it's a numpy array with the right type
        ids_list = list(self.exclude_faiss_ids)
        if not ids_list:
            return None

        # Create a long array (int64)
        ids_array = np.array(ids_list, dtype="int64")

        # Create the exclude selector
        exclude_selector = faiss.IDSelectorBatch(ids_array)

        # Invert it to get "everything except these IDs"
        return faiss.IDSelectorNot(exclude_selector)

    def filter_results(
        self, distances: np.ndarray, indices: np.ndarray, id_mapping: Dict[int, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        valid_indices = []
        for i, idx in enumerate(indices):
            if idx == -1:  # FAISS uses -1 for invalid/empty results
                continue
            doc_id = id_mapping.get(idx)
            if doc_id not in self.exclude_ids:
                valid_indices.append(i)

        if not valid_indices:
            return np.array([]), np.array([])

        return distances[valid_indices], indices[valid_indices]


class PrefixFilter(IDFilter):
    """Filter that includes only documents with IDs starting with specific prefixes"""

    def __init__(
        self, prefixes: List[str], doc_ids: Set[str], reverse_id_map: Dict[str, int]
    ):
        """
        Initialize with ID prefixes to include.

        Args:
            prefixes: List of prefixes to match document IDs against
            doc_ids: Set of all document IDs to check against prefixes
            reverse_id_map: Mapping from doc_id to FAISS integer ID
        """
        self.prefixes = prefixes
        # Find all doc_ids that match any prefix
        self.matching_ids = {
            doc_id
            for doc_id in doc_ids
            if any(doc_id.startswith(prefix) for prefix in prefixes)
        }
        self.matching_faiss_ids = {
            reverse_id_map[doc_id]
            for doc_id in self.matching_ids
            if doc_id in reverse_id_map
        }

    def get_selector(self) -> Optional[faiss.IDSelector]:
        if not self.matching_faiss_ids:
            return None

        # Convert to list and ensure it's a numpy array with the right type
        ids_list = list(self.matching_faiss_ids)
        if not ids_list:
            return None

        # Create a long array (int64)
        ids_array = np.array(ids_list, dtype="int64")

        # Pass the array directly
        return faiss.IDSelectorBatch(ids_array)

    def filter_results(
        self, distances: np.ndarray, indices: np.ndarray, id_mapping: Dict[int, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        valid_indices = []
        for i, idx in enumerate(indices):
            if idx == -1:  # FAISS uses -1 for invalid/empty results
                continue
            doc_id = id_mapping.get(idx)
            if doc_id in self.matching_ids:
                valid_indices.append(i)

        if not valid_indices:
            return np.array([]), np.array([])

        return distances[valid_indices], indices[valid_indices]


class CustomFilter(IDFilter):
    """Custom filter that uses a user-provided function to filter document IDs"""

    def __init__(
        self,
        filter_fn: Callable[[str], bool],
        doc_ids: Set[str],
        reverse_id_map: Dict[str, int],
    ):
        """
        Initialize with a custom filtering function.

        Args:
            filter_fn: Function that takes a doc_id and returns True if it should be included
            doc_ids: Set of all document IDs to check
            reverse_id_map: Mapping from doc_id to FAISS integer ID
        """
        self.filter_fn = filter_fn
        # Apply the filter function to all doc_ids
        self.matching_ids = {doc_id for doc_id in doc_ids if filter_fn(doc_id)}
        self.matching_faiss_ids = {
            reverse_id_map[doc_id]
            for doc_id in self.matching_ids
            if doc_id in reverse_id_map
        }

    def get_selector(self) -> Optional[faiss.IDSelector]:
        if not self.matching_faiss_ids:
            return None

        # Convert to list and ensure it's a numpy array with the right type
        ids_list = list(self.matching_faiss_ids)
        if not ids_list:
            return None

        # Create a long array (int64)
        ids_array = np.array(ids_list, dtype="int64")

        # Pass the array directly
        return faiss.IDSelectorBatch(ids_array)

    def filter_results(
        self, distances: np.ndarray, indices: np.ndarray, id_mapping: Dict[int, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        valid_indices = []
        for i, idx in enumerate(indices):
            if idx == -1:
                continue
            doc_id = id_mapping.get(idx)
            if doc_id in self.matching_ids:
                valid_indices.append(i)

        if not valid_indices:
            return np.array([]), np.array([])

        return distances[valid_indices], indices[valid_indices]


class FAISS_search:
    """
    A wrapper class for FAISS similarity search with efficient document management.
    Uses all-MiniLM-L6-v2 as default embedding model if none is provided.
    """

    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, embedding_model: Optional[Any] = None):
        """
        Initialize FAISS_search with an embedding model.
        """
        self.doc_dict: Dict[str, str] = {}  # Maps doc_id to document text
        # Maps doc_id to an integer ID for FAISS
        self.id_map: Dict[str, int] = {}
        # Maps integer ID back to doc_id
        self.reverse_id_map: Dict[int, str] = {}
        self.next_index_id: int = 0  # Next available integer ID
        self.embedding_model = self._initialize_embedding_model(
            embedding_model)
        self.dimension = self.get_embedding_dimension()
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

        self.lock = threading.Lock()

    def save(self, path: str | Path) -> None:
        """
        Save both the FAISS index **and** the id maps.
        Creates   <path>/faiss.index      (binary)
                  <path>/ids.pkl.gz       (python dicts)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "faiss.index"))

        with gzip.open(path / "ids.pkl.gz", "wb") as fp:
            pickle.dump(
                {
                    "id_map": self.id_map,
                    "reverse_id_map": self.reverse_id_map,
                    "next_index_id": self.next_index_id,
                },
                fp,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("FAISS saved -> %s", path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not (path / "faiss.index").exists():
            logger.warning("FAISS load: %s missing – nothing restored", path)
            return

        self.index = faiss.read_index(str(path / "faiss.index"))

        ids_path = path / "ids.pkl.gz"
        if ids_path.exists():
            with gzip.open(ids_path, "rb") as fp:
                data = pickle.load(fp)
            self.id_map = data["id_map"]
            self.reverse_id_map = data["reverse_id_map"]
            self.next_index_id = data.get(
                "next_index_id", max(self.id_map.values(), default=-1) + 1
            )
        logger.info("FAISS loaded <- %s (%d vectors)", path, len(self.id_map))

    def _initialize_embedding_model(
        self, embedding_model: Optional[Union[str, SentenceTransformer]] = None
    ) -> SentenceTransformer:
        """
        Initialize a SentenceTransformer on GPU if CUDA is available, otherwise on CPU.
        """
        try:
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if embedding_model is None:
                logger.info(
                    f"Using default model '{self.DEFAULT_MODEL_NAME}' on device={device}"
                )
                model = SentenceTransformer(
                    self.DEFAULT_MODEL_NAME, device=device)
            elif isinstance(embedding_model, str):
                logger.info(
                    f"Loading model '{embedding_model}' on device={device}")
                model = SentenceTransformer(embedding_model, device=device)
            elif isinstance(embedding_model, SentenceTransformer):
                # If the user has already loaded a SentenceTransformer, move it to GPU/CPU as needed
                model = embedding_model
                try:
                    model.to(device)
                    logger.info(
                        f"Moved provided SentenceTransformer instance to device={device}"
                    )
                except Exception:
                    logger.warning(
                        "Could not move provided SentenceTransformer to the desired device; continuing as is."
                    )
            else:
                raise ValueError("Invalid embedding_model type")
            return model
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    async def async_initialize_embedding_model(
        self, embedding_model: Optional[Union[str, SentenceTransformer]] = None
    ) -> SentenceTransformer:
        # This will now invoke your updated _initialize_embedding_model, which handles CUDA/CPU.
        return await asyncio.to_thread(
            self._initialize_embedding_model, embedding_model
        )

    def get_embedding_dimension(self) -> int:
        if self.embedding_model is None:
            return 0
        else:
            # Encode a dummy text to determine embedding dimension
            return len(self.embedding_model.encode("embedding"))

    async def async_get_embedding_dimension(self) -> int:
        """
        Asynchronously get the embedding dimension.
        """
        if self.embedding_model is None:
            return 0
        else:
            # Use to_thread to run the encoding in a separate thread
            dummy_embedding = await asyncio.to_thread(
                self.embedding_model.encode, "embedding", convert_to_numpy=True
            )
            return len(dummy_embedding)

    def _search_embeddings(
        self,
        embeddings: np.ndarray,
        k: int,
        selector: Optional[faiss.IDSelector] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Low-level wrapper around `index.search`.
        Expects `embeddings` to be np.float32 with shape (n_queries, dim).
        Does **not** acquire locks or map IDs – callers handle that.
        """
        if selector is not None:
            params = faiss.SearchParameters()
            params.sel = selector
            return self.index.search(embeddings, k, params)
        # fallback: plain search
        return self.index.search(embeddings, k)

    def add_document(self, doc_id: str, doc: str) -> None:
        """
        Adds a single document to the FAISS index.
        """
        self.add_documents([(doc_id, doc)])

    async def add_document_async(self, doc_id: str, doc: str) -> None:
        """
        Asynchronously adds a single document to the FAISS index.
        """
        await self.add_documents_async([(doc_id, doc)])

    @log_processing_time
    def add_documents(
        self, docs_with_ids: List[Tuple[str, str]], return_vectors: bool = False
    ) -> None:
        """
        Efficiently adds multiple documents to the FAISS index and updates internal structures.
        """
        if not docs_with_ids:
            return

        new_texts = []
        new_ids = []
        with self.lock:
            for doc_id, doc in docs_with_ids:
                if not isinstance(doc, str) or not isinstance(doc_id, str):
                    logger.warning(
                        f"Skipping invalid document or ID: {doc_id}")
                    continue
                # If the document already exists, you might decide to update or skip.
                if doc_id in self.doc_dict:
                    logger.info(
                        f"Document ID {doc_id} already exists; skipping addition."
                    )
                    continue
                # Store the document text for O(1) lookup
                self.doc_dict[doc_id] = doc
                # Assign a new integer ID for FAISS
                current_index = self.next_index_id
                self.id_map[doc_id] = current_index
                # For reverse lookup
                self.reverse_id_map[current_index] = doc_id
                new_ids.append(current_index)
                new_texts.append(doc)
                self.next_index_id += 1

        if new_texts:
            try:
                embeddings = self.embedding_model.encode(
                    new_texts, convert_to_numpy=True
                ).astype("float32")
                if embeddings.shape[0] == 0:
                    raise ValueError("Empty embeddings generated")
                with self.lock:
                    self.index.add_with_ids(
                        embeddings, np.array(new_ids).astype("int64")
                    )
                    print(f"Expected dimension: {self.index.d}")
                    print(f"Embedding shape: {embeddings.shape}")

                    if return_vectors:
                        return [
                            (doc_id, embeddings[i])
                            for i, doc_id in enumerate(
                                [d for d, _ in docs_with_ids if d in self.id_map]
                            )
                        ]
            except Exception as e:
                raise RuntimeError(f"Failed to add documents: {str(e)}")
        return None

    async def add_documents_async(
        self, docs_with_ids: List[Tuple[str, str]], return_vectors: bool = False
    ) -> None:
        """
        Asynchronously adds multiple documents to the FAISS index and updates internal structures.
        """
        if not docs_with_ids:
            return

        new_texts = []
        new_ids = []
        with self.lock:
            for doc_id, doc in docs_with_ids:
                if not isinstance(doc, str) or not isinstance(doc_id, str):
                    logger.warning(
                        f"Skipping invalid document or ID: {doc_id}")
                    continue
                # If the document already exists, you might decide to update or skip.
                if doc_id in self.doc_dict:
                    logger.info(
                        f"Document ID {doc_id} already exists; skipping addition."
                    )
                    continue
                # Store the document text for O(1) lookup
                self.doc_dict[doc_id] = doc
                # Assign a new integer ID for FAISS
                current_index = self.next_index_id
                self.id_map[doc_id] = current_index
                # For reverse lookup
                self.reverse_id_map[current_index] = doc_id
                new_ids.append(current_index)
                new_texts.append(doc)
                self.next_index_id += 1

        if new_texts:
            try:
                # Asynchronously generate embeddings
                embeddings = await asyncio.to_thread(
                    self.embedding_model.encode, new_texts, convert_to_numpy=True
                )
                embeddings = embeddings.astype("float32")

                if embeddings.shape[0] == 0:
                    raise ValueError("Empty embeddings generated")

                # Add to index
                with self.lock:
                    await asyncio.to_thread(
                        self.index.add_with_ids,
                        embeddings,
                        np.array(new_ids).astype("int64"),
                    )
                    if return_vectors:
                        return [
                            (doc_id, embeddings[i])
                            for i, doc_id in enumerate(
                                [d for d, _ in docs_with_ids if d in self.id_map]
                            )
                        ]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to add documents asynchronously: {str(e)}")

        return None

    def remove_document(self, doc_id: str) -> bool:
        """
        O(1) dictionary lookup + in-place FAISS removal.
        Leaves 'holes' in the index – harmless for search speed, only wastes a
        tiny bit of RAM. Rebuild later during off-peak maintenance if desired.
        """
        with self.lock:  # critical section
            faiss_id: Optional[int] = self.id_map.pop(doc_id, None)
            if faiss_id is None:
                logger.warning("Document ID '%s' not found; skipping.", doc_id)
                return False

            # remove from Python-side maps
            self.doc_dict.pop(doc_id, None)
            self.reverse_id_map.pop(faiss_id, None)
            # NOTE: self.next_index_id is *not* touched – gaps are fine

        # heavy FAISS call outside the lock
        self.index.remove_ids(
            faiss.IDSelectorBatch(np.array([faiss_id], dtype="int64"))
        )
        logger.info("Removed document '%s' (faiss id %d) in-place.",
                    doc_id, faiss_id)
        return True

    async def async_remove_document(self, doc_id: str) -> bool:
        """
        Asynchronous wrapper that defers the FAISS call to a worker thread so
        the event-loop stays responsive.
        """
        with self.lock:
            faiss_id: Optional[int] = self.id_map.pop(doc_id, None)
            if faiss_id is None:
                logger.warning("Document ID '%s' not found; skipping.", doc_id)
                return False

            self.doc_dict.pop(doc_id, None)
            self.reverse_id_map.pop(faiss_id, None)

        # run the blocking FAISS operation off-thread
        await asyncio.to_thread(
            self.index.remove_ids,
            faiss.IDSelectorBatch(np.array([faiss_id], dtype="int64")),
        )
        logger.info(
            "Asynchronously removed document '%s' (faiss id %d).", doc_id, faiss_id
        )
        return True

    def remove_documents(self, doc_ids: list[str]) -> list[str]:
        """
        Fast, in-place batch delete using faiss.Index.remove_ids.
        Returns the list of successfully removed doc_ids.
        """
        removed: list[str] = []
        faiss_ids: list[int] = []

        # --- critical section: update Python maps ------------------
        with self.lock:
            for did in doc_ids:
                fid = self.id_map.pop(did, None)
                if fid is None:
                    logger.warning(
                        "Document ID '%s' not found; skipping.", did)
                    continue
                removed.append(did)
                faiss_ids.append(fid)
                self.doc_dict.pop(did, None)
                self.reverse_id_map.pop(fid, None)
            # NOTE: we do **not** touch next_index_id here – holes are OK.

        # --- heavy FAISS call outside the lock ---------------------
        if faiss_ids:
            ids_to_remove = np.array(faiss_ids, dtype="int64")
            self.index.remove_ids(faiss.IDSelectorBatch(ids_to_remove))
            logger.info("Batch removed %d docs (in-place).", len(removed))
        else:
            logger.info("No documents were removed; nothing to do.")

        return removed

    async def async_remove_documents(self, doc_ids: list[str]) -> list[str]:
        """
        Async wrapper that keeps the event-loop free.
        """
        removed: list[str] = []
        faiss_ids: list[int] = []

        # ---- critical section ------------------------------------
        with self.lock:
            for did in doc_ids:
                fid = self.id_map.pop(did, None)
                if fid is None:
                    logger.warning(
                        "Document ID '%s' not found; skipping.", did)
                    continue
                removed.append(did)
                faiss_ids.append(fid)
                self.doc_dict.pop(did, None)
                self.reverse_id_map.pop(fid, None)

        # ---- FAISS call off-thread -------------------------------
        if faiss_ids:
            ids_to_remove = np.array(faiss_ids, dtype="int64")
            await asyncio.to_thread(
                self.index.remove_ids, faiss.IDSelectorBatch(ids_to_remove)
            )
            logger.info(
                "Batch asynchronously removed %d docs (in-place).", len(
                    removed)
            )
        else:
            logger.info("No documents were removed asynchronously.")

        return removed

    def _rebuild_index(self) -> None:
        # snapshot what you need under the lock
        with self.lock:
            if not self.doc_dict:
                self.index = faiss.IndexIDMap(
                    faiss.IndexFlatL2(self.dimension))
                logger.info("FAISS index emptied (0 documents left).")
                return
            all_doc_ids = list(self.doc_dict.keys())
            all_texts = [self.doc_dict[d] for d in all_doc_ids]

        # heavy work *without* the lock
        embeddings = self.embedding_model.encode(
            all_texts, convert_to_numpy=True
        ).astype("float32")

        # back inside the lock just to mutate shared state
        with self.lock:
            self.id_map.clear()
            self.reverse_id_map.clear()
            self.next_index_id = 0

            new_ids = np.arange(len(all_doc_ids), dtype="int64")
            for did, fid in zip(all_doc_ids, new_ids):
                self.id_map[did] = fid
                self.reverse_id_map[fid] = did
            self.next_index_id = len(all_doc_ids)

            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
            if embeddings.size:
                self.index.add_with_ids(embeddings, new_ids)

        logger.info("FAISS index rebuilt (documents left: %d)",
                    len(all_doc_ids))

    # -- helper -------------------------------------------------------------

    @staticmethod
    def _build_faiss_index(
        dimension: int, embeddings: np.ndarray, ids: np.ndarray
    ) -> faiss.IndexIDMap:
        """Runs in a worker thread; no shared state touched."""
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        if embeddings.size:
            index.add_with_ids(embeddings, ids)
        return index

    # ---------- asynchronous ----------
    async def _async_rebuild_index(self) -> None:
        """
        Rebuild the FAISS index off-thread.
        Leaves writers/searchers free most of the time.
        """
        # ---------- 1. Snapshot state (fast) ----------
        with self.lock:
            if not self.doc_dict:
                self.index = faiss.IndexIDMap(
                    faiss.IndexFlatL2(self.dimension))
                self.id_map.clear()
                self.reverse_id_map.clear()
                self.next_index_id = 0
                logger.info("FAISS index emptied (0 documents left).")
                return

            all_doc_ids = list(self.doc_dict.keys())
            all_texts = [self.doc_dict[d] for d in all_doc_ids]

        # ---------- 2. Heavy embedding in worker thread ----------
        try:
            embeddings = await asyncio.to_thread(
                self.embedding_model.encode, all_texts, convert_to_numpy=True
            )
            embeddings = embeddings.astype("float32")
        except Exception as exc:
            raise RuntimeError(
                f"Embedding rebuild failed asynchronously: {exc}"
            ) from exc

        # ---------- 3. Build maps & FAISS index ----------
        new_ids = np.arange(len(all_doc_ids), dtype="int64")

        # build FAISS off-thread too (can be expensive for large arrays)
        new_index = await asyncio.to_thread(
            self._build_faiss_index,
            self.dimension,
            embeddings,
            new_ids,
        )

        # ---------- 4. Commit under lock (very quick) ----------
        with self.lock:
            # replace index and python-side maps atomically
            self.index = new_index
            self.id_map.clear()
            self.reverse_id_map.clear()
            for did, fid in zip(all_doc_ids, new_ids):
                self.id_map[did] = fid
                self.reverse_id_map[fid] = did
            self.next_index_id = len(all_doc_ids)

        logger.info(
            "FAISS index rebuilt asynchronously (documents left: %d)", len(
                all_doc_ids)
        )

    def search(
        self, query: str, k: int, id_filter: Optional[IDFilter] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Searches the FAISS index for the top-k documents matching the query.

        Args:
            query: The search query text
            k: Number of results to return
            id_filter: Optional IDFilter to apply during search

        Returns:
            A tuple of (distances, doc_ids)
        """
        with self.lock:
            if self.index.ntotal == 0:
                logger.info(
                    "FAISS index is empty. No results can be returned.")
                return np.array([]), []

        try:
            query_embedding = self.embedding_model.encode(
                [query], convert_to_numpy=True
            ).astype("float32")

            # Apply ID filter if provided during search
            if id_filter is not None:
                selector = id_filter.get_selector()
                if selector is not None:
                    params = faiss.SearchParameters()
                    params.sel = selector
                    distances, indices = self.index.search(
                        query_embedding, k, params=params
                    )
                else:
                    distances, indices = self.index.search(query_embedding, k)

                # Apply post-filtering if needed
                distances, indices = distances[0], indices[0]
                distances, indices = id_filter.filter_results(
                    distances, indices, self.reverse_id_map
                )
            else:
                distances, indices = self.index.search(query_embedding, k)
                distances, indices = distances[0], indices[0]

            # Convert FAISS integer IDs back to document IDs
            doc_ids = []
            for idx in indices:
                if idx == -1:  # FAISS uses -1 for invalid/empty results
                    continue
                doc_id = self.reverse_id_map.get(idx)
                if doc_id:
                    doc_ids.append(doc_id)

            return distances[: len(doc_ids)], doc_ids

        except Exception as e:
            raise RuntimeError(f"Failed to encode query: {str(e)}")

    async def async_search(
        self, query: str, k: int, id_filter: Optional[IDFilter] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Asynchronously searches the FAISS index for the top-k documents matching the query.

        Args:
            query: The search query text
            k: Number of results to return
            id_filter: Optional IDFilter to apply during search

        Returns:
            A tuple of (distances, doc_ids)
        """
        with self.lock:
            if self.index.ntotal == 0:
                logger.info(
                    "FAISS index is empty. No results can be returned.")
                return np.array([]), []

        try:
            # Asynchronously encode the query
            query_embedding = await asyncio.to_thread(
                self.embedding_model.encode, [query], convert_to_numpy=True
            )
            query_embedding = query_embedding.astype("float32")

            # Apply ID filter if provided during search
            if id_filter is not None:
                selector = id_filter.get_selector()
                if selector is not None:
                    params = faiss.SearchParameters()
                    params.sel = selector
                    distances, indices = await asyncio.to_thread(
                        self.index.search, query_embedding, k, params
                    )
                else:
                    distances, indices = await asyncio.to_thread(
                        self.index.search, query_embedding, k
                    )

                # Apply post-filtering if needed
                distances, indices = distances[0], indices[0]
                distances, indices = id_filter.filter_results(
                    distances, indices, self.reverse_id_map
                )
            else:
                # Asynchronously search the index
                distances, indices = await asyncio.to_thread(
                    self.index.search, query_embedding, k
                )
                distances, indices = distances[0], indices[0]

            # Convert FAISS integer IDs back to document IDs
            doc_ids = []
            for idx in indices:
                if idx == -1:  # FAISS uses -1 for invalid/empty results
                    continue
                doc_id = self.reverse_id_map.get(idx)
                if doc_id:
                    doc_ids.append(doc_id)

            return distances[: len(doc_ids)], doc_ids

        except Exception as e:
            raise RuntimeError(
                f"Failed to encode query asynchronously: {str(e)}")

    def search_batch(
        self,
        queries: List[str],
        k: int,
        id_filter: Optional[IDFilter] = None,
    ) -> List[Tuple[np.ndarray, List[str]]]:
        """
        Blocking version of batch search.
        Returns: [(distances, [doc_id, ...]), ...]  – one tuple per query.
        """
        if not queries:
            return []

        # ---------- fast read-only section ----------
        with self.lock:
            if self.index.ntotal == 0:
                return [(np.array([]), [])] * len(queries)

        # ---------- encode all queries at once ----------
        try:
            embeddings = self.embedding_model.encode(
                queries, convert_to_numpy=True
            ).astype("float32")
        except Exception as e:
            raise RuntimeError(f"Failed to encode queries: {e}") from e

        # ---------- single FAISS call ----------
        selector = id_filter.get_selector() if id_filter else None
        distances, indices = self._search_embeddings(embeddings, k, selector)

        # ---------- vectorised post-filter & ID remap ----------
        results: List[Tuple[np.ndarray, List[str]]] = []
        for row_d, row_i in zip(distances, indices):
            d, i = row_d, row_i
            if id_filter:
                d, i = id_filter.filter_results(d, i, self.reverse_id_map)

            valid = i != -1
            doc_ids = [self.reverse_id_map[ix] for ix in i[valid]]

            results.append((d[valid], doc_ids))

        return results

    def get_document(self, doc_id: str) -> str:
        """
        Retrieves a document by its document ID in O(1) time.
        """
        with self.lock:
            return self.doc_dict.get(doc_id, "")

    async def async_get_document(self, doc_id: str) -> str:
        """
        Asynchronously retrieves a document by its document ID.
        """
        # This is already O(1), but we add the async method for API consistency
        with self.lock:
            return self.doc_dict.get(doc_id, "")

    def restore_index_from_vectors(self, vectors_map: Dict[str, np.ndarray]) -> None:
        """
        Rebuild the FAISS index from precomputed vectors instead of re-encoding texts.
        'vectors_map' is a dictionary mapping doc_id to numpy array vector.
        """
        with self.lock:
            # Reset id_map and next_index_id.
            self.id_map = {}
            self.reverse_id_map = {}
            new_ids = []
            new_vectors = []
            # Use the ordering from the vectors_map.
            for doc_id, vector in vectors_map.items():
                # Store the document in the id_map with a new integer id.
                self.id_map[doc_id] = self.next_index_id
                self.reverse_id_map[self.next_index_id] = doc_id
                new_ids.append(self.next_index_id)
                new_vectors.append(vector)
                self.next_index_id += 1
            # Convert list of vectors into a single numpy array.
            if new_vectors:
                new_vectors_np = np.array(new_vectors, dtype="float32")
                # Reinitialize the FAISS index.
                self.index = faiss.IndexIDMap(
                    faiss.IndexFlatL2(self.dimension))
                self.index.add_with_ids(
                    new_vectors_np, np.array(new_ids, dtype="int64")
                )
                logger.info(
                    f"Rebuilt FAISS index from {len(new_ids)} stored vectors.")
            else:
                logger.warning(
                    "No vectors provided to restore the FAISS index.")

    async def async_restore_index_from_vectors(
        self, vectors_map: Dict[str, np.ndarray]
    ) -> None:
        """
        Asynchronously rebuild the FAISS index from precomputed vectors
        without blocking the event loop for potentially CPU-bound operations.
        """
        with self.lock:
            # Reset id_map and next_index_id.
            self.id_map = {}
            self.reverse_id_map = {}
            new_ids = []
            new_vectors = []

            for doc_id, vector in vectors_map.items():
                self.id_map[doc_id] = self.next_index_id
                self.reverse_id_map[self.next_index_id] = doc_id
                new_ids.append(self.next_index_id)
                new_vectors.append(vector)
                self.next_index_id += 1

        # Offload CPU-bound tasks outside the lock if possible.
        if new_vectors:
            # Offload the creation of the numpy array to a thread.
            new_vectors_np = await asyncio.to_thread(
                np.array, new_vectors, dtype="float32"
            )
            # Reinitialize the FAISS index in a thread, as adding vectors might be CPU intensive.
            with self.lock:
                self.index = faiss.IndexIDMap(
                    faiss.IndexFlatL2(self.dimension))
                # Offload the index addition to a thread.
                await asyncio.to_thread(
                    self.index.add_with_ids,
                    new_vectors_np,
                    np.array(new_ids, dtype="int64"),
                )
                logger.info(
                    f"Rebuilt FAISS index from {len(new_ids)} stored vectors.")
        else:
            logger.warning("No vectors provided to restore the FAISS index.")

    def clear_documents(self) -> None:
        """
        Clears all documents from the FAISS index.
        """
        with self.lock:
            self.doc_dict.clear()
            self.id_map.clear()
            self.reverse_id_map.clear()
            self.next_index_id = 0
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        logger.info("FAISS documents cleared and index reset.")

    async def async_clear_documents(self) -> None:
        """
        Asynchronously clears all documents from the FAISS index.
        """
        with self.lock:
            self.doc_dict.clear()
            self.id_map.clear()
            self.reverse_id_map.clear()
            self.next_index_id = 0
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        logger.info("FAISS documents cleared and index reset.")

    def get_vector(self, doc_id: str) -> Optional[np.ndarray]:
        """
        Encode and return the FAISS vector for a single document in memory.
        If doc_id not found, returns None.
        """
        with self.lock:
            if doc_id not in self.doc_dict:
                return None
            doc_text = self.doc_dict[doc_id]
        # lock is released, now safely encode
        try:
            vector = self.embedding_model.encode(
                [doc_text], convert_to_numpy=True
            ).astype("float32")
            return vector[0]  # single document vector
        except Exception as e:
            logger.error(f"Failed to encode doc {doc_id}: {e}")
            return None

    async def async_get_vector(self, doc_id: str) -> Optional[np.ndarray]:
        """
        Asynchronously encode and return the FAISS vector for a single document.
        """
        with self.lock:
            if doc_id not in self.doc_dict:
                return None
            doc_text = self.doc_dict[doc_id]

        # lock is released, now safely encode asynchronously
        try:
            vector = await asyncio.to_thread(
                self.embedding_model.encode, [doc_text], convert_to_numpy=True
            )
            return vector[0].astype("float32")  # single document vector
        except Exception as e:
            logger.error(f"Failed to encode doc {doc_id} asynchronously: {e}")
            return None

    def get_multiple_vectors(self, doc_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Return FAISS vectors for multiple documents at once.
        """
        texts = []
        valid_ids = []
        with self.lock:
            for d_id in doc_ids:
                doc_text = self.doc_dict.get(d_id)
                if doc_text:
                    valid_ids.append(d_id)
                    texts.append(doc_text)
        if not texts:
            return {}
        try:
            matrix = self.embedding_model.encode(texts, convert_to_numpy=True).astype(
                "float32"
            )
            # matrix is shape (len(valid_ids), dimension)
            return {d_id: matrix[i] for i, d_id in enumerate(valid_ids)}
        except Exception as e:
            logger.error(f"Failed to batch encode documents: {e}")
            return {}

    async def async_get_multiple_vectors(
        self, doc_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Asynchronously return FAISS vectors for multiple documents at once.
        """
        texts = []
        valid_ids = []
        with self.lock:
            for d_id in doc_ids:
                doc_text = self.doc_dict.get(d_id)
                if doc_text:
                    valid_ids.append(d_id)
                    texts.append(doc_text)
        if not texts:
            return {}

        try:
            # Asynchronously encode all texts
            matrix = await asyncio.to_thread(
                self.embedding_model.encode, texts, convert_to_numpy=True
            )
            matrix = matrix.astype("float32")

            # Create dictionary mapping document IDs to vectors
            return {d_id: matrix[i] for i, d_id in enumerate(valid_ids)}
        except Exception as e:
            logger.error(
                f"Failed to batch encode documents asynchronously: {e}")
            return {}

    # Additional filtering methods

    def create_include_filter(self, include_ids: List[str]) -> IDFilter:
        """
        Creates a filter that includes only documents with specific IDs.

        Args:
            include_ids: List of document IDs to include in search results

        Returns:
            An IDFilter that can be passed to search methods
        """
        return IncludeIDsFilter(set(include_ids), self.id_map)

    def create_exclude_filter(self, exclude_ids: List[str]) -> IDFilter:
        """
        Creates a filter that excludes documents with specific IDs.

        Args:
            exclude_ids: List of document IDs to exclude from search results

        Returns:
            An IDFilter that can be passed to search methods
        """
        return ExcludeIDsFilter(set(exclude_ids), self.id_map)

    def create_prefix_filter(self, prefixes: List[str]) -> IDFilter:
        """
        Creates a filter that includes only documents with IDs starting with specific prefixes.

        Args:
            prefixes: List of prefixes to match document IDs against

        Returns:
            An IDFilter that can be passed to search methods
        """
        return PrefixFilter(
            prefixes, set(self.id_map.keys()), self.id_map  # <── changed
        )

    def create_custom_filter(self, filter_fn: Callable[[str], bool]) -> IDFilter:
        """
        Creates a filter using a custom function to determine which document IDs to include.

        Args:
            filter_fn: Function that takes a doc_id and returns True if it should be included

        Returns:
            An IDFilter that can be passed to search methods
        """
        return CustomFilter(filter_fn, set(self.doc_dict.keys()), self.id_map)

    async def async_search_batch(
        self, queries: List[str], k: int, id_filter: Optional[IDFilter] = None
    ) -> List[Tuple[np.ndarray, List[str]]]:
        """
        Asynchronously search multiple queries at once with optional ID filtering.
        Returns a list of (distances, doc_ids) tuples.

        Args:
            queries: List of query strings to search
            k: Number of results to return for each query
            id_filter: Optional IDFilter to apply to all queries

        Returns:
            List of (distances, doc_ids) tuples for each query
        """
        with self.lock:
            if self.index.ntotal == 0:
                logger.info(
                    "FAISS index is empty. No results can be returned.")
                return [(np.array([]), [])] * len(queries)

        try:
            # Encode all queries at once asynchronously
            query_embeddings = await asyncio.to_thread(
                self.embedding_model.encode, queries, convert_to_numpy=True
            )
            query_embeddings = query_embeddings.astype("float32")

            # Search each query
            results = []
            for i in range(len(queries)):
                single_query = query_embeddings[i: i + 1]  # Keep as 2D array

                # Apply ID filter if provided during search
                if id_filter is not None:
                    selector = id_filter.get_selector()
                    if selector is not None:
                        params = faiss.SearchParameters()
                        params.sel = selector
                        distances, indices = await asyncio.to_thread(
                            self.index.search, single_query, k, params
                        )
                    else:
                        distances, indices = await asyncio.to_thread(
                            self.index.search, single_query, k
                        )

                    # Apply post-filtering if needed
                    distances, indices = distances[0], indices[0]
                    distances, indices = id_filter.filter_results(
                        distances, indices, self.reverse_id_map
                    )
                else:
                    distances, indices = await asyncio.to_thread(
                        self.index.search, single_query, k
                    )
                    distances, indices = distances[0], indices[0]

                # Convert FAISS integer IDs back to document IDs
                doc_ids = []
                for idx in indices:
                    if idx == -1:  # FAISS uses -1 for invalid/empty results
                        continue
                    doc_id = self.reverse_id_map.get(idx)
                    if doc_id:
                        doc_ids.append(doc_id)

                results.append((distances[: len(doc_ids)], doc_ids))

            return results
        except Exception as e:
            raise RuntimeError(
                f"Failed to batch search queries asynchronously: {str(e)}"
            )

    @staticmethod
    async def create_async(embedding_model: Optional[Any] = None) -> "FAISS_search":
        """
        Static factory method to create and initialize a FAISS_search instance asynchronously.
        """
        searcher = FAISS_search(None)  # Create with no model first
        searcher.embedding_model = await searcher.async_initialize_embedding_model(
            embedding_model
        )
        searcher.dimension = await searcher.async_get_embedding_dimension()
        searcher.index = faiss.IndexIDMap(
            faiss.IndexFlatL2(searcher.dimension))
        return searcher

    async def compact_rebuild(self) -> None:
        """
        Rebuild a dense FAISS index from current `doc_dict`.
        Designed for nightly maintenance – no hot-path callers.
        """
        # 1. snapshot shared state quickly
        with self.lock:
            if not self.doc_dict:
                logger.info("Compaction skipped - no documents.")
                return
            doc_items = list(self.doc_dict.items())
            dim = self.dimension

        # 2. heavy work off the event-loop
        all_ids, all_texts = zip(*doc_items)
        embeddings = await asyncio.to_thread(
            self.embedding_model.encode, all_texts, convert_to_numpy=True
        )
        embeddings = embeddings.astype("float32")

        # 3. build brand-new index (still off-thread)
        new_index = await asyncio.to_thread(
            self._build_index_from_embeddings, embeddings
        )

        # 4. swap under lock (tiny critical section)
        with self.lock:
            self.index = new_index
            self.id_map = {d: i for i, d in enumerate(all_ids)}
            self.reverse_id_map = {i: d for i, d in enumerate(all_ids)}
            self.next_index_id = len(all_ids)

        logger.info("Nightly compaction finished – %d vectors", len(all_ids))

    def _build_index_from_embeddings(self, embs: np.ndarray) -> faiss.IndexIDMap:
        idx = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        if embs.size:
            ids = np.arange(embs.shape[0], dtype="int64")
            idx.add_with_ids(embs, ids)
        return idx
