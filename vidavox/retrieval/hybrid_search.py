import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import CrossEncoder
import torch
from typing import Optional, List, Tuple, Dict
from enum import Enum

import asyncio

class SearchMode(Enum):
    HYBRID = "hybrid"
    BM25 = "bm25"
    FAISS = "faiss"

class Hybrid_search:
    def __init__(
        self,
        bm25_search,
        faiss_search,
        reranker_model_name="BAAI/bge-reranker-v2-gemma",
        initial_bm25_weight=0.5,
        search_mode: SearchMode = SearchMode.HYBRID
    ):
        self.bm25_search = bm25_search
        self.faiss_search = faiss_search
        self.bm25_weight = initial_bm25_weight
        # Uncomment and configure if using a re-ranker:
        # self.reranker = FlagReranker(reranker_model_name, use_fp16=True)
        self.search_mode = search_mode
        self.logger = logging.getLogger(__name__)

    async def advanced_search_async(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        top_n: int = 5,
        threshold: float = 0.53,
        prefixes: Optional[List[str]] = None,
        include_doc_ids: Optional[List[str]] = None,   #  ← NEW
        exclude_doc_ids: Optional[List[str]] = None    #  ← NEW
        ) -> List[Tuple[str, float]]:
        # Adjust BM25 weighting based on query length.
        await self._dynamic_weighting_async(len(query.split()))
        kw_str = " ".join(keywords) if keywords else ""

        bm25_scores, bm25_doc_ids = await self._get_bm25_results_async(kw_str, top_n, prefixes, include_doc_ids, exclude_doc_ids)
        faiss_distances, faiss_doc_ids = await self._get_faiss_results_async(query, top_n, prefixes, include_doc_ids, exclude_doc_ids)
        
        # Map document IDs to scores.
        bm25_scores_dict, faiss_scores_dict = await self._map_scores_to_doc_ids_async(
            bm25_doc_ids, bm25_scores, faiss_doc_ids, faiss_distances
        )
        # Create a unified set of document IDs.
        all_doc_ids = sorted(set(bm25_doc_ids).union(faiss_doc_ids))
        
        self.logger.info(f"all_doc_ids: {all_doc_ids}")
        self.logger.info(f"prefix: {prefixes}")


        # Optionally filter doc_ids based on prefixes.
        filtered_doc_ids = await self._filter_doc_ids_by_prefixes_async(all_doc_ids, prefixes)
        if not filtered_doc_ids:
            self.logger.info("No documents match the prefixes.")
            return []
        
        # Align and normalize scores.
        filtered_bm25_scores, filtered_faiss_scores = await self._get_filtered_scores_async(
            filtered_doc_ids, bm25_scores_dict, faiss_scores_dict
        )
        bm25_scores_normalized, faiss_scores_normalized = await self._normalize_scores_async(
            filtered_bm25_scores, filtered_faiss_scores
        )
        # Compute hybrid scores.
        hybrid_scores = await self._calculate_hybrid_scores_async(bm25_scores_normalized, faiss_scores_normalized)
        # Return top_n results that meet the threshold.
        results = await self._get_top_n_results_async(filtered_doc_ids, hybrid_scores, top_n, threshold)
        return results

    def advanced_search(
            self,
            query: str,
            keywords: Optional[List[str]] = None,
            top_n: int = 5,
            threshold: float = 0.53,
            prefixes: Optional[List[str]] = None,
            include_doc_ids: Optional[List[str]] = None,   #  ← NEW
            exclude_doc_ids: Optional[List[str]] = None,   #  ← NEW
            search_mode: Optional[SearchMode] = None,
    ) -> List[Tuple[str, float]]:
        
        if search_mode:
            self.search_mode = search_mode
            
        self._dynamic_weighting(len(query.split()))
        keywords_str = " ".join(keywords) if keywords else ""

        bm25_scores, bm25_doc_ids = self._get_bm25_results(keywords_str, top_n, prefixes, include_doc_ids, exclude_doc_ids)
        faiss_distances, faiss_doc_ids = self._get_faiss_results(query, top_n, prefixes, include_doc_ids, exclude_doc_ids)
        
        bm25_scores_dict, faiss_scores_dict = self._map_scores_to_doc_ids(
            bm25_doc_ids, bm25_scores, faiss_doc_ids, faiss_distances
        )
        all_doc_ids = sorted(set(bm25_doc_ids).union(faiss_doc_ids))
        filtered_doc_ids = self._filter_doc_ids_by_prefixes(all_doc_ids, prefixes)
        if not filtered_doc_ids:
            self.logger.info("No documents match the prefixes.")
            return []
            
        filtered_bm25_scores, filtered_faiss_scores = self._get_filtered_scores(
            filtered_doc_ids, bm25_scores_dict, faiss_scores_dict
        )
        bm25_scores_normalized, faiss_scores_normalized = self._normalize_scores(
            filtered_bm25_scores, filtered_faiss_scores
        )
        hybrid_scores = self._calculate_hybrid_scores(bm25_scores_normalized, faiss_scores_normalized)
        results = self._get_top_n_results(filtered_doc_ids, hybrid_scores, top_n, threshold)
        return results
    
    def advanced_search_batch(
    self,
    queries      : List[str],
    keywords_list: Optional[List[List[str]]] = None,   # optional per-query keywords
    top_n        : int  = 5,
    threshold    : float = 0.53,
    prefixes     : Optional[List[str]] = None,
    include_doc_ids: Optional[List[str]] = None,
    exclude_doc_ids: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        Perform hybrid search for *each* query in `queries`.
        Returns a list with one (doc_id, hybrid_score) list per query.
        """
        if not queries:
            return []

        # 1. BM25 & FAISS batch retrieval
        bm25_batch  = self._get_bm25_results_batch(
            [" ".join(kws) if kws else "" for kws in (keywords_list or [[]]*len(queries))],
            top_n, prefixes, include_doc_ids, exclude_doc_ids,
        )
        faiss_batch = self._get_faiss_results_batch(
            queries, top_n, prefixes, include_doc_ids, exclude_doc_ids,
        )

        # 2. Stitch results per query
        results_per_query: List[List[Tuple[str, float]]] = []
        for (bm_scores, bm_ids), (fs_dists, fs_ids) in zip(bm25_batch, faiss_batch):

            bm_dict, fs_dict = self._map_scores_to_doc_ids(
                bm_ids, bm_scores, fs_ids, fs_dists
            )
            all_ids      = sorted(set(bm_ids).union(fs_ids))
            filtered_ids = self._filter_doc_ids_by_prefixes(all_ids, prefixes)
            if not filtered_ids:
                results_per_query.append([])
                continue

            bm_filt, fs_filt = self._get_filtered_scores(
                filtered_ids, bm_dict, fs_dict
            )
            bm_norm, fs_norm = self._normalize_scores(bm_filt, fs_filt)
            hybrid           = self._calculate_hybrid_scores(bm_norm, fs_norm)
            results          = self._get_top_n_results(
                filtered_ids, hybrid, top_n, threshold
            )
            results_per_query.append(results)

        return results_per_query

    # ─────────────────────────────────────────────────────────────────────────────
    #  Async batch hybrid search
    # ─────────────────────────────────────────────────────────────────────────────
    async def advanced_search_batch_async(
        self,
        queries      : List[str],
        keywords_list: Optional[List[List[str]]] = None,   # optional per-query keywords
        top_n        : int  = 5,
        threshold    : float = 0.53,
        prefixes     : Optional[List[str]] = None,
        include_doc_ids: Optional[List[str]] = None,
        exclude_doc_ids: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        Fully asynchronous version of `advanced_search_batch`.
        Returns one `(doc_id, hybrid_score)` list per query.
        """
        if not queries:
            return []

        # ---------- 1. Batch retrieval (BM25 & FAISS) ----------
        kw_strings = [
            " ".join(kws) if kws else ""
            for kws in (keywords_list or [[]] * len(queries))
        ]

        bm25_batch, faiss_batch = await asyncio.gather(
            # BM25
            self._get_bm25_results_batch_async(
                kw_strings, top_n,
                prefixes, include_doc_ids, exclude_doc_ids,
            ),
            # FAISS
            self._get_faiss_results_batch_async(
                queries, top_n,
                prefixes, include_doc_ids, exclude_doc_ids,
            )
        )

        # ---------- 2. Per-query fusion ----------
        results_per_query: List[List[Tuple[str, float]]] = []
        for (bm_scores, bm_ids), (fs_dists, fs_ids) in zip(bm25_batch, faiss_batch):

            bm_dict, fs_dict = await self._map_scores_to_doc_ids_async(
                bm_ids, bm_scores, fs_ids, fs_dists
            )

            all_ids      = sorted(set(bm_ids).union(fs_ids))
            filtered_ids = await self._filter_doc_ids_by_prefixes_async(
                all_ids, prefixes
            )
            if not filtered_ids:
                results_per_query.append([])
                continue

            bm_filt, fs_filt = await self._get_filtered_scores_async(
                filtered_ids, bm_dict, fs_dict
            )
            bm_norm, fs_norm = await self._normalize_scores_async(
                bm_filt, fs_filt
            )
            hybrid           = await self._calculate_hybrid_scores_async(
                bm_norm, fs_norm
            )
            results          = await self._get_top_n_results_async(
                filtered_ids, hybrid, top_n, threshold
            )
            results_per_query.append(results)

        return results_per_query

    def _dynamic_weighting(self, query_length: int) -> None:
        self.bm25_weight = 0.7 if query_length <= 5 else 0.5
        self.logger.info(f"Dynamic BM25 weight set to: {self.bm25_weight}")

    async def _dynamic_weighting_async(self, query_length: int) -> None:
        self.bm25_weight = 0.7 if query_length <= 5 else 0.5
        self.logger.info(f"Dynamic BM25 weight set to: {self.bm25_weight}")


    
    def _get_bm25_results(self,
                      keywords: str,
                      top_n: int,
                      prefixes=None,
                      include_ids=None,
                      exclude_ids=None) -> Tuple[np.ndarray, np.ndarray]:

        # Use the engine’s native, highly-optimised filtering
        if prefixes or include_ids or exclude_ids:
            raw = self.bm25_search.get_top_n_docs(
                keywords,
                n = top_n or len(self.bm25_search.doc_ids),
                include_doc_ids  = include_ids,
                exclude_doc_ids  = exclude_ids,
                include_prefixes = prefixes,
                exclude_prefixes = None          # you can expose this later
            )
            if not raw:
                return np.empty(0), np.empty(0, dtype=str)
            ids, scores = zip(*[(d, s) for d, _, s in raw])
            return np.asarray(scores), np.asarray(ids)
        scores = np.array(self.bm25_search.get_scores(keywords))
        ids    = np.array(self.bm25_search.doc_ids)
        if scores.size and top_n:                # keep only top-n if requested
           idx = scores.argsort()[-top_n:][::-1]
           return scores[idx], ids[idx]
        return scores, ids

    async def _get_bm25_results_async(self,
                                  keywords     : str,
                                  top_n        : int,
                                  prefixes     = None,
                                  include_ids  = None,
                                  exclude_ids  = None) -> Tuple[np.ndarray, np.ndarray]:

        loop = asyncio.get_running_loop()

        if prefixes or include_ids or exclude_ids:
            raw = await loop.run_in_executor(
                None,
                self.bm25_search.get_top_n_docs,
                keywords,
                top_n or len(self.bm25_search.doc_ids),
                include_ids,
                exclude_ids,
                prefixes,
                None                       # exclude_prefixes (not exposed yet)
            )
            if not raw:
                return np.empty(0), np.empty(0, dtype=str)
            ids, scores = zip(*[(d, s) for d, _, s in raw])
            return np.asarray(scores), np.asarray(ids)

        # fast path when no filter is required
        scores = np.array(await loop.run_in_executor(None, self.bm25_search.get_scores, keywords))
        ids    = np.array(self.bm25_search.doc_ids)
        if scores.size and top_n:
            idx = scores.argsort()[-top_n:][::-1]
            return scores[idx], ids[idx]
        return scores, ids

    # ─── inside Hybrid_search ──────────────────────────────────────────
    def _get_faiss_results_batch(
        self,
        queries: List[str],
        top_n : int,
        prefixes=None, include_ids=None, exclude_ids=None
    ) -> List[Tuple[np.ndarray, List[str]]]:
        id_filter = self._build_faiss_filter(prefixes, include_ids, exclude_ids)
        k         = top_n or len(self.faiss_search.doc_dict)
        return self.faiss_search.search_batch(queries, k, id_filter)

    def _get_bm25_results_batch(
        self,
        queries : List[str],
        top_n   : int,
        prefixes=None, include_ids=None, exclude_ids=None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        # simple loop – BM25 lacks a true matrix API, but this hides it
        return [
            self._get_bm25_results(q, top_n, prefixes, include_ids, exclude_ids)
            for q in queries
        ]

    # ─────────────────────────────────────────────────────────────────────────────
    #  Async helpers – batch FAISS & BM25 retrieval
    # ─────────────────────────────────────────────────────────────────────────────
    async def _get_faiss_results_batch_async(
        self,
        queries : List[str],
        top_n   : int,
        prefixes=None,
        include_ids=None,
        exclude_ids=None
    ) -> List[Tuple[np.ndarray, List[str]]]:
        """
        Asynchronous matrix search in FAISS. Uses the engine’s async
        `search_batch` (added earlier), so the heavy work runs off-thread.
        """
        id_filter = self._build_faiss_filter(prefixes, include_ids, exclude_ids)
        k         = top_n or len(self.faiss_search.doc_dict)
        return await self.faiss_search.async_search_batch(queries, k, id_filter)


    async def _get_bm25_results_batch_async(
        self,
        queries : List[str],
        top_n   : int,
        prefixes=None,
        include_ids=None,
        exclude_ids=None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        BM25 has no native matrix API, so we spawn one task per query and
        gather them in parallel.  Each task uses the *existing*
        `_get_bm25_results_async` single-query helper.
        """
        async def one(q: str):
            return await self._get_bm25_results_async(
                q, top_n, prefixes, include_ids, exclude_ids
            )

        return await asyncio.gather(*(one(q) for q in queries))

    
    def _build_faiss_filter(self, prefixes, include_ids, exclude_ids):
        if not any([prefixes, include_ids, exclude_ids]):
            return None

        allowed = set(self.faiss_search.doc_dict.keys())
        if prefixes:
            allowed &= {d for d in allowed if any(d.startswith(p) for p in prefixes)}
        if include_ids:
            allowed &= set(include_ids)
        if exclude_ids:
            allowed -= set(exclude_ids)

        return self.faiss_search.create_include_filter(list(allowed))


    def _get_faiss_results(self,
                       query: str,
                       top_n: int,
                       prefixes=None,
                       include_ids=None,
                       exclude_ids=None) -> Tuple[np.ndarray, List[str]]:

        id_filter = self._build_faiss_filter(prefixes, include_ids, exclude_ids)
        top_k     = top_n or len(self.faiss_search.doc_dict)
        dists, ids = self.faiss_search.search(query, k=top_k, id_filter=id_filter)
        return np.asarray(dists), ids

    async def _get_faiss_results_async(self,
                                   query        : str,
                                   top_n        : int,
                                   prefixes     = None,
                                   include_ids  = None,
                                   exclude_ids  = None) -> Tuple[np.ndarray, List[str]]:

        id_filter = self._build_faiss_filter(prefixes, include_ids, exclude_ids)
        k         = top_n or len(self.faiss_search.doc_dict)
        loop      = asyncio.get_running_loop()

        distances, ids = await loop.run_in_executor(
            None,
            self.faiss_search.search,
            query,
            k,
            id_filter
        )
        return np.asarray(distances), ids


    def _map_scores_to_doc_ids(
        self,
        bm25_doc_ids: np.ndarray,
        bm25_scores: np.ndarray,
        faiss_doc_ids: List[str],
        faiss_scores: np.ndarray
    ) -> Tuple[dict, dict]:
        bm25_scores_dict = dict(zip(bm25_doc_ids, bm25_scores))
        faiss_scores_dict = dict(zip(faiss_doc_ids, faiss_scores))
        return bm25_scores_dict, faiss_scores_dict

    async def _map_scores_to_doc_ids_async(
        self,
        bm25_doc_ids: np.ndarray,
        bm25_scores: np.ndarray,
        faiss_doc_ids: List[str],
        faiss_scores: np.ndarray
    ) -> Tuple[dict, dict]:
        bm25_scores_dict = dict(zip(bm25_doc_ids, bm25_scores))
        faiss_scores_dict = dict(zip(faiss_doc_ids, faiss_scores))
        return bm25_scores_dict, faiss_scores_dict

    def _filter_doc_ids_by_prefixes(
        self,
        all_doc_ids: List[str],
        prefixes: Optional[List[str]]
    ) -> List[str]:
        if prefixes:
            return [doc_id for doc_id in all_doc_ids if any(doc_id.startswith(prefix) for prefix in prefixes)]
        return list(all_doc_ids)

    async def _filter_doc_ids_by_prefixes_async(
        self,
        all_doc_ids: List[str],
        prefixes: Optional[List[str]]
    ) -> List[str]:
        if prefixes:
            return [doc_id for doc_id in all_doc_ids if any(doc_id.startswith(prefix) for prefix in prefixes)]
        return list(all_doc_ids)

    def _get_filtered_scores(
        self,
        filtered_doc_ids: List[str],
        bm25_scores_dict: dict,
        faiss_scores_dict: dict
    ) -> Tuple[List[float], List[float]]:
        bm25_aligned_scores = []
        faiss_aligned_scores = []
        for doc_id in filtered_doc_ids:
            bm25_aligned_scores.append(bm25_scores_dict.get(doc_id, 0))
            # If not found in FAISS, assign a high distance (low similarity)
            default_faiss = max(faiss_scores_dict.values()) + 1 if faiss_scores_dict else 1
            faiss_aligned_scores.append(faiss_scores_dict.get(doc_id, default_faiss))
        # Invert FAISS distances so that higher values mean higher similarity.
        faiss_aligned_scores = [1 / score if score != 0 else 0 for score in faiss_aligned_scores]
        return bm25_aligned_scores, faiss_aligned_scores

    async def _get_filtered_scores_async(
        self,
        filtered_doc_ids: List[str],
        bm25_scores_dict: dict,
        faiss_scores_dict: dict
    ) -> Tuple[List[float], List[float]]:
        bm25_aligned_scores = []
        faiss_aligned_scores = []
        for doc_id in filtered_doc_ids:
            bm25_aligned_scores.append(bm25_scores_dict.get(doc_id, 0))
            # If not found in FAISS, assign a high distance (low similarity)
            default_faiss = max(faiss_scores_dict.values()) + 1 if faiss_scores_dict else 1
            faiss_aligned_scores.append(faiss_scores_dict.get(doc_id, default_faiss))
        # Invert FAISS distances so that higher values mean higher similarity.
        faiss_aligned_scores = [1 / score if score != 0 else 0 for score in faiss_aligned_scores]
        return bm25_aligned_scores, faiss_aligned_scores

    def _normalize_scores(
        self,
        filtered_bm25_scores: List[float],
        filtered_faiss_scores: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        scaler_bm25 = MinMaxScaler()
        bm25_scores_normalized = self._normalize_array(filtered_bm25_scores, scaler_bm25)
        scaler_faiss = MinMaxScaler()
        faiss_scores_normalized = self._normalize_array(filtered_faiss_scores, scaler_faiss)
        return bm25_scores_normalized, faiss_scores_normalized

    async def _normalize_scores_async(
        self,
        filtered_bm25_scores: List[float],
        filtered_faiss_scores: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        scaler_bm25 = MinMaxScaler()
        bm25_scores_normalized = await self._normalize_array_async(filtered_bm25_scores, scaler_bm25)
        scaler_faiss = MinMaxScaler()
        faiss_scores_normalized = await self._normalize_array_async(filtered_faiss_scores, scaler_faiss)
        return bm25_scores_normalized, faiss_scores_normalized

    def _normalize_array(self, scores: List[float], scaler: MinMaxScaler) -> np.ndarray:
        scores_array = np.array(scores)
        if np.ptp(scores_array) > 0:
            normalized_scores = scaler.fit_transform(scores_array.reshape(-1, 1)).flatten()
        else:
            normalized_scores = np.full_like(scores_array, 0.5, dtype=float)
        return normalized_scores

    async def _normalize_array_async(self, scores: List[float], scaler: MinMaxScaler) -> np.ndarray:
        scores_array = np.array(scores)
        if np.ptp(scores_array) > 0:
            # Use loop.run_in_executor for potentially CPU-intensive operations
            loop = asyncio.get_event_loop()
            normalized_scores = await loop.run_in_executor(
                None,
                lambda: scaler.fit_transform(scores_array.reshape(-1, 1)).flatten()
            )
        else:
            normalized_scores = np.full_like(scores_array, 0.5, dtype=float)
        return normalized_scores

    def _calculate_hybrid_scores(
        self,
        bm25_scores_normalized: np.ndarray,
        faiss_scores_normalized: np.ndarray
    ) -> np.ndarray:
        if self.search_mode == SearchMode.BM25:
            return bm25_scores_normalized
        elif self.search_mode == SearchMode.FAISS:
            return faiss_scores_normalized
        else:
            return self.bm25_weight * bm25_scores_normalized + (1 - self.bm25_weight) * faiss_scores_normalized

    async def _calculate_hybrid_scores_async(
        self,
        bm25_scores_normalized: np.ndarray,
        faiss_scores_normalized: np.ndarray
    ) -> np.ndarray:
        if self.search_mode == SearchMode.BM25:
            return bm25_scores_normalized
        elif self.search_mode == SearchMode.FAISS:
            return faiss_scores_normalized
        else:
            return self.bm25_weight * bm25_scores_normalized + (1 - self.bm25_weight) * faiss_scores_normalized

    def _get_top_n_results(
        self,
        filtered_doc_ids: List[str],
        hybrid_scores: List[float],
        top_n: int,
        threshold: float
    ) -> List[Tuple[str, float]]:
        hybrid_scores = np.array(hybrid_scores)
        threshold_indices = np.where(hybrid_scores >= threshold)[0]
        if len(threshold_indices) == 0:
            self.logger.info("No documents meet the threshold.")
            return []
        sorted_indices = threshold_indices[np.argsort(hybrid_scores[threshold_indices])[::-1]]
        top_indices = sorted_indices[:top_n]
        results = [(filtered_doc_ids[idx], hybrid_scores[idx]) for idx in top_indices]
        if top_n < 10:
            self.logger.info(f"Top {top_n} results: {results}")
        return results

    async def _get_top_n_results_async(
        self,
        filtered_doc_ids: List[str],
        hybrid_scores: List[float],
        top_n: int,
        threshold: float
    ) -> List[Tuple[str, float]]:
        hybrid_scores = np.array(hybrid_scores)
        threshold_indices = np.where(hybrid_scores >= threshold)[0]
        if len(threshold_indices) == 0:
            self.logger.info("No documents meet the threshold.")
            return []
        
        # For numpy operations that might be CPU-intensive
        loop = asyncio.get_event_loop()
        sorted_indices_calculation = lambda: threshold_indices[np.argsort(hybrid_scores[threshold_indices])[::-1]]
        sorted_indices = await loop.run_in_executor(None, sorted_indices_calculation)
        
        top_indices = sorted_indices[:top_n]
        results = [(filtered_doc_ids[idx], hybrid_scores[idx]) for idx in top_indices]
        if top_n < 10:
            self.logger.info(f"Top {top_n} results: {results}")
        return results

    def _rerank_results(self, query: str, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        # Example: use the BM25 search to retrieve document texts.
        document_texts = [self.bm25_search.get_document(doc_id) for doc_id, _ in results]
        doc_ids = [doc_id for doc_id, _ in results]
        rerank_inputs = [[query, doc] for doc in document_texts]
        with torch.no_grad():
            rerank_scores = self.reranker.compute_score(rerank_inputs, normalize=True)
        reranked_results = sorted(zip(doc_ids, rerank_scores), key=lambda x: x[1], reverse=True)
        return reranked_results

    async def _rerank_results_async(self, query: str, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        # Retrieve document texts asynchronously
        async def get_documents_async():
            loop = asyncio.get_event_loop()
            document_texts = []
            for doc_id, _ in results:
                doc_text = await loop.run_in_executor(None, self.bm25_search.get_document, doc_id)
                document_texts.append(doc_text)
            return document_texts
        
        document_texts = await get_documents_async()
        doc_ids = [doc_id for doc_id, _ in results]
        rerank_inputs = [[query, doc] for doc in document_texts]
        
        # Run reranker in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            rerank_scores = await loop.run_in_executor(
                None,
                lambda: self.reranker.compute_score(rerank_inputs, normalize=True)
            )
        
        reranked_results = sorted(zip(doc_ids, rerank_scores), key=lambda x: x[1], reverse=True)
        return reranked_results
    
        # ────────────────────────────────────────────────────────────────────────
    # 1) New: remove a single document from both indices
    def remove_document(self, doc_id: str) -> Dict[str, bool]:
        """
        Remove a single document (by ID) from both BM25 and FAISS indexes.
        Returns a dict { "bm25_removed": bool, "faiss_removed": bool } indicating success.
        """
        bm25_removed  = False
        faiss_removed = False

        # 1a) BM25 side
        try:
            bm25_removed = self.bm25_search.remove_document(doc_id)
        except Exception as e:
            self.logger.warning(f"BM25 failed to remove '{doc_id}': {e}")

        # 1b) FAISS side
        try:
            faiss_removed = self.faiss_search.remove_document(doc_id)
        except Exception as e:
            self.logger.warning(f"FAISS failed to remove '{doc_id}': {e}")

        return {
            "bm25_removed":  bm25_removed,
            "faiss_removed": faiss_removed
        }

    # ────────────────────────────────────────────────────────────────────────
    # 2) New: remove multiple documents in batch from both indices
    def remove_documents(self, doc_ids: List[str]) -> Dict[str, List[str]]:
        """
        Remove a list of document IDs from both BM25 and FAISS indexes in one batch.
        Returns a dict:
          {
            "bm25_removed":  [list of IDs BM25 actually deleted],
            "faiss_removed": [list of IDs FAISS actually deleted]
          }
        """
        bm25_removed_list  = []
        faiss_removed_list = []

        # 2a) BM25 side: call its batch‐delete
        try:
            bm25_removed_list = self.bm25_search.remove_documents(doc_ids)
        except Exception as e:
            self.logger.warning(f"BM25 batch removal failed: {e}")

        # 2b) FAISS side: call its batch‐delete
        try:
            faiss_removed_list = self.faiss_search.remove_documents(doc_ids)
        except Exception as e:
            self.logger.warning(f"FAISS batch removal failed: {e}")

        return {
            "bm25_removed":  bm25_removed_list,
            "faiss_removed": faiss_removed_list
        }

    # ────────────────────────────────────────────────────────────────────────
    # 3) New: async version for single document
    async def async_remove_document(self, doc_id: str) -> Dict[str, bool]:
        """
        Asynchronously remove one document from both indices.
        Returns a dict { "bm25_removed": bool, "faiss_removed": bool }.
        """
        bm25_removed  = False
        faiss_removed = False

        # BM25: offload to thread if needed
        try:
            bm25_removed = await asyncio.to_thread(self.bm25_search.remove_document, doc_id)
        except Exception as e:
            self.logger.warning(f"BM25 async failed to remove '{doc_id}': {e}")

        # FAISS: offload to thread
        try:
            faiss_removed = await asyncio.to_thread(self.faiss_search.remove_document, doc_id)
        except Exception as e:
            self.logger.warning(f"FAISS async failed to remove '{doc_id}': {e}")

        return {
            "bm25_removed":  bm25_removed,
            "faiss_removed": faiss_removed
        }

    # ────────────────────────────────────────────────────────────────────────
    # 4) New: async version for batch delete
    async def async_remove_documents(self, doc_ids: List[str]) -> Dict[str, List[str]]:
        """
        Asynchronously remove multiple documents from both indices in one shot.
        Returns dict {
            "bm25_removed":  [IDs removed from BM25],
            "faiss_removed": [IDs removed from FAISS]
        }.
        """
        bm25_removed_list  = []
        faiss_removed_list = []

        # BM25 part
        try:
            bm25_removed_list = await asyncio.to_thread(self.bm25_search.remove_documents, doc_ids)
        except Exception as e:
            self.logger.warning(f"BM25 async batch removal failed: {e}")

        # FAISS part
        try:
            faiss_removed_list = await asyncio.to_thread(self.faiss_search.remove_documents, doc_ids)
        except Exception as e:
            self.logger.warning(f"FAISS async batch removal failed: {e}")

        return {
            "bm25_removed":  bm25_removed_list,
            "faiss_removed": faiss_removed_list
        }