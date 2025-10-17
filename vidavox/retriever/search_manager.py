# search/search_manager.py
"""Search operations and result processing."""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set

from vidavox.search import SearchMode
from vidavox.retriever.base import BaseRetriever
from vidavox.retriever.formatters import SearchResult
from vidavox.retriever.formatters import BaseResultFormatter
from vidavox.retriever.formatters import CustomResultFormatter

logger = logging.getLogger(__name__)


class SearchManager:
    """Manages search operations and result processing."""

    def __init__(self,  retriever: BaseRetriever, doc_manager):
        self.retriever = retriever
        self.doc_manager = doc_manager

    def search(
        self,
        query_text: str,
        keywords: Optional[List[str]] = None,
        top_k: int = 5,
        threshold: float = 0.4,
        prefixes=None,
        include_doc_ids=None,
        exclude_doc_ids=None,
        user_id: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """Perform synchronous search."""
        if user_id:
            include_doc_ids = self._get_allowed_ids(user_id)
            logger.info(
                f"Include doc_ids: {include_doc_ids} for user {user_id}")
        # logger.error(f"[DEBUG] retriever type: {type(self.retriever)}")
        # logger.error(f"[DEBUG] retriever dir: {dir(self.retriever)[:15]}")

        try:
            return self.retriever.search(
                query_text,
                keywords=keywords,
                top_n=top_k,
                threshold=threshold,
                prefixes=prefixes,
                include_doc_ids=include_doc_ids,
                exclude_doc_ids=exclude_doc_ids
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def search_async(
        self,
        query_text: str,
        keywords: Optional[List[str]] = None,
        top_k: int = 5,
        threshold: float = 0.4,
        prefixes=None,
        include_doc_ids=None,
        exclude_doc_ids=None,
        user_id: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """Perform asynchronous search."""
        if user_id is not None:
            include_doc_ids = self._get_allowed_ids(user_id)

        try:
            return await self.retriever.async_search(
                query_text,
                keywords=keywords,
                top_n=top_k,
                threshold=threshold,
                prefixes=prefixes,
                include_doc_ids=include_doc_ids,
                exclude_doc_ids=exclude_doc_ids
            )
        except Exception as e:
            logger.error(f"Async search failed: {e}")
            return []

    def search_batch(
        self,
        queries: List[str],
        keywords_list: Optional[List[List[str]]],
        top_k: int,
        threshold: float,
        prefixes: Optional[List[str]],
        include_doc_ids: Optional[List[str]],
        exclude_doc_ids: Optional[List[str]],
        user_id: Optional[str],
    ) -> List[List[Tuple[str, float]]]:
        """Perform batch search operations."""
        return self.retriever.search_batch(
            queries=queries,
            keywords_list=keywords_list,
            top_n=top_k,
            threshold=threshold,
            prefixes=prefixes,
            include_doc_ids=include_doc_ids,
            exclude_doc_ids=exclude_doc_ids,
        )

    async def search_batch_async(
        self,
        queries: List[str],
        keywords_list: Optional[List[List[str]]],
        top_k: int,
        threshold: float,
        prefixes: Optional[List[str]],
        include_doc_ids: Optional[List[str]],
        exclude_doc_ids: Optional[List[str]],
        user_id: Optional[str],
    ) -> List[List[Tuple[str, float]]]:
        """Perform asynchronous batch search operations."""
        return await self.retriever.async_search_batch(
            queries=queries,
            keywords_list=keywords_list,
            top_n=top_k,
            threshold=threshold,
            prefixes=prefixes,
            include_doc_ids=include_doc_ids,
            exclude_doc_ids=exclude_doc_ids,
        )

    def search_best_chunk_per_document(
        self,
        query_text: str,
        keywords: Optional[List[str]],
        per_doc_top_n: int = 5,
        threshold: float = 0.53,
        prefixes=None,
        search_mode: Optional[SearchMode] = SearchMode.HYBRID,
        sort_globally: Optional[bool] = False,
        include_doc_ids=None,
        exclude_doc_ids=None,
        user_id: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """Perform advanced search and return top chunks per document."""
        if user_id is not None:
            include_doc_ids = self._get_allowed_ids(user_id)

        # Retrieve a large candidate pool
        candidate_results = self.retriever.search(
            query_text,
            keywords,
            top_n=1000,
            threshold=threshold,
            search_mode=search_mode,
            prefixes=prefixes,
            include_doc_ids=include_doc_ids,
            exclude_doc_ids=exclude_doc_ids
        )

        if not candidate_results:
            logger.info("No candidate results found.")
            return []

        # Group results by document
        grouped_results = {}
        for doc_id, score in candidate_results:
            doc_key = doc_id.split('_')[0]  # Extract fileName
            grouped_results.setdefault(doc_key, []).append((doc_id, score))

        # For each document group, sort chunks by score and take top per_doc_top_n
        final_results = []
        for doc_key, chunks in grouped_results.items():
            sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
            final_results.extend(sorted_chunks[:per_doc_top_n])

        # Optionally sort globally by score
        if sort_globally:
            final_results = sorted(
                final_results, key=lambda x: x[1], reverse=True)

        return final_results

    def process_search_results(
        self,
        results: List[Tuple[str, float]],
        result_formatter: Optional[BaseResultFormatter] = None
    ) -> List[Dict]:
        """Process search results into formatted output."""
        if not results:
            return [{"id": "None.", "url": "None.", "text": None, "score": 0.0}]

        formatter = result_formatter or CustomResultFormatter()
        seen, output = set(), []

        for doc_id, text, score in results:
            if doc_id in seen or doc_id not in self.doc_manager.documents:
                continue

            seen.add(doc_id)
            doc_obj = self.doc_manager.documents[doc_id]
            # print(f"text: {text}")
            # print(f"doc_obj.text: {doc_obj.text}")

            search_result = SearchResult(
                doc_id, doc_obj.text, doc_obj.meta_data, score)
            output.append(formatter.format(search_result))

        return output or [{"id": "None.", "url": "None.", "text": None, "score": 0.0}]

    def merge_batches(self, batches: List[List[Dict]]) -> List[Dict]:
        """Merge and deduplicate batch results, keeping best score per document."""
        best: Dict[str, Dict] = {}

        for batch in batches:  # outer list == per-query list
            for item in batch:  # inner list == individual hits
                doc_id = item.get("id")
                score = item.get("score", 0)
                if doc_id is None:
                    continue

                # Keep item only if first time seeing ID or better score
                current_best = best.get(doc_id)
                if current_best is None or score > current_best["score"]:
                    best[doc_id] = item

        # Convert dict to list and sort by score (descending)
        return sorted(best.values(), key=lambda x: x["score"], reverse=True)

    def _get_allowed_ids(self, user_id: str | None) -> list[str] | None:
        """Return the list of doc_ids this user may see (or None == no filter)."""
        if user_id is None:
            return None  # admin / background tasks
        return self.doc_manager.get_user_docs(user_id)
