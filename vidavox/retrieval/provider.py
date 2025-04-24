# from vidavox.retrieval.bm25_search import BM25_search
# from vidavox.retrieval.faiss_search import FAISS_search
# from vidavox.retrieval.hybrid_search import Hybrid_search
# from vidavox.retrieval.base_search import SearchProvider
# from typing import List, Tuple

# #-------------------------------------------------------------------------
# # BM25SearchProvider with Dependency Injection
# # -------------------------------------------------------------------------
# class BM25SearchProvider(SearchProvider):
#     def __init__(self, remove_stopwords: bool = True, perform_lemmatization: bool = False):
#         """
#         Initializes BM25SearchProvider by injecting the BM25_search dependency.
#         """
#         self.backend = BM25_search(remove_stopwords=remove_stopwords, 
#                                    perform_lemmatization=perform_lemmatization)
    
#     # Synchronous methods
#     def search(self, query: str, **kwargs) -> List[Tuple[str, float]]:
#         n = kwargs.get("n", 5)
#         results = self.backend.get_top_n_docs(query, n)
#         return [(doc_id, score) for doc_id, text, score in results]

#     def add_documents(self, docs: List[Tuple[str, str]]) -> None:
#         self.backend.add_documents(docs)

#     def clear_documents(self) -> None:
#         self.backend.clear_documents()

#     # Asynchronous methods
#     async def async_search(self, query: str, **kwargs) -> List[Tuple[str, float]]:
#         n = kwargs.get("n", 5)
#         results = await self.backend.async_get_top_n_docs(query, n)
#         return [(doc_id, score) for doc_id, text, score in results]

#     async def async_add_documents(self, docs: List[Tuple[str, str]]) -> None:
#         await self.backend.async_add_documents(docs)

#     async def async_clear_documents(self) -> None:
#         await self.backend.async_clear_documents()

# class FAISSSearchProvider(SearchProvider):
#     def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
#         """
#         Initializes FAISSSearchProvider by injecting the FAISS_search dependency.
#         """
#         self.backend = FAISS_search(embedding_model)

#     # Synchronous methods
#     def search(self, query: str, **kwargs) -> List[Tuple[str, float]]:
#         n = kwargs.get("n", 5)
#         results = self.backend.search(query, k=n)
#         return [(doc_id, score) for doc_id, score in results]