import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import CrossEncoder
import torch
from typing import Optional
# from FlagEmbedding import FlagReranker


class Hybrid_search:
    def __init__(self, bm25_search, faiss_search, reranker_model_name="BAAI/bge-reranker-v2-gemma", initial_bm25_weight=0.5):
        self.bm25_search = bm25_search
        self.faiss_search = faiss_search
        self.bm25_weight = initial_bm25_weight
        # self.reranker = FlagReranker(reranker_model_name, use_fp16=True)
        self.logger = logging.getLogger(__name__)

    async def advanced_search_async(self, query, keywords, top_n=5, threshold=0.53, prefixes=None):
        # Dynamic BM25 weighting
        self._dynamic_weighting(len(query.split()))
        keywords = f"{' '.join(keywords)}" if keywords else ""
        # self.logger.info(f"Query: {query}")
        # self.logger.info(f"Keywords: {keywords}")

        # Get BM25 scores and doc_ids
        bm25_scores, bm25_doc_ids = self._get_bm25_results(keywords, top_n = top_n)
        # self.logger.info(f"BM25 Scores: {bm25_scores}, BM25 doc_ids: {bm25_doc_ids}")
        # Get FAISS distances, indices, and doc_ids
        faiss_distances, faiss_indices, faiss_doc_ids = self._get_faiss_results(query)
        try:
            faiss_distances, indices, faiss_doc_ids = self._get_faiss_results(query, top_n = top_n)
            
            # for dist, idx, doc_id in zip(faiss_distances, indices, faiss_doc_ids):
            #     print(f"Distance: {dist:.4f}, Index: {idx}, Doc ID: {doc_id}")
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
        # Map doc_ids to scores
        bm25_scores_dict, faiss_scores_dict = self._map_scores_to_doc_ids(
            bm25_doc_ids, bm25_scores, faiss_doc_ids, faiss_distances
        )
        # Create a unified set of doc IDs
        all_doc_ids = sorted(set(bm25_doc_ids).union(faiss_doc_ids))
        # print(f"All doc_ids: {all_doc_ids}, BM25 doc_ids: {bm25_doc_ids}, FAISS doc_ids: {faiss_doc_ids}")

        # Filter doc_ids based on prefixes
        filtered_doc_ids = self._filter_doc_ids_by_prefixes(all_doc_ids, prefixes)
        # self.logger.info(f"Filtered doc_ids: {filtered_doc_ids}")

        if not filtered_doc_ids:
            self.logger.info("No documents match the prefixes.")
            return []

        # Prepare score lists
        filtered_bm25_scores, filtered_faiss_scores = self._get_filtered_scores(
            filtered_doc_ids, bm25_scores_dict, faiss_scores_dict
        )

        
        # self.logger.info(f"Filtered BM25 scores: {filtered_bm25_scores}")
        # self.logger.info(f"Filtered FAISS scores: {filtered_faiss_scores}")
       

        # Normalize scores
        bm25_scores_normalized, faiss_scores_normalized = self._normalize_scores(
            filtered_bm25_scores, filtered_faiss_scores
        )

        # Calculate hybrid scores
        hybrid_scores = self._calculate_hybrid_scores(bm25_scores_normalized, faiss_scores_normalized)

        # Display hybrid scores
        # for idx, doc_id in enumerate(filtered_doc_ids):
        #     print(f"Hybrid Score: {hybrid_scores[idx]:.4f}, Doc ID: {doc_id}")

        # Apply threshold and get top_n results
        results = self._get_top_n_results(filtered_doc_ids, hybrid_scores, top_n, threshold)
        # self.logger.info(f"Results before reranking: {results}")

        # If results exist, apply re-ranking
        # if results:
        #     re_ranked_results = self._rerank_results(query, results)
        #     self.logger.info(f"Results after reranking: {re_ranked_results}")
        #     return re_ranked_results

        return results
    def advanced_search(self, query, keywords, top_n:Optional[int] = 5, threshold:Optional[float] = 0.53, prefixes:Optional[list] = None):
        # Dynamic BM25 weighting
        self._dynamic_weighting(len(query.split()))
        keywords = f"{' '.join(keywords)}" if keywords else ""
        # self.logger.info(f"Query: {query}")
        # self.logger.info(f"Keywords: {keywords}")

        # Get BM25 scores and doc_ids
        bm25_scores, bm25_doc_ids = self._get_bm25_results(keywords, top_n = top_n)
        # self.logger.info(f"BM25 Scores: {bm25_scores}, BM25 doc_ids: {bm25_doc_ids}")
        # Get FAISS distances, indices, and doc_ids
        faiss_distances, faiss_indices, faiss_doc_ids = self._get_faiss_results(query, top_n=top_n)
        # try:
        #     faiss_distances, indices, faiss_doc_ids = self._get_faiss_results(query, top_n = top_n)
            
        #     for dist, idx, doc_id in zip(faiss_distances, indices, faiss_doc_ids):
        #         print(f"Distance: {dist:.4f}, Index: {idx}, Doc ID: {doc_id}")
        # except Exception as e:
        #     self.logger.error(f"Search failed: {str(e)}")
        # Map doc_ids to scores
        bm25_scores_dict, faiss_scores_dict = self._map_scores_to_doc_ids(
            bm25_doc_ids, bm25_scores, faiss_doc_ids, faiss_distances
        )
        # Create a unified set of doc IDs
        all_doc_ids = sorted(set(bm25_doc_ids).union(faiss_doc_ids))
        # print(f"All doc_ids: {all_doc_ids}, BM25 doc_ids: {bm25_doc_ids}, FAISS doc_ids: {faiss_doc_ids}")

        # Filter doc_ids based on prefixes
        filtered_doc_ids = self._filter_doc_ids_by_prefixes(all_doc_ids, prefixes)
        # self.logger.info(f"Filtered doc_ids: {filtered_doc_ids}")

        if not filtered_doc_ids:
            self.logger.info("No documents match the prefixes.")
            return []

        # Prepare score lists
        filtered_bm25_scores, filtered_faiss_scores = self._get_filtered_scores(
            filtered_doc_ids, bm25_scores_dict, faiss_scores_dict
        )

        
        # self.logger.info(f"Filtered BM25 scores: {filtered_bm25_scores}")
        # self.logger.info(f"Filtered FAISS scores: {filtered_faiss_scores}")
       

        # Normalize scores
        bm25_scores_normalized, faiss_scores_normalized = self._normalize_scores(
            filtered_bm25_scores, filtered_faiss_scores
        )

        # Calculate hybrid scores
        hybrid_scores = self._calculate_hybrid_scores(bm25_scores_normalized, faiss_scores_normalized)

        # Display hybrid scores
        # for idx, doc_id in enumerate(filtered_doc_ids):
        #     print(f"Hybrid Score: {hybrid_scores[idx]:.4f}, Doc ID: {doc_id}")

        # Apply threshold and get top_n results
        results = self._get_top_n_results(filtered_doc_ids, hybrid_scores, top_n, threshold)
        # self.logger.info(f"Results before reranking: {results}")

        # If results exist, apply re-ranking
        # if results:
        #     re_ranked_results = self._rerank_results(query, results)
        #     self.logger.info(f"Results after reranking: {re_ranked_results}")
        #     return re_ranked_results

        return results
    


    def _dynamic_weighting(self, query_length):
        if query_length <= 5:
            self.bm25_weight = 0.7
        else:
            self.bm25_weight = 0.5
        self.logger.info(f"Dynamic BM25 weight set to: {self.bm25_weight}")

    def _get_bm25_results(self, keywords, top_n:int = None):
        # Get BM25 scores
        bm25_scores = np.array(self.bm25_search.get_scores(keywords))
        bm25_doc_ids = np.array(self.bm25_search.doc_ids)  # Assuming doc_ids is a list of document IDs

        # Log the scores and IDs before filtering
        # self.logger.info(f"BM25 scores: {bm25_scores}")
        # self.logger.info(f"BM25 doc_ids: {bm25_doc_ids}")

        # Get the top k indices based on BM25 scores
        top_k_indices = np.argsort(bm25_scores)[-top_n:][::-1]

        # Retrieve top k scores and corresponding document IDs
        top_k_scores = bm25_scores[top_k_indices]
        top_k_doc_ids = bm25_doc_ids[top_k_indices]

        # Return top k scores and document IDs
        return top_k_scores, top_k_doc_ids

    def _get_faiss_results(self, query, top_n: int = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    
        try:
            # If top_k is not specified, use all documents
            if top_n is None:
                top_n = len(self.faiss_search.doc_ids)
                
            # Use the search's search method which handles the embedding
            distances, indices = self.faiss_search.search(query, k=top_n) 
            
            if len(distances) == 0 or len(indices) == 0:
                # Handle case where FAISS returns empty results
                self.logger.info("FAISS search returned no results.")
                return np.array([]), np.array([]), []
            
            # Filter out invalid indices (-1)
            valid_mask = indices != -1
            filtered_distances = distances[valid_mask]
            filtered_indices = indices[valid_mask]
            print(f"FAISS returned indices: {filtered_indices}")

            # Map indices to doc_ids
            doc_ids = [self.faiss_search.doc_ids[idx] for idx in filtered_indices 
                    if 0 <= idx < len(self.faiss_search.doc_ids)]
            
            # self.logger.info(f"FAISS distances: {filtered_distances}")
            # self.logger.info(f"FAISS indices: {filtered_indices}")
            # self.logger.info(f"FAISS doc_ids: {doc_ids}")
            
            return filtered_distances, filtered_indices, doc_ids
            
        except Exception as e:
            self.logger.error(f"Error in FAISS search: {str(e)}")
            raise

    def _map_scores_to_doc_ids(self, bm25_doc_ids, bm25_scores, faiss_doc_ids, faiss_scores):
        bm25_scores_dict = dict(zip(bm25_doc_ids, bm25_scores))
        faiss_scores_dict = dict(zip(faiss_doc_ids, faiss_scores))
        # self.logger.info(f"BM25 scores dict: {bm25_scores_dict}")
        # self.logger.info(f"FAISS scores dict: {faiss_scores_dict}")
        return bm25_scores_dict, faiss_scores_dict

    def _filter_doc_ids_by_prefixes(self, all_doc_ids, prefixes):
        if prefixes:
            filtered_doc_ids = [
                doc_id
                for doc_id in all_doc_ids
                if any(doc_id.startswith(prefix) for prefix in prefixes)
            ]
        else:
            filtered_doc_ids = list(all_doc_ids)
        return filtered_doc_ids

    def _get_filtered_scores(self, filtered_doc_ids, bm25_scores_dict, faiss_scores_dict):
         # Initialize lists to hold scores in the unified doc ID order
        bm25_aligned_scores = []
        faiss_aligned_scores = []

        # Populate aligned score lists, filling missing scores with neutral values
        for doc_id in filtered_doc_ids:
            bm25_aligned_scores.append(bm25_scores_dict.get(doc_id, 0))  # Use 0 if not found in BM25
            faiss_aligned_scores.append(faiss_scores_dict.get(doc_id, max(faiss_scores_dict.values()) + 1))  # Use a high distance if not found in FAISS
        
        # Invert the FAISS scores
        faiss_aligned_scores = [1 / score if score != 0 else 0 for score in faiss_aligned_scores]

        return bm25_aligned_scores, faiss_aligned_scores

    def _normalize_scores(self, filtered_bm25_scores, filtered_faiss_scores):
        scaler_bm25 = MinMaxScaler()
        bm25_scores_normalized = self._normalize_array(filtered_bm25_scores, scaler_bm25)

        scaler_faiss = MinMaxScaler()
        faiss_scores_normalized = self._normalize_array(filtered_faiss_scores, scaler_faiss)

        # self.logger.info(f"Normalized BM25 scores: {bm25_scores_normalized}")
        # self.logger.info(f"Normalized FAISS scores: {faiss_scores_normalized}")
        return bm25_scores_normalized, faiss_scores_normalized

    def _normalize_array(self, scores, scaler):
        scores_array = np.array(scores)
        if np.ptp(scores_array) > 0:
            normalized_scores = scaler.fit_transform(scores_array.reshape(-1, 1)).flatten()
        else:
            # Handle identical scores with a fallback to uniform 0.5
            normalized_scores = np.full_like(scores_array, 0.5, dtype=float)
        return normalized_scores

    def _calculate_hybrid_scores(self, bm25_scores_normalized, faiss_scores_normalized):
        hybrid_scores = self.bm25_weight * bm25_scores_normalized + (1 - self.bm25_weight) * faiss_scores_normalized
        # self.logger.info(f"Hybrid scores: {hybrid_scores}")
        return hybrid_scores

    def _get_top_n_results(self, filtered_doc_ids, hybrid_scores, top_n, threshold):
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
    
    def _rerank_results(self, query, results):
        """
        Re-rank the retrieved documents using FlagReranker with normalized scores.

        Parameters:
        - query (str): The search query.
        - results (List[Tuple[str, float]]): A list of (doc_id, score) tuples.

        Returns:
        - List[Tuple[str, float]]: Re-ranked list of (doc_id, score) tuples with normalized scores.
        """
        # Prepare input for the re-ranker
        document_texts = [self.bm25_search.get_document(doc_id) for doc_id, _ in results]
        doc_ids = [doc_id for doc_id, _ in results]

        # Generate pairwise scores using the FlagReranker
        rerank_inputs = [[query, doc] for doc in document_texts]
        with torch.no_grad():
            rerank_scores = self.reranker.compute_score(rerank_inputs, normalize=True)

        # rerank_scores = self.reranker.compute_score(rerank_inputs, normalize=True)

        # Combine doc_ids with normalized re-rank scores and sort by scores
        reranked_results = sorted(
            zip(doc_ids, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Log and return results
        # self.logger.info(f"Re-ranked results with normalized scores: {reranked_results}")
        return reranked_results


