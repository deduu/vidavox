import threading
from typing import List, Tuple, Optional, Any, Union, Dict
from sentence_transformers import SentenceTransformer
import logging
import faiss
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISS_search:
    """
    A wrapper class for FAISS similarity search with efficient document management.
    Uses all-MiniLM-L6-v2 as default embedding model if none is provided.
    """
    
    DEFAULT_MODEL_NAME = 'all-MiniLM-L6-v2'

    def __init__(self, embedding_model: Optional[Any] = None):
        """
        Initialize FAISS_search with an embedding model.
        """
        self.doc_dict: Dict[str, str] = {}   # Maps doc_id to document text
        self.id_map: Dict[str, int] = {}       # Maps doc_id to an integer ID for FAISS
        self.next_index_id: int = 0            # Next available integer ID
        self.embedding_model = self._initialize_embedding_model(embedding_model)
        self.dimension = self.get_embedding_dimension()
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        self.lock = threading.Lock()

    def _initialize_embedding_model(
        self, 
        embedding_model: Optional[Union[str, SentenceTransformer]] = None
    ) -> SentenceTransformer:
        try:
            if embedding_model is None:
                logger.info(f"Using default model: {self.DEFAULT_MODEL_NAME}")
                embedding_model = SentenceTransformer(self.DEFAULT_MODEL_NAME)
            elif isinstance(embedding_model, str):
                logger.info(f"Loading model: {embedding_model}")
                embedding_model = SentenceTransformer(embedding_model)
            elif isinstance(embedding_model, SentenceTransformer):
                logger.info("Using provided SentenceTransformer instance")
            else:
                raise ValueError("Invalid embedding_model type")
            return embedding_model
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def get_embedding_dimension(self) -> int:
        if self.embedding_model is None:
            return 0
        else:
            # Encode a dummy text to determine embedding dimension
            return len(self.embedding_model.encode("embedding"))

    def add_document(self, doc_id: str, doc: str) -> None:
        """
        Adds a single document to the FAISS index.
        """
        self.add_documents([(doc_id, doc)])

    def add_documents(self, docs_with_ids: List[Tuple[str, str]]) -> None:
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
                    logger.warning(f"Skipping invalid document or ID: {doc_id}")
                    continue
                # If the document already exists, you might decide to update or skip.
                if doc_id in self.doc_dict:
                    logger.info(f"Document ID {doc_id} already exists; skipping addition.")
                    continue
                # Store the document text for O(1) lookup
                self.doc_dict[doc_id] = doc
                # Assign a new integer ID for FAISS
                current_index = self.next_index_id
                self.id_map[doc_id] = current_index
                new_ids.append(current_index)
                new_texts.append(doc)
                self.next_index_id += 1
        
        if new_texts:
            try:
                embeddings = self.embedding_model.encode(new_texts, convert_to_numpy=True).astype('float32')
                if embeddings.shape[0] == 0:
                    raise ValueError("Empty embeddings generated")
                with self.lock:
                    self.index.add_with_ids(embeddings, np.array(new_ids).astype('int64'))
            except Exception as e:
                raise RuntimeError(f"Failed to add documents: {str(e)}")

    def remove_document(self, doc_id: str) -> bool:
        """
        Removes a document from the FAISS index using O(1) dictionary lookup,
        then rebuilds the index.
        """
        with self.lock:
            if doc_id not in self.doc_dict:
                logger.warning(f"Document ID {doc_id} not found.")
                return False
            # Remove document from dictionaries
            del self.doc_dict[doc_id]
            del self.id_map[doc_id]
        # Rebuild the index since individual deletion is nontrivial
        self._rebuild_index()
        logger.info(f"Removed document ID: {doc_id}")
        return True

    def _rebuild_index(self) -> None:
        """
        Rebuilds the FAISS index from the current documents.
        """
        with self.lock:
            all_doc_ids = list(self.doc_dict.keys())
            all_texts = [self.doc_dict[d] for d in all_doc_ids]
            # Reassign integer IDs continuously
            self.id_map = {}
            new_ids = []
            self.next_index_id = 0
            for doc_id in all_doc_ids:
                self.id_map[doc_id] = self.next_index_id
                new_ids.append(self.next_index_id)
                self.next_index_id += 1
            try:
                embeddings = self.embedding_model.encode(all_texts, convert_to_numpy=True).astype('float32')
            except Exception as e:
                raise RuntimeError(f"Failed to rebuild embeddings: {str(e)}")
            # Reinitialize and rebuild the FAISS index
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
            self.index.add_with_ids(embeddings, np.array(new_ids).astype('int64'))

    def search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the FAISS index for the top-k documents matching the query.
        Returns a tuple of (distances, indices).
        """
        with self.lock:
            if self.index.ntotal == 0:
                logger.info("FAISS index is empty. No results can be returned.")
                return np.array([]), np.array([])
        try:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        except Exception as e:
            raise RuntimeError(f"Failed to encode query: {str(e)}")
        distances, indices = self.index.search(query_embedding, k)
        return distances[0], indices[0]
    
    def get_document(self, doc_id: str) -> str:
        """
        Retrieves a document by its document ID in O(1) time.
        """
        with self.lock:
            return self.doc_dict.get(doc_id, "")
    
    def clear_documents(self) -> None:
        """
        Clears all documents from the FAISS index.
        """
        with self.lock:
            self.doc_dict.clear()
            self.id_map.clear()
            self.next_index_id = 0
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        logger.info("FAISS documents cleared and index reset.")
