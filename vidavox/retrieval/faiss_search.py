# faiss_wrapper.py
from typing import List, Tuple, Optional, Any, Union
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
        Initialize FAISSSearch with an embedding model.
        
        Parameters:
        - embedding_model: Model that implements encode() method for text embeddings.
                         If None, uses all-MiniLM-L6-v2 by default.
        """
    
        self.documents = []
        self.doc_ids = []
        
        self.embedding_model = self._initialize_embedding_model(embedding_model)
        self.dimension = self.get_embedding_dimension()
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

    def _initialize_embedding_model(self, embedding_model = Optional[Union[str, SentenceTransformer]])-> SentenceTransformer:
        try:
                if embedding_model is None:
                    print(f"Using default model: {self.DEFAULT_MODEL_NAME}")
                    embedding_model = SentenceTransformer(self.DEFAULT_MODEL_NAME)
                
                    return embedding_model
                elif isinstance(embedding_model, str):
                    logger.info(f"Loading model: {embedding_model}")
                    embedding_model = SentenceTransformer(embedding_model)
                 
                    return embedding_model
                elif isinstance(embedding_model, SentenceTransformer):
                    print("Using provided SentenceTransformer instance")
                
                    return embedding_model
                else:
                    raise ValueError("Invalid embedding_model type")
                
                
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")
        self.dimension = len(embedding_model.encode("embedding"))
        return embedding_model
    def get_embedding_dimension(self):
        if self.embedding_model is None:
            return 0
        else:
            return len(self.embedding_model.encode("embedding"))
        
    def add_documents(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        if not docs_with_ids:
            return

        new_ids, new_docs = zip(*docs_with_ids)
        try:
            embeddings = self.embedding_model.encode(new_docs, convert_to_numpy=True).astype('float32')
            if embeddings.shape[0] == 0:
                raise ValueError("Empty embeddings generated")

            start_idx = len(self.documents)
            id_array = np.arange(start_idx, start_idx + len(new_docs)).astype('int64')
            
            self.index.add_with_ids(embeddings, id_array)
            self.documents.extend(new_docs)
            self.doc_ids.extend(new_ids)
        except Exception as e:
            raise RuntimeError(f"Failed to add documents: {str(e)}")

    def add_document(self, doc_id: str, doc: str) -> None:
        self.add_documents([(doc_id, doc)])
    # def add_document(self, doc_id, new_doc):
        
    #     # Encode and add document with its index as ID
    #     embedding = self.embedding_model.encode([new_doc], convert_to_numpy=True).astype('float32')

    #     if embedding.size == 0:
    #         print("No documents to add to FAISS index.")
    #         return

    #     idx = len(self.documents) - 1
    #     id_array = np.array([idx]).astype('int64')
    #     self.index.add_with_ids(embedding, id_array)
    #     self.documents.append(new_doc)
    #     self.doc_ids.append(doc_id)

    def remove_document(self, index):
        if 0 <= index < len(self.documents):
            del self.documents[index]
            del self.doc_ids[index]
            # Rebuild the index
            self.build_index()
        else:
            print(f"Index {index} is out of bounds.")

    def build_index(self):
        embeddings = self.embedding_model.encode(self.documents, convert_to_numpy=True).astype('float32')
        idx_array = np.arange(len(self.documents)).astype('int64')
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        self.index.add_with_ids(embeddings, idx_array)

    def search(self, query, k):
        if self.index.ntotal == 0:
            # No documents in the index
            print("FAISS index is empty. No results can be returned.")
            return np.array([]), np.array([])  # Return empty arrays for distances and indices
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        return distances[0], indices[0]
    
    def clear_documents(self) -> None:
        """
        Clears all documents from the FAISS index.
        """
        self.documents = []
        self.doc_ids = []
        # Reset the FAISS index
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        print("FAISS documents cleared and index reset.")
    
    def get_document(self, doc_id: str) -> str:
        """
        Retrieves a document by its document ID.
        
        Parameters:
        - doc_id (str): The ID of the document to retrieve.

        Returns:
        - str: The document text if found, otherwise an empty string.
        """
        try:
            index = self.doc_ids.index(doc_id)
            return self.documents[index]
        except ValueError:
            print(f"Document ID {doc_id} not found.")
            return ""
