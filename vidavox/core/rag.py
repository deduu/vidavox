import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time
import pickle
import os
import logging
from pathlib import Path

import asyncio
from typing import List, Optional, Any, Tuple
from vidavox.retrieval import BM25_search, FAISS_search, Hybrid_search

from vidavox.utils.token_counter import TokenCounter
# from .document.doc_processor import process_doc_file
# from .generation.llm import Client
from vidavox.document import DocumentSplitter, ProcessingConfig
from typing import List, Optional, Any, Tuple, Callable


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_file_metadata(file_path: Path) -> dict:
    """
    A simple function to extract file metadata.
    You can expand this function to extract more information
    similar to the reference code.
    """
    try:
        stat = file_path.stat()
        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": stat.st_size,
            "creation_time": time.ctime(stat.st_ctime),
            "modification_time": time.ctime(stat.st_mtime),
        }
        return metadata
    except Exception as e:
        logger.warning(f"Could not get metadata for {file_path}: {e}")
        return {}

class RAG_Engine:
    def __init__(self, use_async:Optional[bool] = False, embedding_model: Optional[Any] = 'all-MiniLM-L6-v2'):
        self.use_async = use_async
        self.token_counter = TokenCounter()
        self.documents = []
        self.doc_ids = []
        self.results = []
        self.meta_data = []
        self.embedding_model = embedding_model
        self.bm25_wrapper = BM25_search()
        self.faiss_wrapper = FAISS_search(embedding_model)
        self.hybrid_search = Hybrid_search(self.bm25_wrapper, self.faiss_wrapper)
    
    
    def from_paths(
        self, file_paths: List[str], config:Optional[ProcessingConfig] = ProcessingConfig(), chunker: Optional[Callable] = None,
         show_progress: bool = False
    ):
        """Create a new RetrievalEngine instance from a list of document file paths with optional custom chunking."""
        

         # Set up tqdm progress bar if show_progress is enabled
        progress = tqdm(file_paths, desc="Processing documents", unit="file") if show_progress else file_paths
        
        for file_path in progress:
            file_name = os.path.basename(file_path)
            logger.info(f"Processing file: {file_name}")

            try:
               
                nodes = DocumentSplitter(config).process_file(file_path, chunker)
                print(nodes)

                # Add each chunk to the engine
                for idx, doc in enumerate(nodes):
                    
                    timestamp = int(time.time())  # Seconds since epoch
                    doc_id = f"{file_name}_{timestamp}_chunk{idx}"

                     # Optionally merge loader-provided metadata with file metadata
                    file_meta = get_file_metadata(Path(file_path))
                    combined_meta = {**file_meta, **doc.metadata}
                    
                    self.add_document(doc_id=doc_id, text=doc.page_content, meta_data=combined_meta)

                    # self.add_document(doc_id=doc_id, text=doc.page_content, meta_data=doc.metadata)

                logger.info(f"Successfully added {len(nodes)} chunks from {file_name}")
            
            except Exception as e:
                logger.error(f"Failed to process file {file_name}: {e}")

        return self

    def from_directory(
        self,
        directory: str,
        config: Optional[ProcessingConfig] = ProcessingConfig(),
        chunker: Optional[Callable[[Any], List[Any]]] = None,
        recursive: bool = True,
        show_progress: bool = False,
        allowed_extensions: Optional[List[str]] = None
    ):
        """
        Process all files in a directory.
        
        Args:
            directory (str): The directory path.
            config (ProcessingConfig, optional): Processing configuration.
            chunker (Callable, optional): Custom chunker function.
            recursive (bool): Whether to search subdirectories recursively.
            show_progress (bool): Whether to display a progress bar.
            allowed_extensions (List[str], optional): List of file extensions to include (e.g. [".txt", ".md"]).
        
        Returns:
            self: The updated RAG_Engine instance.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Directory {directory} does not exist.")

        # Collect files; rglob for recursive, glob for non-recursive.
        if recursive:
            files = list(dir_path.rglob("*"))
        else:
            files = list(dir_path.glob("*"))

        # Filter to include only files and exclude hidden files.
        file_paths = []
        for f in files:
            if f.is_file() and not f.name.startswith("."):
                if allowed_extensions:
                    if f.suffix.lower() in [ext.lower() for ext in allowed_extensions]:
                        file_paths.append(str(f))
                else:
                    file_paths.append(str(f))

        if not file_paths:
            raise ValueError(f"No files found in directory {directory} matching criteria.")

        logger.info(f"Found {len(file_paths)} files in {directory}.")
        return self.from_paths(file_paths, config=config, chunker=chunker, show_progress=show_progress)

    def add_document(self, doc_id, text, meta_data=None):
        self.token_counter.add_document(doc_id, text)
        self.doc_ids.append(doc_id)
        self.documents.append(text)
        self.meta_data.append(meta_data)
        self.bm25_wrapper.add_document(doc_id, text)
        self.faiss_wrapper.add_document(doc_id, text)

    def add_documents(self, docs: List[Tuple[str, str, Optional[Any]]] ) -> None:
        """Batch add documents to all components."""
        try:
            # Prepare batch data
            doc_ids, texts, meta_datas = zip(*docs)
            
            # Handle missing meta_data (crucial change)
            meta_datas = [meta if meta is not None else {} for meta in meta_datas]  # Provide default {}

            # Update internal state
            self.doc_ids.extend(doc_ids)
            self.documents.extend(texts)
            self.meta_data.extend(meta_datas)
            
            # Batch update token counter
            for doc_id, text in zip(doc_ids, texts):
                self.token_counter.add_document(doc_id, text)
            
            # Batch update search indices
            self.bm25_wrapper.add_documents(list(zip(doc_ids, texts)))
            self.faiss_wrapper.add_documents(list(zip(doc_ids, texts)))
            
        except Exception as e:
            raise RuntimeError(f"Failed to add documents in batch: {str(e)}")

    # take longer time
    # def add_document(self, doc_id: str, text: str, meta_data: Optional[Any] = None) -> None:
    #     """Add single document using batch operation."""
    #     self.add_documents([(doc_id, text, meta_data)])
    

    def delete_document(self, doc_id):
        try:
            index = self.doc_ids.index(doc_id)
            del self.doc_ids[index]
            del self.documents[index]
            self.bm25_wrapper.remove_document(index)
            self.faiss_wrapper.remove_document(index)
            self.token_counter.remove_document(doc_id)
        except ValueError:
            logging.warning(f"Document ID {doc_id} not found.")

    def query(self, query_text, keywords:Optional[List[str]] = None, threshold: Optional[float] = 0.4, top_k: Optional[int] = 5,prefixes=None):
        if self.use_async:
            # task = [self.retrieve_async(query_text,keywords,threshold,top_k,prefixes)]
            result = asyncio.run(self.retrieve_async(query_text,keywords,threshold,top_k,prefixes))
        else:
            result = self.retrieve(query_text,keywords,threshold,top_k,prefixes)
        
        return result


    def retrieve(self, query_text, keywords:Optional[List[str]] = None, threshold: Optional[float] = 0.4, top_k: Optional[int] = 5,prefixes=None):
        results = self.search(
            query_text,
            keywords=keywords,
            top_k=top_k,
            threshold=threshold,
            prefixes=prefixes
        )
        retrieved_docs = []
        if results:
            seen_docs = set()
            for doc_id, score in results:
                if doc_id not in seen_docs:
                     # Check if the doc_id exists in self.doc_ids
                    if doc_id not in self.doc_ids:
                        logger.error(f"doc_id {doc_id} not found in self.doc_ids")
                    seen_docs.add(doc_id)
                  
                    # Fetch the index of the document
                    try:
                        index = self.doc_ids.index(doc_id)
                    except ValueError as e:
                        logger.error(f"Error finding index for doc_id {doc_id}: {e}")
                        continue

                     # Validate index range
                    if index >= len(self.documents) or index >= len(self.meta_data):
                        logger.error(f"Index {index} out of range for documents or metadata")
                        continue

                    doc = self.documents[index]
                    
                    meta_data = self.meta_data[index]
                 
                    url = meta_data.get('source', 'unknown_url')  # Default URL fallback
                  
                    self.results.append(doc)
                    retrieved_docs.append({"id":doc_id, "url":url, "text": doc})
            return retrieved_docs
        else:
            return [{"id": "None.", "url": "None.", "text": None}]
        
    async def retrieve_async(self, query_text, keywords:Optional[List[str]] = None, threshold: Optional[float] = 0.4, top_k: Optional[int] = 5,prefixes=None):
        results = await self.search_async(
            query_text,
            keywords=keywords,
            top_k=top_k,
            threshold=threshold,
            prefixes=prefixes
        )
        retrieved_docs = []
        if results:
            seen_docs = set()
            for doc_id, score in results:
                if doc_id not in seen_docs:
                     # Check if the doc_id exists in self.doc_ids
                    if doc_id not in self.doc_ids:
                        logger.error(f"doc_id {doc_id} not found in self.doc_ids")
                    seen_docs.add(doc_id)
                  
                    # Fetch the index of the document
                    try:
                        index = self.doc_ids.index(doc_id)
                    except ValueError as e:
                        logger.error(f"Error finding index for doc_id {doc_id}: {e}")
                        continue

                     # Validate index range
                    if index >= len(self.documents) or index >= len(self.meta_data):
                        logger.error(f"Index {index} out of range for documents or metadata")
                        continue

                    doc = self.documents[index]
                    
                    meta_data = self.meta_data[index]
                 
                    url = meta_data.get('source', 'unknown_url')  # Default URL fallback
                  
                    self.results.append(doc)
                    retrieved_docs.append({"id":doc_id, "url":url, "text": doc})
            return retrieved_docs
        else:
            return [{"id":doc_id, "url": "None.", "text": None}]
        
    async def search_async(self, query_text, keywords:Optional[List[str]] = None, threshold: Optional[float] = 0.4, top_k: Optional[int] = 5,prefixes=None):
        try:
            results = await self.hybrid_search.advanced_search_async(
                query_text,
                keywords=keywords,
                top_n=top_k,
                threshold=threshold,
                prefixes=prefixes
            )

            return results
        except Exception as e:
            raise RuntimeError(f"Failed to search: {str(e)}")
    
    def search(self, query_text, keywords:Optional[List[str]] = None, threshold: Optional[float] = 0.4, top_k: Optional[int] = 5,prefixes=None):
        try:
            results = self.hybrid_search.advanced_search(
                query_text,
                keywords=keywords,
                top_n=top_k,
                threshold=threshold,
                prefixes=prefixes
            )

            return results
        except Exception as e:
            raise RuntimeError(f"Failed to search: {str(e)}")

    def get_document(self, doc_id: str) -> Optional[str]:
        """
        Retrieves a document by its document ID.
        
        Parameters:
        - doc_id (str): The ID of the document to retrieve.

        Returns:
        - Optional[str]: The document text if found, None if not found.
        """
        try:
            index = self.doc_ids.index(doc_id)
            return self.documents[index]
        except ValueError:
            print(f"Document ID {doc_id} not found.")
            return None

    def get_document_count(self) -> int:
        """
        Returns the total number of documents in the index.

        Returns:
        - int: Number of documents.
        """
        return len(self.documents)  
    

    def get_total_tokens(self):
        return self.token_counter.get_total_tokens()
    def get_context(self):
        context = "\n".join(self.results)
        return context

    def save_state(self, path):
    # Save doc_ids, documents, and token counter state
        with open(f"{path}_state.pkl", 'wb') as f:
            pickle.dump({
                "doc_ids": self.doc_ids,
                "documents": self.documents,
                "meta_data": self.meta_data,
                "token_counts": self.token_counter.doc_tokens
            }, f)

    def load_state(self, path):
        if os.path.exists(f"{path}_state.pkl"):
            with open(f"{path}_state.pkl", 'rb') as f:
                state_data = pickle.load(f)
                self.doc_ids = state_data["doc_ids"]
                self.documents = state_data["documents"]
                self.meta_data = state_data["meta_data"]
                self.token_counter.doc_tokens = state_data["token_counts"]

            # Clear and rebuild BM25 and FAISS
            self.bm25_wrapper.clear_documents()
            self.faiss_wrapper.clear_documents()
            for doc_id, document in zip(self.doc_ids, self.documents):
                self.bm25_wrapper.add_document(doc_id, document)
                self.faiss_wrapper.add_document(doc_id, document)

            self.token_counter.total_tokens = sum(self.token_counter.doc_tokens.values())
            logging.info("System state loaded successfully with documents and indices rebuilt.")
        else:
            logging.info("No previous state found, initializing fresh state.")
            self.doc_ids = []
            self.documents = []
            self.meta_data = []  # Reset meta_data
            self.token_counter = TokenCounter()
            self.bm25_wrapper = BM25_search()
            self.faiss_wrapper = FAISS_search(self.embedding_model)
            self.hybrid_search = Hybrid_search(self.bm25_wrapper, self.faiss_wrapper)


