# Core components module (rag_components.py)
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Union, Optional
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    doc_id: str
    text: str
    meta_data: Dict[str, Any]
    score: float

class BaseResultFormatter:
    def format(self, result: SearchResult) -> Dict[str, Any]:
        # Default implementation: simply convert the data class to a dictionary
        return asdict(result)

class CustomResultFormatter(BaseResultFormatter):
    def format(self, result: SearchResult) -> Dict[str, Any]:
        # Customize the result format as needed
        return {
            "doc_id": result.doc_id,
            "page_content": result.text,
   
            "relevance": result.score,
        }
class DefaultResultFormatter(BaseResultFormatter):
    def format(self, result: SearchResult) -> Dict[str, Any]:
        # Default implementation: simply convert the data class to a dictionary
        return{
            "id": result.doc_id,
            "url": result.meta_data.get('source', 'unknown_url'),
            "text": result.text,
            "score": result.score,
        }

class DocumentManager:
    """Manages document storage and basic operations."""
    
    def __init__(self):
        self.doc_ids = []
        self.documents = []
        self.meta_data = []
        
    def add_document(self, doc_id: str, text: str, meta_data: Optional[Dict] = None) -> None:
        """Add a single document to the collection."""
        self.doc_ids.append(doc_id)
        self.documents.append(text)
        self.meta_data.append(meta_data or {})
        
    def add_documents(self, docs: List[Tuple[str, str, Optional[Dict]]]) -> None:
        """Add multiple documents to the collection."""
        for doc_id, text, meta in docs:
            self.add_document(doc_id, text, meta)
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Retrieve a document by its ID."""
        try:
            index = self.doc_ids.index(doc_id)
            return self.documents[index]
        except ValueError:
            logger.warning(f"Document ID {doc_id} not found.")
            return None
    
    def get_metadata(self, doc_id: str) -> Optional[Dict]:
        """Retrieve metadata for a document by its ID."""
        try:
            index = self.doc_ids.index(doc_id)
            return self.meta_data[index]
        except ValueError:
            logger.warning(f"Document ID {doc_id} not found.")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from the collection."""
        try:
            index = self.doc_ids.index(doc_id)
            del self.doc_ids[index]
            del self.documents[index]
            del self.meta_data[index]
            return True
        except ValueError:
            logger.warning(f"Cannot delete: Document ID {doc_id} not found.")
            return False
    
    def get_document_count(self) -> int:
        """Return the total number of documents."""
        return len(self.documents)
    
    def get_all_contents(self) -> str:
        """Return a single string containing all document texts concatenated with newlines."""
        docs = self.documents.copy()
        text_content = "\n".join(docs)
        return text_content

    def clear(self) -> None:
        """Clear all documents."""
        self.doc_ids.clear()
        self.documents.clear()
        self.meta_data.clear()


class FileProcessor:
    """Handles document loading and processing from files and directories."""
    
    @staticmethod
    def get_file_metadata(file_path: Path) -> dict:
        """Extract basic metadata from a file."""
        try:
            stat = file_path.stat()
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": stat.st_size,
                "creation_time": time.ctime(stat.st_ctime),
                "modification_time": time.ctime(stat.st_mtime),
            }
        except Exception as e:
            logger.warning(f"Could not get metadata for {file_path}: {e}")
            return {}
    
    @staticmethod
    def collect_files(directory: str, recursive: bool = True, 
                     allowed_extensions: Optional[List[str]] = None) -> List[str]:
        """Collect all relevant files from a directory."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Directory {directory} does not exist.")

        # Collect files using appropriate glob method
        files = list(dir_path.rglob("*")) if recursive else list(dir_path.glob("*"))

        # Filter files based on criteria
        file_paths = []
        for f in files:
            if f.is_file() and not f.name.startswith("."):
                if allowed_extensions:
                    if f.suffix.lower() in [ext.lower() for ext in allowed_extensions]:
                        file_paths.append(str(f))
                else:
                    file_paths.append(str(f))

        if not file_paths:
            logger.warning(f"No files found in directory {directory} matching criteria.")
            
        return file_paths


# State management module (state_manager.py)
import os
import pickle
from typing import Dict, Any

class StateManager:
    """Handles persistence of RAG engine state."""
    
    @staticmethod
    def save_state(path: str, state_data: Dict[str, Any]) -> bool:
        """Save engine state to disk."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(f"{path}_state.pkl", 'wb') as f:
                pickle.dump(state_data, f)
            logger.info(f"State successfully saved to {path}_state.pkl")
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    @staticmethod
    def load_state(path: str) -> Optional[Dict[str, Any]]:
        """Load engine state from disk."""
        if os.path.exists(f"{path}_state.pkl"):
            try:
                with open(f"{path}_state.pkl", 'rb') as f:
                    state_data = pickle.load(f)
                logger.info(f"State successfully loaded from {path}_state.pkl")
                return state_data
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                return None
        else:
            logger.info(f"No state file found at {path}_state.pkl")
            return None


# Main RAG Engine module (rag_engine.py)
import asyncio
from typing import List, Optional, Any, Dict, Tuple, Callable, Union

from vidavox.retrieval import BM25_search, FAISS_search, Hybrid_search, SearchMode
from vidavox.utils.token_counter import TokenCounter
from vidavox.document import DocumentSplitter, ProcessingConfig

class RAG_Engine:
    """Main Retrieval Augmented Generation engine with modular components."""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', use_async: bool = False):
        """Initialize the RAG engine with all required components."""
        self.use_async = use_async
        self.token_counter = TokenCounter()
        self.embedding_model = embedding_model
        
        # Initialize document store
        self.doc_manager = DocumentManager()
        
        # Initialize search components
        self.bm25_wrapper = BM25_search()
        self.faiss_wrapper = FAISS_search(embedding_model)
        self.hybrid_search = Hybrid_search(self.bm25_wrapper, self.faiss_wrapper)
        
        # Results cache
        self.results = []
    
    def _process_csv_as_dataframe(
        self, 
        file_path: str, 
        text_col: str,
        metadata_cols: List[str]
    ) -> List[Tuple[str, str, Dict]]:
        """Process a CSV file as a DataFrame."""
        try:
            df = pd.read_csv(file_path)
            batch_docs = []
            file_name = os.path.basename(file_path)
            
            # Verify text column exists
            if text_col not in df.columns:
                logger.error(f"Text column '{text_col}' not found in CSV file {file_name}. Skipping file.")
                return []
            
            # Process each row
            for idx, row in df.iterrows():
                doc_id = f"{file_name}_{idx}"
                doc_text = str(row[text_col])
                
                # Create metadata dictionary
                doc_metadata = {
                    'source': file_path,
                    'file_name': file_name,
                    'row_index': idx,
                    'file_type': 'csv'
                }
                
                # Add specified metadata columns
                for col in metadata_cols:
                    if col in df.columns:
                        doc_metadata[col] = row[col]
                    
                batch_docs.append((doc_id, doc_text, doc_metadata))
                
            logger.info(f"Successfully processed {len(df)} rows from CSV file {file_name}")
            return batch_docs
            
        except Exception as e:
            logger.error(f"Failed to process CSV file {file_path}: {e}")
            return []
    
    def _process_excel_as_dataframe(
        self, 
        file_path: str, 
        text_col: str,
        metadata_cols: List[str]
    ) -> List[Tuple[str, str, Dict]]:
        """Process an Excel file as a DataFrame."""
        try:
            df = pd.read_excel(file_path)
            batch_docs = []
            file_name = os.path.basename(file_path)

            # Verify text column exists
            if text_col not in df.columns:
                logger.error(f"Text column '{text_col}' not found in Excel file {file_name}. Skipping file.")
                return []

            # Process each row
            for idx, row in df.iterrows():
                doc_id = f"{file_name}_{idx}"
                doc_text = str(row[text_col])

                # Create metadata dictionary
                doc_metadata = {
                    'source': file_path,
                    'file_name': file_name,
                    'row_index': idx,
                    'file_type': 'excel'
                }

                # Add specified metadata columns
                for col in metadata_cols:
                    if col in df.columns:
                        doc_metadata[col] = row[col]

                batch_docs.append((doc_id, doc_text, doc_metadata))

            logger.info(f"Successfully processed {len(df)} rows from Excel file {file_name}")
            return batch_docs

        except Exception as e:
            logger.error(f"Failed to process Excel file {file_path}: {e}")
            return []


    def _process_file(self, file_path: str, config: ProcessingConfig, chunker: Optional[Callable] = None) -> List[Tuple[str, str, Dict]]:
        """Process a single file into the required document format."""
        processor = FileProcessor()
        file_name = os.path.basename(file_path)
        batch_docs = []

        try:
            # Split the document into nodes
            nodes = DocumentSplitter(config).run(file_path, chunker)
            
            # Process each chunk
            for idx, doc in enumerate(nodes):
                timestamp = int(time.time())
                doc_id = f"{file_name}_{timestamp}_chunk_{idx}"
                
                # Get and merge metadata
                file_meta = processor.get_file_metadata(Path(file_path))
                file_meta['file_type'] = Path(file_path).suffix.lower()[1:]  # Add file type
                combined_meta = {**file_meta, **doc.metadata}
                
                batch_docs.append((doc_id, doc.page_content, combined_meta))
                
            logger.info(f"Successfully processed {len(nodes)} chunks from {file_name}")
            
        except Exception as e:
            logger.error(f"Failed to process file {file_name}: {e}")
            
        return batch_docs
    
    def from_paths(
        self,
        sources: List[str],
        config: Optional[ProcessingConfig] = None,
        chunker: Optional[Callable] = None,
        show_progress: bool = False,
        load_csv_as_pandas_dataframe: Optional[bool] = False,
        load_excel_as_pandas_dataframe: bool = False,
        text_col: Optional[str] = None,
        metadata_cols: Optional[List[str]] = None
    ) -> 'RAG_Engine':
        """
        Build engine from a list of file paths, with mixed format support.
        
        Args:
            sources (List[str]): List of file paths (can be mixed formats)
            config (ProcessingConfig, optional): Processing configuration for non-CSV files
            chunker (Callable, optional): Optional custom chunking function
            show_progress (bool): Whether to show progress bar
            load_as_pandas_dataframe (bool): If True, processes .csv files using pandas
            load_excel_as_pandas_dataframe (bool): If True, processes .xlsx/.xls files using pandas
            text_col (str, optional): Column name containing main text (required for CSV processing)
            metadata_cols (List[str], optional): List of columns to include as metadata for CSV files
        """
        if load_csv_as_pandas_dataframe and not text_col:
            raise ValueError("text_col must be specified when load_as_pandas_dataframe is True")
        
        if load_excel_as_pandas_dataframe and not text_col:
            raise ValueError("text_col must be specified when load_excel_as_pandas_dataframe is True")
            
        config = config or ProcessingConfig()
        metadata_cols = metadata_cols or []
        batch_docs = []
        BATCH_SIZE = 100

        # Configure progress tracking
        iterator = tqdm(sources, desc="Processing files", unit="file") if show_progress else sources
        
        # Collect processing statistics
        stats = {
            'csv_files_processed': 0,
            'excel_files_processed': 0,
            'other_files_processed': 0,
            'failed_files': 0
        }
        
        # Process files
        for file_path in iterator:
            try:
                is_csv = file_path.lower().endswith('.csv')
                is_excel = file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls')
                
                # 1) CSV processing
                if is_csv and load_csv_as_pandas_dataframe:
                    file_docs = self._process_csv_as_dataframe(
                        file_path, 
                        text_col=text_col,
                        metadata_cols=metadata_cols
                    )
                    # print(f"file_docs: {file_docs}")
                    if file_docs:
                        stats['csv_files_processed'] += 1

                # 2) Excel processing
                elif is_excel and load_excel_as_pandas_dataframe:
                    file_docs = self._process_excel_as_dataframe(
                        file_path,
                        text_col=text_col,
                        metadata_cols=metadata_cols
                    )
                    if file_docs:
                        stats['excel_files_processed'] += 1

                # 3) Other file types
                else:
                    file_docs = self._process_file(file_path, config, chunker)
                    if file_docs:
                        stats['other_files_processed'] += 1
                
                if not file_docs:
                    stats['failed_files'] += 1
                    continue
                    
                batch_docs.extend(file_docs)
                
                # Process batch if it reaches threshold
                if len(batch_docs) >= BATCH_SIZE:
                    self._process_document_batch(batch_docs)
                    batch_docs = []
                    
            except Exception as e:
                stats['failed_files'] += 1
                logger.error(f"Failed to process file {file_path}: {e}")
                continue

        # Process any remaining documents
        if batch_docs:
            self._process_document_batch(batch_docs)
            
        # Log processing summary
        logger.info(
            f"Processing complete:\n"
            f"- CSV files processed: {stats['csv_files_processed']}\n"
            f"- Excel files processed: {stats['excel_files_processed']}\n"
            f"- Other files processed: {stats['other_files_processed']}\n"
            f"- Failed files: {stats['failed_files']}"
        )
            
        return self
    
    def from_directory(
        self,
        directory: str,
        config: Optional[ProcessingConfig] = None,
        chunker: Optional[Callable] = None,
        recursive: bool = True,
        show_progress: bool = False,
        allowed_extensions: Optional[List[str]] = None,
        load_csv_as_pandas_dataframe: Optional[bool] = False,
        load_excel_as_pandas_dataframe: bool = False,
        text_col: Optional[str] = None,
        metadata_cols: Optional[List[str]] = None
    ) -> 'RAG_Engine':
        """
        Build engine from all files in a directory.
        
        Args:
            directory (str): Directory path to process
            config (ProcessingConfig, optional): Processing configuration for non-CSV files
            chunker (Callable, optional): Optional custom chunking function
            recursive (bool): Whether to search subdirectories
            show_progress (bool): Whether to show progress bar
            allowed_extensions (List[str], optional): List of allowed file extensions
            load_as_pandas_dataframe (bool): If True, processes .csv files using pandas
            load_excel_as_pandas_dataframe (bool): If True, processes .xlsx/.xls files using pandas
            text_col (str, optional): Column name containing main text (required for CSV processing)
            metadata_cols (List[str], optional): List of columns to include as metadata for CSV files
        """
        processor = FileProcessor()
        file_paths = processor.collect_files(directory, recursive, allowed_extensions)
        
        if not file_paths:
            raise ValueError(f"No files found in directory {directory} matching criteria.")
            
        logger.info(f"Found {len(file_paths)} files in {directory}.")
        return self.from_paths(
            file_paths,
            config=config or ProcessingConfig(),
            chunker=chunker,
            show_progress=show_progress,
            load_csv_as_pandas_dataframe=load_csv_as_pandas_dataframe,
            load_excel_as_pandas_dataframe=load_excel_as_pandas_dataframe,
            text_col=text_col,
            metadata_cols=metadata_cols
        )
  

    
    def _process_document_batch(self, docs: List[Tuple[str, str, Dict]]) -> None:
        """Process a batch of documents efficiently."""
        try:
            # Extract components for batch processing
            doc_ids, texts, meta_datas = zip(*docs)
            # print(f"doc_ids: {doc_ids}, texts: {texts}, meta_datas: {meta_datas}")
            
            # Update document manager
            self.doc_manager.add_documents(docs)
            
            # Update token counter
            for doc_id, text in zip(doc_ids, texts):
                self.token_counter.add_document(doc_id, text)
            
            # Update search indices in batch
            self.bm25_wrapper.add_documents(list(zip(doc_ids, texts)))
            self.faiss_wrapper.add_documents(list(zip(doc_ids, texts)))
            
        except Exception as e:
            logger.error(f"Failed to process document batch: {e}")
            raise
    
    def add_document(self, doc_id: str, text: str, meta_data: Optional[Dict] = None) -> None:
        """Add a single document to the engine."""
        try:
            # Update document manager
            self.doc_manager.add_document(doc_id, text, meta_data)
            
            # Update token counter
            self.token_counter.add_document(doc_id, text)
            
            # Update search indices
            self.bm25_wrapper.add_document(doc_id, text)
            self.faiss_wrapper.add_document(doc_id, text)
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            raise
    
    def add_documents(self, docs: List[Tuple[str, str, Optional[Dict]]]) -> None:
        """Add multiple documents to the engine efficiently."""
        self._process_document_batch(docs)
    
    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from the engine."""
        try:
            # Find the document index
            if doc_id not in self.doc_manager.doc_ids:
                logger.warning(f"Document ID {doc_id} not found.")
                return False
                
            index = self.doc_manager.doc_ids.index(doc_id)
            
            # Remove from search indices
            self.bm25_wrapper.remove_document(index)
            self.faiss_wrapper.remove_document(index)
            
            # Remove from token counter
            self.token_counter.remove_document(doc_id)
            
            # Remove from document manager
            self.doc_manager.delete_document(doc_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def query(self, query_text: str, keywords: Optional[List[str]] = None, 
             threshold: float = 0.4, top_k: int = 5, result_formatter: Optional[BaseResultFormatter] = None, prefixes=None) -> List[Dict]:
        """Query the engine for relevant documents."""
        if self.use_async:
            return asyncio.run(self.retrieve_async(query_text, keywords, threshold, top_k, result_formatter=result_formatter, prefixes=prefixes))
        else:
            return self.retrieve(query_text, keywords, threshold, top_k, result_formatter=result_formatter, prefixes=prefixes)
    
    def retrieve(self, query_text: str, keywords: Optional[List[str]] = None,
                threshold: float = 0.4, top_k: int = 5, result_formatter: Optional[BaseResultFormatter] = None,prefixes=None) -> List[Dict]:
        """Synchronously retrieve relevant documents."""
        results = self.search(query_text, keywords, top_k, threshold, prefixes)
        return self._process_search_results(results, result_formatter=result_formatter)
    
    async def retrieve_async(self, query_text: str, keywords: Optional[List[str]] = None,
                           threshold: float = 0.4, top_k: int = 5, result_formatter: Optional[BaseResultFormatter] = None,prefixes=None) -> List[Dict]:
        """Asynchronously retrieve relevant documents."""
        results = await self.search_async(query_text, keywords, top_k, threshold, prefixes)
        return self._process_search_results(results, result_formatter=result_formatter)
    
    def retrieve_best_chunk_per_document(
        self,
        query_text: str,
        keywords: Optional[List[str]] = None,
        per_doc_top_n: int = 5,
        threshold: float = 0.53,
        prefixes=None,
        result_formatter: Optional[BaseResultFormatter] = None,
        search_mode: Optional[SearchMode] = SearchMode.HYBRID,
        sort_globally: Optional[bool] = False
    ) -> List[Dict]:
        """
        Retrieve the best matching chunks per document for a given query.
        
        This method first retrieves candidate chunks using the advanced search, groups them by document,
        and then processes and formats the results.
        
        Args:
            query_text (str): The search query.
            keywords (List[str], optional): Keywords for BM25.
            per_doc_top_n (int): Maximum number of chunks to return per document.
            threshold (float): Minimum score threshold.
            prefixes: Optional prefixes to filter document IDs.
            result_formatter (BaseResultFormatter, optional): Formatter to standardize output.
            search_mode (SearchMode, optional): The search mode to use.
            sort_globally (bool, optional): Whether to sort results across all documents.
        
        Returns:
            List[Dict]: A list of formatted search results.
        """
        # Retrieve candidate chunks grouped by document.
        candidate_results = self.search_best_chunk_per_document(
            query_text, keywords, per_doc_top_n=per_doc_top_n,
            threshold=threshold, prefixes=prefixes, search_mode=search_mode,
            sort_globally=sort_globally
        )
        
        # Process the candidate results into a standard format.
        return self._process_search_results(candidate_results, result_formatter=result_formatter)

    
    def _process_search_results(self, results: List[Tuple[str, float]], result_formatter: Optional[BaseResultFormatter] = None) -> List[Dict]:
        """Process search results into a standardized format."""
        if not results:
            return [{"id": "None.", "url": "None.", "text": None}]
            
        retrieved_docs = []
        seen_docs = set()
        self.results = []  # Clear previous results

          # Use the provided formatter or fall back to the default
        formatter = result_formatter or DefaultResultFormatter()
        
        for doc_id, score in results:
            if doc_id in seen_docs or doc_id not in self.doc_manager.doc_ids:
                continue
                
            seen_docs.add(doc_id)
            
            try:
                index = self.doc_manager.doc_ids.index(doc_id)
                doc = self.doc_manager.documents[index]
                meta_data = self.doc_manager.meta_data[index]
                # url = meta_data.get('source', 'unknown_url')
                self.results.append(doc)
                 # Create a SearchResult instance
                search_result = SearchResult(doc_id=doc_id, text=doc, meta_data=meta_data, score=score)
                # Format the result using the formatter
                formatted_result = formatter.format(search_result)
               
                retrieved_docs.append(formatted_result)


                
            except (ValueError, IndexError) as e:
                logger.error(f"Error processing result for doc_id {doc_id}: {e}")
        
        return retrieved_docs if retrieved_docs else [{"id": "None.", "url": "None.", "text": None}]
    

    def search_best_chunk_per_document(self, query_text, keywords, per_doc_top_n=5, threshold=0.53, prefixes=None, search_mode: Optional[SearchMode] = SearchMode.HYBRID, sort_globally: Optional[bool]= False):
        """
        Perform an advanced search and return the top 'per_doc_top_n' chunks per document.
        Assumes that the doc_id is structured as: fileName_timestamp_chunk{idx}.
        
        Parameters:
        query (str): The search query.
        keywords (list): Keywords for BM25.
        per_doc_top_n (int): Maximum number of chunks to return for each document.
        threshold (float): Minimum score threshold.
        prefixes: Optional prefixes to filter document IDs.
        
        Returns:
        List[Tuple[str, float]]: A list of tuples (doc_id, hybrid_score) for the top chunks.
        """
        # Retrieve a large candidate pool by setting top_n high.
        # (You may adjust this number based on your collection size.)
        candidate_results = self.hybrid_search.advanced_search(query_text, keywords, top_n=1000, threshold=threshold, search_mode = search_mode, prefixes=prefixes)
        # print(f"candidate_results: {candidate_results}")
        
        if not candidate_results:
            self.logger.info("No candidate results found.")
            return []
        
        # Group the results by document.
        # Here we extract the document key from the doc_id (assumes doc_id starts with fileName)
        grouped_results = {}
        for doc_id, score in candidate_results:
            doc_key = doc_id.split('_')[0]  # Extract fileName
            grouped_results.setdefault(doc_key, []).append((doc_id, score))
        
        # print(f"grouped_results: {grouped_results}")
        
        # For each document group, sort the chunks by score (highest first) and take top per_doc_top_n
        final_results = []
        for doc_key, chunks in grouped_results.items():
            sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
            final_results.extend(sorted_chunks[:per_doc_top_n])
        
        # Optionally, sort across documents by score if you want a global ranking too.
        final_results = sorted(final_results, key=lambda x: x[1], reverse=True)
        
        return final_results
    
    
    def search(self, query_text: str, keywords: Optional[List[str]] = None,
              top_k: int = 5, threshold: float = 0.4, prefixes=None) -> List[Tuple[str, float]]:
        """Perform synchronous search."""
        try:
            return self.hybrid_search.advanced_search(
                query_text, keywords=keywords, top_n=top_k, threshold=threshold, prefixes=prefixes
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def search_async(self, query_text: str, keywords: Optional[List[str]] = None,
                         top_k: int = 5, threshold: float = 0.4, prefixes=None) -> List[Tuple[str, float]]:
        """Perform asynchronous search."""
        try:
            return await self.hybrid_search.advanced_search_async(
                query_text, keywords=keywords, top_n=top_k, threshold=threshold, prefixes=prefixes
            )
        except Exception as e:
            logger.error(f"Async search failed: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Retrieve a document by its ID."""
        return self.doc_manager.get_document(doc_id)
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the engine."""
        return self.doc_manager.get_document_count()
    
    def get_total_tokens(self) -> int:
        """Get the total token count across all documents."""
        return self.token_counter.get_total_tokens()
    
    def get_context(self) -> str:
        """Get the concatenated text of all retrieved documents."""
        return "\n".join(self.results)
    
    def save_state(self, path: str) -> bool:
        """Save the engine state to disk."""
        state_manager = StateManager()
        state_data = {
            "doc_ids": self.doc_manager.doc_ids,
            "documents": self.doc_manager.documents,
            "meta_data": self.doc_manager.meta_data,
            "token_counts": self.token_counter.doc_tokens,
            "embedding_model": self.embedding_model,
        }
        return state_manager.save_state(path, state_data)
    
    def load_state(self, path: str) -> bool:
        """Load the engine state from disk."""
        state_manager = StateManager()
        state_data = state_manager.load_state(path)
        
        if not state_data:
            logger.info("No previous state found, initializing fresh state.")
            return False
            
        try:
            # Restore document manager state
            self.doc_manager.doc_ids = state_data["doc_ids"]
            self.doc_manager.documents = state_data["documents"]
            self.doc_manager.meta_data = state_data["meta_data"]
            
            # Restore token counter
            self.token_counter.doc_tokens = state_data["token_counts"]
            self.token_counter.total_tokens = sum(self.token_counter.doc_tokens.values())
            
            # Check if embedding model matches
            stored_model = state_data.get("embedding_model")
            if stored_model and stored_model != self.embedding_model:
                logger.warning(f"Loaded state used embedding model '{stored_model}' "
                              f"but current instance uses '{self.embedding_model}'")
            
            # Rebuild search indices
            self.bm25_wrapper.clear_documents()
            self.faiss_wrapper.clear_documents()
            
            # Use batch operations for efficiency
            BATCH_SIZE = 100
            doc_batches = [
                list(zip(
                    self.doc_manager.doc_ids[i:i+BATCH_SIZE],
                    self.doc_manager.documents[i:i+BATCH_SIZE]
                ))
                for i in range(0, len(self.doc_manager.doc_ids), BATCH_SIZE)
            ]
            
            for batch in doc_batches:
                self.bm25_wrapper.add_documents(batch)
                self.faiss_wrapper.add_documents(batch)
            
            logger.info("System state loaded successfully with documents and indices rebuilt.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
            self._reset_state()
            return False
    
    def _reset_state(self) -> None:
        """Reset the engine to initial state."""
        self.doc_manager = DocumentManager()
        self.token_counter = TokenCounter()
        self.bm25_wrapper = BM25_search()
        self.faiss_wrapper = FAISS_search(self.embedding_model)
        self.hybrid_search = Hybrid_search(self.bm25_wrapper, self.faiss_wrapper)
        self.results = []