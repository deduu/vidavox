# Core components module (Retrieval_components.py)
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Union, Optional, Set, Sequence      
from collections.abc import Iterable                                       
from dataclasses import dataclass, asdict
from starlette.concurrency import run_in_threadpool
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

from vidavox.utils.pretty_logger import pretty_json_log
from vidavox.utils.doc_tracker import compute_checksum
from vidavox.utils.script_tracker import log_processing_time
from vidavox.retrieval.persistence_search import AsyncPersistence
from vidavox.schemas.common import DocItem


# Configure logging
logging.basicConfig()
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
            "page": result.meta_data.get('page', 'unknown')
        }

@dataclass
class Doc:
    doc_id: str
    text: str
    meta_data: Dict[str, any]

class DocumentManager:
    """
    Single-engine, multi-tenant document registry with shared documents.
    • All docs live in one dict -> O(1) lookup by id
    • user_to_doc_ids keeps a per-tenant index -> O(1) filtering
    • owner_id is injected into each doc.meta_data for persistence-layer filters
    • Documents without a specific user_id belong to all users (shared)
    """
    def __init__(self) -> None:
        self.documents: Dict[str, Doc] = {}
        self.user_to_doc_ids: Dict[str, Set[str]] = defaultdict(set)
        self.shared_doc_ids: Set[str] = set()  # Tracks documents that belong to all users
        self.lock = threading.RLock()
        self._just_added: Dict[Optional[str], List[str]] = defaultdict(list)

    # ---------- INGEST ----------
    def add_document(
        self,
        doc_id: str,
        text: str,
        meta_data: Optional[Dict] = None,
        user_id: Optional[str] = None,
    ) -> None:
        meta = (meta_data or {}).copy()
        if user_id is not None:
            meta["owner_id"] = user_id
        
        with self.lock:
            self.documents[doc_id] = Doc(doc_id, text, meta)
            if user_id is not None:
                self.user_to_doc_ids[user_id].add(doc_id)
            else:
                self.shared_doc_ids.add(doc_id)

            self._just_added[user_id].append(doc_id)

    def add_documents(
        self,
        docs: List[Tuple[str, str, Dict]],
        user_id: Optional[str] = None,
    ) -> None:
        with self.lock:
            # Prepare all documents at once
            new_docs = {}
            doc_ids = set()
            newly_added_ids = []
            
            for doc_id, text, meta_data in docs:
                meta = (meta_data or {}).copy()
                if user_id is not None:
                    meta["owner_id"] = user_id
                new_docs[doc_id] = Doc(doc_id, text, meta)
                # doc_ids.add(doc_id)
                newly_added_ids.append(doc_id)
            
            # Batch update documents dictionary
            self.documents.update(new_docs)
            
            # Associate documents with user or mark as shared
            if user_id is not None:
                self.user_to_doc_ids[user_id].update(newly_added_ids)
            else:
                self.shared_doc_ids.update(newly_added_ids)
            
            self._just_added[user_id].extend(newly_added_ids)

    # ---------- READ ----------
    def get_document(self, user_id: str, doc_id: str) -> Optional[str]:
        """
        Return text if this user owns the doc or if it's shared.
        """
        with self.lock:
            if doc_id not in self.user_to_doc_ids[user_id] and doc_id not in self.shared_doc_ids:
                return None
            return self.documents[doc_id].text

    def get_metadata(self, user_id: str, doc_id: str) -> Optional[Dict]:
        with self.lock:
            if doc_id not in self.user_to_doc_ids[user_id] and doc_id not in self.shared_doc_ids:
                return None
            return self.documents[doc_id].meta_data

    def get_user_docs(self, user_id: str) -> List[str]:
        """Fast O(#docs_user + #shared_docs) listing for query filters."""
        with self.lock:
            # Return both user-specific and shared documents
            user_docs = set(self.user_to_doc_ids.get(user_id, set()))
            return list(user_docs | self.shared_doc_ids)
    
    def get_all_documents(self) -> List[Doc]:
        """
        Return a list of every Doc in the registry (shared + per‐user).
        """
        with self.lock:
            # .values() is O(n) in number of docs
            return {doc_id: doc.text for doc_id, doc in self.documents.items()}
    
    def get_all_just_added_documents(
        self,
        user_id: Optional[str],
        clear_after: bool = True
    ) -> Dict[str, str]:
        """
        Return a map of doc_id -> text for everything that was
        added *since the last time you called this method* (for this user_id).
        If clear_after=True (default), empties the buffer.
        """
        with self.lock:
            just_ids = list(self._just_added.get(user_id, []))
            result = {doc_id: self.documents[doc_id].text for doc_id in just_ids}
            if clear_after:
                self._just_added[user_id].clear()
            return result

    


    def get_all_documents_by_user(self) -> Dict[Optional[str], List[Doc]]:
        """
        Return a mapping from user_id to that user’s Docs,
        plus key None for shared-only docs.
        """
        with self.lock:
            result: Dict[Optional[str], List[Doc]] = {}

            # per‐user docs
            for uid, doc_ids in self.user_to_doc_ids.items():
                result[uid] = [self.documents[d] for d in doc_ids]

            # shared docs (no owner)
            result[None] = [self.documents[d] for d in self.shared_doc_ids]

            return result

    # ---------- DELETE ----------
    def _delete_document_nolock(self, doc_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a document without locking."""
        # Check document exists
        if doc_id not in self.documents:
            return False
            
        # If user_id provided, check ownership
        if user_id is not None and doc_id not in self.user_to_doc_ids[user_id] and doc_id not in self.shared_doc_ids:
            return False
            
        # Remove from user's collection if it belongs to a user
        for uid, docs in self.user_to_doc_ids.items():
            if doc_id in docs:
                docs.remove(doc_id)
                
        # Remove from shared if it's shared
        self.shared_doc_ids.discard(doc_id)
        
        # Remove from main documents dictionary
        self.documents.pop(doc_id, None)
        return True
    
    def delete_document(self, doc_id: str, user_id: Optional[str] = None) -> bool:
        with self.lock:
            return self._delete_document_nolock(doc_id, user_id)

    def delete_documents(self, doc_ids: List[str], user_id: Optional[str] = None) -> int:
        """Batch delete multiple documents, returns count of deleted docs"""
        deleted_count = 0
        with self.lock:
            for doc_id in doc_ids:
                if self._delete_document_nolock(doc_id, user_id):
                    deleted_count += 1
            return deleted_count

    # ---------- UTIL ----------
    def clear_user(self, user_id: str) -> None:
        with self.lock:
            # Get user's document IDs
            doc_ids = self.user_to_doc_ids.pop(user_id, set())
            
            # Delete documents that belong only to this user
            # (not shared and not belonging to other users)
            for doc_id in doc_ids:
                is_used_elsewhere = any(doc_id in docs for uid, docs in self.user_to_doc_ids.items())
                if not is_used_elsewhere and doc_id not in self.shared_doc_ids:
                    self.documents.pop(doc_id, None)

    def doc_count(self, user_id: Optional[str] = None) -> int:
        with self.lock:
            if user_id is None:
                return len(self.documents)
            # Return count of user's documents plus shared documents
            return len(self.user_to_doc_ids.get(user_id, set())) + len(self.shared_doc_ids)

    def doc_ids(self, user_id: Optional[str] = None) -> List[str]:
        with self.lock:
            if user_id is None:
                return sorted(self.documents.keys())
            # Return both user-specific and shared documents
            return sorted(set(self.user_to_doc_ids.get(user_id, set())) | self.shared_doc_ids)


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
                "creation_time": time.ctime(stat.st_birthtime),
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
    """Handles persistence of Retrieval engine state."""
    
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


# Main Retrieval Engine module (Retrieval_engine.py)
import asyncio
from typing import List, Optional, Any, Dict, Tuple, Callable, Union

from vidavox.retrieval import BM25_search, FAISS_search, Hybrid_search, SearchMode

from vidavox.utils.token_counter import TokenCounter
from vidavox.document import DocumentSplitter, ProcessingConfig, DocumentNodes
from vidavox.document_store.models import EngineMetadata, Document  # Your ORM models
from vidavox.document_store.store import VectorStorePsql
from sqlalchemy.future import select
import datetime


# Helper functions for incremental index update
async def get_last_index_update(async_session) -> datetime.datetime:
    async with async_session() as session:
        result = await session.execute(
            select(EngineMetadata).filter(EngineMetadata.key == "last_index_update")
        )
        meta = result.scalar_one_or_none()
        if meta:
            return datetime.datetime.fromisoformat(meta.value.get("timestamp")).replace(tzinfo=datetime.timezone.utc)
        return datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)

async def update_last_index_update(async_session, new_time: datetime.datetime):
    async with async_session() as session:
        result = await session.execute(
            select(EngineMetadata).filter(EngineMetadata.key == "last_index_update")
        )
        meta = result.scalar_one_or_none()
        new_value = {"timestamp": new_time.isoformat()}
        if meta:
            meta.value = new_value
        else:
            from vidavox.document_store.models import EngineMetadata  # adjust import if needed
            meta = EngineMetadata(key="last_index_update", value=new_value)
            session.add(meta)
        await session.commit()

class Retrieval_Engine:
    """Main Retrieval Augmented Generation engine with modular components."""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', use_async: bool = False, show_docs: bool = False, vector_store:VectorStorePsql = None):
        """Initialize the Retrieval engine with all required components."""
        self.use_async = use_async
        self.show_docs = show_docs
        self.token_counter = TokenCounter()
        self.embedding_model = embedding_model
        
        # Initialize vector store
        self.vector_store = vector_store
        self.persistence = AsyncPersistence(self.vector_store) if vector_store else None

        # Initialize document store
        self.doc_manager = DocumentManager()

        # Initialize nodes
        self.nodes = None
        
        # Initialize search components
        self.bm25_wrapper = BM25_search()
        self.faiss_wrapper = FAISS_search(embedding_model)
        self.hybrid_search = Hybrid_search(self.bm25_wrapper, self.faiss_wrapper)

        # Results cache
        self.results = []
        self.toAdd = []

        # New: file → last seen mtime
        self._file_mtime_index: Dict[str, str] = {}



        # Batch lock
        self.batch_lock = threading.Lock()
    
    def _allowed_ids(self, user_id: str | None) -> list[str] | None:
        """Return the list of doc_ids this user may see (or None == no filter)."""
        if user_id is None:
            return None                     # admin / background tasks
        return self.doc_manager.get_user_docs(user_id)
    
    async def initialize_vector_store(self):
        """Initialize the vector store."""
        if self.vector_store:
            await self.vector_store.initialize()
    
    def _process_csv_as_dataframe(
        self, 
        file_path: str, 
        text_col: str,
        metadata_cols: List[str],
        existing_docs: Dict[str, Document]
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
                # doc_id = f"{file_name}_{idx}"
                file_processing_uuid = str(uuid.uuid4())
                doc_id = f"{file_name}_{file_processing_uuid}_chunk{idx}"
                doc_text = str(row[text_col])
                doc_checksum = compute_checksum(doc_text)
                
                # Create metadata dictionary
                doc_metadata = {
                    'source': file_path,
                    'file_name': file_name,
                    'row_index': idx,
                    'file_type': 'csv',
                    'checksum': doc_checksum
                }
                
                # Add specified metadata columns
                for col in metadata_cols:
                    if col in df.columns:
                        doc_metadata[col] = row[col]
                
                existing_doc = existing_docs.get(doc_id)
                if existing_doc and existing_doc.meta_data.get("checksum") == doc_checksum:
                    continue

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
        metadata_cols: List[str],
        existing_docs: Dict[str, Document]
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
                file_processing_uuid = str(uuid.uuid4())
                doc_id = f"{file_name}_{file_processing_uuid}_chunk{idx}"
                doc_text = str(row[text_col])
                doc_checksum = compute_checksum(doc_text)

                # Create metadata dictionary
                doc_metadata = {
                    'source': file_path,
                    'file_name': file_name,
                    'row_index': idx,
                    'file_type': 'excel',
                    'checksum': doc_checksum,
                }

                # Add specified metadata columns
                for col in metadata_cols:
                    if col in df.columns:
                        doc_metadata[col] = row[col]

                # Skip if the document exists and checksum is unchanged.
                existing_doc = existing_docs.get(doc_id)
                if existing_doc and existing_doc.meta_data.get("checksum") == doc_checksum:
                    continue
                batch_docs.append((doc_id, doc_text, doc_metadata))

            logger.info(f"Successfully processed {len(df)} rows from Excel file {file_name}")
            return batch_docs

        except Exception as e:
            logger.error(f"Failed to process Excel file {file_path}: {e}")
            return []


    def _process_file(
        self,
        file_path: str,
        config: ProcessingConfig,
        chunker: Optional[Callable] = None,
        use_recursive: bool = True,
        *,                             # <‑‑ makes the next arg keyword‑only
        file_db_id: str | None = None  # <‑‑ NEW
        ) -> List[Tuple[str, str, Dict]]:
        processor = FileProcessor()
        file_name = os.path.basename(file_path)

        # 1) current file mtime
        base_meta     = processor.get_file_metadata(Path(file_path))
        current_mtime = base_meta.get("modification_time")

        # 2) O(1) check
        if self._file_mtime_index.get(file_name) == current_mtime:
            logger.info(f"{file_name} unchanged (mtime={current_mtime}); skipping.")
            return []

        batch_docs: List[Tuple[str, str, Dict]] = []
        try:
            nodes = DocumentSplitter(config, use_recursive=use_recursive).run(file_path, chunker)
            self.nodes = nodes

            for idx, doc in enumerate(nodes):
                file_processing_uuid = str(uuid.uuid4())
                if file_db_id is not None:
                    doc_id = f"{file_db_id}_{file_name}_chunk{idx}"
                else:
                    doc_id = f"{file_name}_{file_processing_uuid}_chunk{idx}"

                meta   = dict(base_meta)  # copy
                meta["file_type"] = Path(file_path).suffix.lower().lstrip(".")
                combined_meta     = {**meta, **doc.metadata}
                batch_docs.append((doc_id, doc.page_content, combined_meta))

                if self.show_docs:
                    pretty_json_log(logger, batch_docs, "batch_docs:")

            logger.info(f"Successfully processed {len(nodes)} chunks from {file_name}")
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")

        return batch_docs

    @log_processing_time
    def from_paths(
        self,
        sources: Sequence[Union[str, 'DocItem']],    
        *,
        config: Optional[ProcessingConfig] = None,
        chunker: Optional[Callable] = None,
        show_progress: bool = False,
        load_csv_as_pandas_dataframe: Optional[bool] = False,
        load_excel_as_pandas_dataframe: bool = False,
        text_col: Optional[str] = None,
        metadata_cols: Optional[List[str]] = None,
        use_recursive:bool=True,
        user_id: Optional[str] = "User A"
    ) -> 'Retrieval_Engine':
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

        # Use existing documents to perform incremental indexing
        existing_docs = self.doc_manager.documents
        # Configure progress tracking
        iterator: Iterable = tqdm(sources, desc="Processing files", unit="file") if show_progress else sources
        iterator = tqdm(sources, desc="Processing files", unit="file") if show_progress else sources
        
        # Collect processing statistics
        stats = {
            'csv_files_processed': 0,
            'excel_files_processed': 0,
            'other_files_processed': 0,
            'failed_files': 0
        }
        
        # Process files
        for doc in iterator:
                 # normalise the input ----------------------------------------------
            if isinstance(doc, DocItem):
                file_path: str = doc.path
                file_db_id: Optional[str] = doc.db_id
            elif isinstance(doc, str):
                file_path = doc
                file_db_id = None          # path supplied without DB id
            else:                           # defensive: unsupported element
                raise TypeError(
                    f"Each element in 'sources' must be a str or DocItem, got {type(doc)}"
                )

        # ------------------------------------------------------------------
            try:
                is_csv = file_path.lower().endswith('.csv')
                is_excel = file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls')
                
                # 1) CSV processing
                if is_csv and load_csv_as_pandas_dataframe:
                    file_docs = self._process_csv_as_dataframe(
                        file_path, 
                        text_col=text_col,
                        metadata_cols=metadata_cols,
                        existing_docs = existing_docs
                    )
                    # print(f"file_docs: {file_docs}")
                    if file_docs:
                        stats['csv_files_processed'] += 1

                # 2) Excel processing
                elif is_excel and load_excel_as_pandas_dataframe:
                    file_docs = self._process_excel_as_dataframe(
                        file_path,
                        text_col=text_col,
                        metadata_cols=metadata_cols,
                        existing_docs = existing_docs
                    )
                    if file_docs:
                        stats['excel_files_processed'] += 1

                # 3) Other file types
                else:
                    file_docs = self._process_file(file_path, config, chunker, use_recursive, file_db_id=file_db_id)
                    if file_docs:
                        stats['other_files_processed'] += 1
                
                if not file_docs:
                    stats['failed_files'] += 1
                    continue
                    
                batch_docs.extend(file_docs)
               
                
                # Process batch if it reaches threshold
                if len(batch_docs) >= BATCH_SIZE:
                    self._process_document_batch(batch_docs, user_id=user_id)
                    batch_docs = []
                    
            except Exception as e:
                stats['failed_files'] += 1
                logger.error(f"Failed to process file {file_path}: {e}")
                continue

        # Process any remaining documents
        if batch_docs:
            self._process_document_batch(batch_docs, user_id=user_id)
            
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
        metadata_cols: Optional[List[str]] = None,
        use_recursive:bool=True
    ) -> 'Retrieval_Engine':
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
        
        # Wrap plain file paths into DocItem instances
        doc_items = [DocItem(path=fp, db_id=None) for fp in file_paths]

        
        logger.info(f"Found {len(file_paths)} files in {directory}.")
        return self.from_paths(
            doc_items,
            config=config or ProcessingConfig(),
            chunker=chunker,
            show_progress=show_progress,
            load_csv_as_pandas_dataframe=load_csv_as_pandas_dataframe,
            load_excel_as_pandas_dataframe=load_excel_as_pandas_dataframe,
            text_col=text_col,
            metadata_cols=metadata_cols,
            use_recursive=use_recursive
        )
  

    @log_processing_time
    def _process_document_batch(self, docs: List[Tuple[str, str, Dict]], user_id: Optional[str] = "User A",) -> None:
        """Process a batch of documents efficiently."""
        try:

            with self.batch_lock:
                # Extract components for batch processing
                doc_ids, texts, meta_datas = zip(*docs)
            
                
                # Update document manager
                self.doc_manager.add_documents(docs, user_id=user_id)
                
                # Update token counter
                for doc_id, text in zip(doc_ids, texts):
                    self.token_counter.add_document(doc_id, text, user_id=user_id)
     
                    # --------------- CPU‑bound indexing off the event loop -----------
 
                async def _index():
                    # BM25 is pure‑python but fast; FAISS is the expensive bit.
                    self.bm25_wrapper.add_documents(list(zip(doc_ids, texts)))
                    inserted = self.faiss_wrapper.add_documents(
                        list(zip(doc_ids, texts)), return_vectors=True
                    )
                    return inserted

                inserted_vectors = asyncio.run(
                    run_in_threadpool(_index)        # one thread‑pool hop
                )


                logger.info(f"Added {len(doc_ids)} documents to the engine for {user_id}")
                if self.persistence:
                    self.persistence.queue_docs(docs)
                    logger.info(f"Persisted {len(doc_ids)} documents to the persistence store.")
                    if inserted_vectors:                       # never None when return_vectors=True
                        self.persistence.queue_vectors(inserted_vectors)
                        logger.info(f"Persisted {len(inserted_vectors)} FAISS vectors to the persistence store.")
                    self.persistence.queue_bm25(self.bm25_wrapper.get_multiple_doc_terms(doc_ids))
                    self.persistence.queue_token_counts([
                        (d, self.token_counter.get_doc_tokens(d)) for d in doc_ids
                ])
            

                _, _, first_meta = docs[0]
                fn    = first_meta.get("file_name")
                mtime = first_meta.get("modification_time")
                if fn and mtime:
                    self._file_mtime_index[fn] = mtime
                


        except Exception as e:
            logger.error(f"Failed to process document batch: {e}")
            raise
    
    def add_document(self, doc_id: str, text: str, meta_data: Optional[Dict] = None, user_id: Optional[str] = None) -> None:
        """Add a single document to the engine."""
        try:
            # Update document manager
            self.doc_manager.add_document(doc_id, text, meta_data, user_id)
            self.token_counter.add_document(doc_id, text, user_id=user_id) 
            
            # Update search indices
            self.bm25_wrapper.add_document(doc_id, text)
            self.faiss_wrapper.add_document(doc_id, text)
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            raise
    
    async def add_document_async(self, doc_id: str, text: str, meta_data: Optional[Dict]=None):
        # document manager is thread-safe, so we can call it directly
        self.doc_manager.add_document(doc_id, text, meta_data)
        self.token_counter.add_document(doc_id, text, user_id=meta_data.get("user_id"))

        # assume your wrappers support async; if not, wrap in to_thread
        await self.bm25_wrapper.add_document_async(doc_id, text)
        await self.faiss_wrapper.add_document_async(doc_id, text)

        # queue for persistence
        if self.persistence:
            self.persistence.queue_docs([(doc_id, text, meta_data or {})])
            # and then maybe immediately flush
            await self.persistence.flush_async()
    
    def add_documents(self, docs: List[Tuple[str, str, Optional[Dict]]]) -> None:
        """Add multiple documents to the engine efficiently."""
        self._process_document_batch(docs)

    async def add_documents_async(self, docs: List[Tuple[str, str, Dict]]):
        # mirror your sync batch logic, but with awaits
        self.doc_manager.add_documents(docs)
        for doc_id, text, _ in docs:
            self.token_counter.add_document(doc_id, text)
        await self.bm25_wrapper.add_documents_async([(d, t) for d, t, _ in docs])
        vecs = await self.faiss_wrapper.add_documents_async([(d, t) for d, t, _ in docs], return_vectors=True)

        # queue and flush
        if self.persistence:
            self.persistence.queue_docs(docs)
            self.persistence.queue_vectors(vecs)
            await self.persistence.flush_async()
    
    def delete_document(self, doc_id: str, user_id: str = None) -> bool:
        """Remove a document from the engine."""
        try:
            # Ensure the doc exists in the manager
            if doc_id not in self.doc_manager.get_user_docs(user_id):
                logger.warning(f"Document ID {doc_id} not found in user {user_id}.")
                return False
            
            # Remove from search indices (by doc_id, not index)
            self.bm25_wrapper.remove_document(doc_id)
            self.faiss_wrapper.remove_document(doc_id)
            
            # Remove from token counter
            self.token_counter.remove_document(doc_id)
            
            # Remove from document manager
            self.doc_manager.delete_document(doc_id, user_id)
            
            return True

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

        
    async def delete_document_async(self, doc_id: str) -> bool:
        if doc_id not in self.doc_manager.doc_ids:
            return False
        idx = self.doc_manager.doc_ids.index(doc_id)

        # remove from search backends
        await self.bm25_wrapper.remove_document_async(idx)
        await self.faiss_wrapper.remove_document_async(idx)
        self.token_counter.remove_document(doc_id)
        self.doc_manager.delete_document(doc_id)
        # persistence might also need 
        if self.persistence:
            await self.persistence.flush_async()
        return True

    
    async def clear_async(self, user_id: Optional[str] = None):
        if user_id is None:
            # implement a full clear if you need it:
            # either add doc_manager.clear_all() or:
            self.doc_manager = DocumentManager()
        else:
            self.doc_manager.clear_user(user_id)

        # reset other components
        self.token_counter = TokenCounter()
        await self.bm25_wrapper.clear_documents_async()
        await self.faiss_wrapper.clear_documents_async()
        if self.persistence:
            await self.persistence.clear_async()
    
    def query(self, query_text: str, keywords: Optional[List[str]] = None,
         threshold: float = 0.4, top_k: int = 5,
         result_formatter: Optional[BaseResultFormatter] = None,
         prefixes: Optional[List[str]] = None,
         include_doc_ids: Optional[List[str]] = None,
         exclude_doc_ids: Optional[List[str]] = None) -> List[Dict]:
        """Query the engine for relevant documents."""
        if self.use_async:
            return asyncio.run(self.retrieve_async(query_text, keywords, threshold, top_k, result_formatter=result_formatter, prefixes=prefixes, include_doc_ids=include_doc_ids, exclude_doc_ids=exclude_doc_ids))
        else:
            return self.retrieve(query_text, keywords, threshold, top_k, result_formatter=result_formatter, prefixes=prefixes, include_doc_ids=include_doc_ids, exclude_doc_ids=exclude_doc_ids)
    
    def retrieve(self, query_text: str, keywords: Optional[List[str]] = None,
                threshold: float = 0.4, top_k: int = 5, result_formatter: Optional[BaseResultFormatter] = None,
                prefixes=None, include_doc_ids=None, exclude_doc_ids=None,  max_results_size: int = 1000, user_id: Optional[str] = None) -> List[Dict]:
        """Synchronously retrieve relevant documents."""
         # Apply reasonable limits to prevent resource exhaustion
        capped_top_k = min(top_k, max_results_size)
        results = self.search(query_text, keywords, capped_top_k, threshold, prefixes, include_doc_ids, exclude_doc_ids, user_id=user_id)
        return self._process_search_results(results)
    
    async def retrieve_async(self, query_text: str, keywords: Optional[List[str]] = None,
                           threshold: float = 0.4, top_k: int = 5, result_formatter: Optional[BaseResultFormatter] = None,
                           prefixes=None, include_doc_ids=None, exclude_doc_ids=None, max_results_size: int = 1000, user_id: Optional[str] = None) -> List[Dict]:
        """Asynchronously retrieve relevant documents."""
        # Apply reasonable limits to prevent resource exhaustion
        capped_top_k = min(top_k, max_results_size)
        results = await self.search_async(query_text, keywords,capped_top_k, threshold, prefixes, include_doc_ids=include_doc_ids, exclude_doc_ids=exclude_doc_ids, user_id=user_id)
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
        sort_globally: Optional[bool] = False,
        include_doc_ids: Optional[List[str]] = None,
        exclude_doc_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None
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
        if user_id is not None:
            include_doc_ids = self._allowed_ids(user_id)
        # Retrieve candidate chunks grouped by document.
        candidate_results = self.search_best_chunk_per_document(
            query_text, keywords, per_doc_top_n=per_doc_top_n,
            threshold=threshold, prefixes=prefixes, search_mode=search_mode,
            sort_globally=sort_globally, 
            include_doc_ids=include_doc_ids,
            exclude_doc_ids=exclude_doc_ids
        )
        
        # Process the candidate results into a standard format.
        return self._process_search_results(candidate_results, result_formatter=result_formatter)

    
    def _process_search_results(
        self,
        results: List[Tuple[str, float]],
        result_formatter: Optional[BaseResultFormatter] = None
    ) -> List[Dict]:
        if not results:
            return [{"id": "None.", "url": "None.", "text": None}]

        formatter = result_formatter or DefaultResultFormatter()
        seen, output = set(), []

        # process in batches if needed
        for doc_id, score in results:
            if doc_id in seen or doc_id not in self.doc_manager.documents:
                continue
            seen.add(doc_id)
            doc_obj = self.doc_manager.documents[doc_id]
          
            sr = SearchResult(doc_id, doc_obj.text, doc_obj.meta_data, score)
            # logger.info(f"\n\n\n")
            # logger.info(f"sr: {sr}")
            output.append(formatter.format(sr))
            # logger.info(f"\n\n\n")
        # logger.info(f"Search result: {output}")
        return output or [{"id": "None.", "url": "None.", "text": None}]
    

    def search_best_chunk_per_document(self, query_text, keywords, per_doc_top_n=5, threshold=0.53, prefixes=None, 
                                       search_mode: Optional[SearchMode] = SearchMode.HYBRID, sort_globally: Optional[bool]= False, 
                                       include_doc_ids=None, exclude_doc_ids=None, user_id: Optional[str] = None):
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
        if user_id is not None:
            include_doc_ids = self._allowed_ids(user_id)
            
        candidate_results = self.hybrid_search.advanced_search(query_text, keywords, top_n=1000, threshold=threshold, search_mode = search_mode, prefixes=prefixes, include_doc_ids=include_doc_ids, exclude_doc_ids=exclude_doc_ids)
   
        
        if not candidate_results:
            logger.info("No candidate results found.")
            return []
        
        # Group the results by document.
        # Here we extract the document key from the doc_id (assumes doc_id starts with fileName)
        grouped_results = {}
        for doc_id, score in candidate_results:
            doc_key = doc_id.split('_')[0]  # Extract fileName
            grouped_results.setdefault(doc_key, []).append((doc_id, score))
        
      
        
        # For each document group, sort the chunks by score (highest first) and take top per_doc_top_n
        final_results = []
        for doc_key, chunks in grouped_results.items():
            sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
            final_results.extend(sorted_chunks[:per_doc_top_n])
        
        # Optionally, sort across documents by score if you want a global ranking too.
        final_results = sorted(final_results, key=lambda x: x[1], reverse=True)
        
        return final_results
    
    
    def search(self, query_text: str, keywords: Optional[List[str]] = None,
              top_k: int = 5, threshold: float = 0.4, prefixes=None, include_doc_ids=None, exclude_doc_ids=None, user_id: Optional[str] = None) -> List[Tuple[str, float]]:
        """Perform synchronous search."""
        if user_id:
            include_doc_ids = self._allowed_ids(user_id)
            logger.info(f"Include doc_ids: {include_doc_ids} for user {user_id}")
        try:
            return self.hybrid_search.advanced_search(
                query_text,
                keywords           = keywords,
                top_n              = top_k,
                threshold          = threshold,
                prefixes           = prefixes,
                include_doc_ids    = include_doc_ids,
                exclude_doc_ids    = exclude_doc_ids
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def search_async(self, query_text: str, keywords: Optional[List[str]] = None,
                         top_k: int = 5, threshold: float = 0.4, prefixes=None, include_doc_ids=None, exclude_doc_ids=None, user_id: Optional[str] = None) -> List[Tuple[str, float]]:
        """Perform asynchronous search."""
        if user_id is not None:
            include_doc_ids = self._allowed_ids(user_id)
        try:
            return await self.hybrid_search.advanced_search_async(
                query_text, keywords=keywords, top_n=top_k, threshold=threshold, prefixes=prefixes, include_doc_ids=include_doc_ids, exclude_doc_ids=exclude_doc_ids
            )
        except Exception as e:
            logger.error(f"Async search failed: {e}")
            return []
    
    def get_nodes(self) -> Optional[DocumentNodes]:
        """Retrieve the nodes."""
        return self.nodes
    
    # Match get_document signature: pass user_id first
    def get_document(self, user_id: str, doc_id: str) -> Optional[str]:
        return self.doc_manager.get_document(user_id, doc_id)

    # Same for count
    def get_document_count(self, user_id: Optional[str] = None) -> int:
        return self.doc_manager.doc_count(user_id)
    
    def get_total_tokens(self) -> int:
        """Get the total token count across all documents."""
        return self.token_counter.get_total_tokens()
    
    def get_user_total_tokens(self, user_id: str) -> int:
        return self.token_counter.get_user_total_tokens(user_id)

    def get_context(self) -> str:
        """Get the concatenated text of all retrieved documents."""
        return "\n".join(self.results)
    
    def save_state(self, path: str) -> bool:
        """
        Save engine state to disk, including each doc’s owner_id in its metadata.
        """
        state_manager = StateManager()
        state_data = {
            "embedding_model": self.embedding_model,
            "documents": {
                doc_id: (doc.text, doc.meta_data)
                for doc_id, doc in self.doc_manager.documents.items()
            },
            "token_snapshot": self.token_counter.snapshot().__dict__,
        }
        return state_manager.save_state(path, state_data)

    def load_state(self, path: str, verbose: bool = False) -> bool:
        """
        Load engine state, restoring both shared docs and per-user docs.
        Uses meta["owner_id"] to re-populate DocumentManager.user_to_doc_ids
        and shared_doc_ids correctly.
        """
        state_manager = StateManager()
        state_data = state_manager.load_state(path)
        
        if verbose:
            pretty_json_log(logger, state_data, "state_data:")

        if not state_data:
            logger.info("No previous state found, initializing fresh state.")
            return False

        try:
            # 1) Fully reset the document manager so its indices start empty
            self.doc_manager = DocumentManager()
            

            # 2) Re-add each doc with its original owner_id (if any)
            for doc_id, (text, meta) in state_data["documents"].items():
                owner = meta.get("owner_id")  # None ⇒ shared
                self.token_counter.add_document(doc_id, text, user_id=owner)
                self.doc_manager.add_document(
                    doc_id=doc_id,
                    text=text,
                    meta_data=meta,
                    user_id=owner
                )

            # 3) Restore token counts
            from vidavox.utils.token_counter import TokenCounter, TokenSnapshot    # import
            snap_dict = state_data.get("token_snapshot", {})
            if snap_dict:
                snap = TokenSnapshot(**snap_dict)
                self.token_counter.total_tokens      = snap.total_tokens
                self.token_counter.doc_tokens        = snap.doc_tokens
                self.token_counter.user_total_tokens = snap.user_total_tokens
                self.token_counter.user_doc_tokens   = snap.user_doc_tokens

            # 4) Warn if the embedding_model has changed
            stored_model = state_data.get("embedding_model")
            if stored_model and stored_model != self.embedding_model:
                logger.warning(
                    "State was saved with model %s, current engine uses %s",
                    stored_model, self.embedding_model,
                )

            # 5) Rebuild indices in one batch
            batch = [(doc_id, text) for doc_id, (text, _) in state_data["documents"].items()]
            self.bm25_wrapper.clear_documents()
            self.faiss_wrapper.clear_documents()
            if batch:
                # self.bm25_wrapper.add_documents(batch)
                # self.faiss_wrapper.add_documents(batch)
                async def _index():
                    # BM25 is pure‑python but fast; FAISS is the expensive bit.
                    self.bm25_wrapper.add_documents(batch)
                    inserted = self.faiss_wrapper.add_documents(
                        batch, return_vectors=True
                    )
                    return inserted

                inserted_vectors = asyncio.run(
                    run_in_threadpool(_index)        # one thread‑pool hop
                )

            logger.info(
                "Loaded state: %d documents rebuilt into BM25 & FAISS",
                len(batch),
            )
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