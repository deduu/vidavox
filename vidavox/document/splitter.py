import os
import asyncio
import logging
from tqdm import tqdm
from typing import List, Optional, Callable, Dict
from langchain.docstore.document import Document
from .config import ProcessingConfig, SplitterConfig
from .loader import LoaderFactory
from .node import DocumentNodes

from vidavox.utils.script_tracker import log_processing_time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from vidavox.utils.pretty_logger import pretty_json_log

class DocumentSplitter:
    def __init__(self, config: Optional[ProcessingConfig] = None, show_progress: bool = False, use_recursive: bool = True):
        # Use the provided ProcessingConfig if given; otherwise, use the default.
        if config is None:
            config = ProcessingConfig()
        self.config = config
        # Start with default splitter configs and update if custom ones are provided.
        default_configs = self.config.get_default_splitter_configs()
        if config.splitter_configs:
            default_configs.update(config.splitter_configs)
        self.splitter_configs = default_configs
        self.show_progress = show_progress
        self.use_recursive = use_recursive

    def get_splitter_config(self) -> Dict[str, SplitterConfig]:
        print(self.splitter_configs)
        return self.splitter_configs

    @log_processing_time
    def split_documents(self, documents: List[Document]) -> List[Document]:
        split_docs = []
        # Wrap documents in tqdm if show_progress is True
        doc_iterator = tqdm(documents, desc="Splitting documents") if self.show_progress else documents

        for doc in doc_iterator:
            pretty_json_log(logger, f"metadata: {doc.metadata}")
            file_extension = os.path.splitext(doc.metadata["source"])[1]  # e.g., ".pdf"
            pretty_json_log(logger, file_extension)

            splitter_config = self.splitter_configs.get(file_extension, self.splitter_configs["default"])
            # pretty_json_log(logger, f"splitter_config: {splitter_config}")
            
            try:
                splitter = splitter_config.splitter_class(**splitter_config.params)
                # print(f"splitter chunk_overlap: {splitter._chunk_overlap}")
                # print(f"splitter chunk_size: {splitter._chunk_size}")
                chunks = splitter.split_text(doc.page_content)
                
                for chunk in chunks:
                    if isinstance(chunk, str):
                        split_docs.append(Document(
                            page_content=chunk,
                            metadata=doc.metadata.copy()
                        ))
                    elif isinstance(chunk, Document):
                        chunk.metadata.update(doc.metadata)
                        split_docs.append(chunk)
            except Exception as e:
                raise ValueError(f"Error splitting document: {str(e)}")
                
        return split_docs

    @log_processing_time
    def run(
        self,
        file_path: str,
        custom_chunker: Optional[Callable[[Document], List[Document]]] = None,
        use_async: bool = False
    ) -> DocumentNodes:
        """
        Process a document file with optional custom chunking strategy.
        
        Args:
            file_path: Path to the document file
            custom_chunker: Optional function to implement custom chunking logic
            use_async: Whether to use async loading (default: False for backward compatibility)
            
        Returns:
            DocumentNodes containing processed document chunks
        """
        if use_async:
            # If async is requested, run the async version
            return asyncio.run(self.arun(file_path, custom_chunker))
        
        try:
            # Get file extension and appropriate loader
            logger.info(f"File path: {file_path}")
            _, file_extension = os.path.splitext(file_path)
            loader = LoaderFactory.get_loader(file_extension)
            logger.info(f"Loader: {loader}")
            
            # Load document synchronously
            documents = loader.load(file_path)
            logger.info(f"Loaded document: {file_path}")
            
            # Apply chunking strategy
            split_docs = self._process_documents(documents, custom_chunker)
                
            logger.info(f"Document split into {len(split_docs)} chunks")
            return DocumentNodes(split_docs)
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    @log_processing_time
    async def arun(
        self,
        file_path: str,
        custom_chunker: Optional[Callable[[Document], List[Document]]] = None
    ) -> DocumentNodes:
        """
        Asynchronously process a document file with optional custom chunking strategy.
        
        Args:
            file_path: Path to the document file
            custom_chunker: Optional function to implement custom chunking logic
            
        Returns:
            DocumentNodes containing processed document chunks
        """
        try:
            # Get file extension and appropriate loader
            logger.info(f"File path: {file_path}")
            _, file_extension = os.path.splitext(file_path)
            loader = LoaderFactory.get_loader(file_extension)
            logger.info(f"Loader: {loader}")
            
            # Load document asynchronously
            documents = await loader.aload(file_path)
            logger.info(f"Loaded document: {file_path}")
            
            # Apply chunking strategy (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            split_docs = await loop.run_in_executor(
                None, 
                self._process_documents, 
                documents, 
                custom_chunker
            )
                
            logger.info(f"Document split into {len(split_docs)} chunks")
            return DocumentNodes(split_docs)
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    def _process_documents(
        self, 
        documents: List[Document], 
        custom_chunker: Optional[Callable[[Document], List[Document]]] = None
    ) -> List[Document]:
        """
        Internal method to process documents with chunking strategy.
        Extracted to avoid code duplication between sync and async methods.
        """
        if custom_chunker:
            logger.info("Using custom chunker")
            split_docs = []
            doc_iterator = tqdm(documents, desc="Custom chunking") if self.show_progress else documents
            for doc in doc_iterator:
                chunks = custom_chunker(doc)
                split_docs.extend(chunks)
        elif not self.use_recursive:
            split_docs = documents
        else:
            split_docs = self.split_documents(documents)
        
        return split_docs