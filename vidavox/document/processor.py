import os
import logging
from typing import List, Optional, Callable
from langchain.docstore.document import Document

from .config import ProcessingConfig
from .loader import LoaderFactory
from .splitter import DocumentSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.splitter = DocumentSplitter(config)

    def process_file(
        self,
        file_path: str,
        custom_chunker: Optional[Callable[[Document], List[Document]]] = None
    ) -> List[Document]:
        """
        Process a document file with optional custom chunking strategy.
        
        Args:
            file_path: Path to the document file
            custom_chunker: Optional function to implement custom chunking logic
            
        Returns:
            List of processed document chunks
        """
        try:
            # Get file extension and appropriate loader
            _, file_extension = os.path.splitext(file_path)
            loader = LoaderFactory.get_loader(file_extension)
            
            # Load document
            documents = loader.load(file_path)
            logger.info(f"Loaded document: {file_path}")
            
            # Apply chunking strategy
            if custom_chunker:
                logger.info(f"Using custom chunker for {file_path}")
                split_docs = []
                for doc in documents:
                    chunks = custom_chunker(doc)
                    split_docs.extend(chunks)
            else:
                split_docs = self.splitter.split_documents(documents)
                
            logger.info(f"Document split into {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise