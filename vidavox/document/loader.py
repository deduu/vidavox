# loaders.py
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Type, Optional
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)

logger = logging.getLogger(__name__)


class BaseDocumentLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """Synchronously load documents from file."""
        pass
    
    async def aload(self, file_path: str) -> List[Document]:
        """Asynchronously load documents from file."""
        # Default implementation runs sync load in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load, file_path)
    
    def count_pages(self, file_path: str) -> int:
        """
        Default "page" count = number of Document objects returned.
        Subclasses can override for more accurate results.
        """
        docs = self.load(file_path)
        return len(docs)


class PDFLoader(BaseDocumentLoader):
    def __init__(self, prefer_pymupdf: bool = False):
        """
        Initialize PDF loader.
        
        Args:
            prefer_pymupdf: If True, try PyMuPDFLoader first, otherwise try PyPDFLoader first
        """
        self.prefer_pymupdf = prefer_pymupdf
    
    def load(self, file_path: str) -> List[Document]:
        """Load PDF with fallback between PyPDFLoader and PyMuPDFLoader."""
        loaders = [
            (PyMuPDFLoader, {"extract_images": True}),
            (PyPDFLoader, {"extract_images": True})  # Disable images to avoid reshape errors
        ]
        
        # Swap order if PyPDF is preferred
        if not self.prefer_pymupdf:
            loaders = loaders[::-1]
        
        last_error = None
        for loader_class, kwargs in loaders:
            try:
                logger.debug(f"Attempting to load PDF with {loader_class.__name__}")
                return loader_class(file_path, **kwargs).load()
            except Exception as e:
                logger.warning(f"{loader_class.__name__} failed: {str(e)}")
                last_error = e
                continue
        
        # If both fail, raise the last error
        raise RuntimeError(f"All PDF loaders failed. Last error: {last_error}")
    
    async def aload(self, file_path: str) -> List[Document]:
        """Asynchronously load PDF with fallback mechanism."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load, file_path)
    
    def count_pages(self, file_path: str) -> int:
        """Get accurate PDF page count using PyPDF2."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            return len(reader.pages)
        except Exception as e:
            logger.warning(f"Failed to count PDF pages with PyPDF2: {e}")
            # Fallback to document count
            return super().count_pages(file_path)


class DocxLoader(BaseDocumentLoader):
    def load(self, file_path: str) -> List[Document]:
        return Docx2txtLoader(file_path).load()


class ExcelLoader(BaseDocumentLoader):
    def load(self, file_path: str) -> List[Document]:
        return UnstructuredExcelLoader(file_path).load()


class CSVLoader(BaseDocumentLoader):
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
    
    def load(self, file_path: str) -> List[Document]:
        return CSVLoader(file_path, encoding=self.encoding).load()


class TextFileLoader(BaseDocumentLoader):
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
    
    def load(self, file_path: str) -> List[Document]:
        try:
            return TextLoader(file_path, encoding=self.encoding).load()
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for {file_path}, trying with 'latin1'")
            return TextLoader(file_path, encoding='latin1').load()


class MarkdownLoader(BaseDocumentLoader):
    def load(self, file_path: str) -> List[Document]:
        return UnstructuredMarkdownLoader(file_path).load()


class AsyncCapableLoader(BaseDocumentLoader):
    """Base class for loaders that support native async operations."""
    
    def __init__(self, base_loader_class, *args, **kwargs):
        self.base_loader_class = base_loader_class
        self.args = args
        self.kwargs = kwargs
    
    def load(self, file_path: str) -> List[Document]:
        loader = self.base_loader_class(file_path, *self.args, **self.kwargs)
        return loader.load()
    
    async def aload(self, file_path: str) -> List[Document]:
        loader = self.base_loader_class(file_path, *self.args, **self.kwargs)
        
        # Check if the loader has native async support
        if hasattr(loader, 'aload'):
            return await loader.aload()
        
        # Fallback to thread pool execution
        return await super().aload(file_path)


class LoaderFactory:
    """Factory for creating document loaders with support for different file types."""
    
    _loaders = {
        '.pdf': PDFLoader,
        '.docx': DocxLoader,
        '.xlsx': ExcelLoader,
        '.xls': ExcelLoader,  # Added support for older Excel format
        '.csv': CSVLoader,
        '.txt': TextFileLoader,
        '.md': MarkdownLoader,
        '.markdown': MarkdownLoader,  # Alternative markdown extension
    }
    
    @classmethod
    def get_loader(cls, file_extension: str, **kwargs) -> BaseDocumentLoader:
        """
        Get a loader for the specified file extension.
        
        Args:
            file_extension: File extension (e.g., '.pdf', '.docx')
            **kwargs: Additional arguments to pass to the loader constructor
            
        Returns:
            BaseDocumentLoader instance
            
        Raises:
            ValueError: If the file format is not supported
        """
        loader_class = cls._loaders.get(file_extension.lower())
        if not loader_class:
            supported_formats = ', '.join(cls._loaders.keys())
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {supported_formats}"
            )
        
        # Pass kwargs to loader constructor if it accepts them
        try:
            return loader_class(**kwargs)
        except TypeError:
            # If the loader doesn't accept kwargs, create with no args
            logger.debug(f"Loader {loader_class.__name__} doesn't accept kwargs, creating with defaults")
            return loader_class()
    
    @classmethod
    def register_loader(cls, extension: str, loader_class: Type[BaseDocumentLoader]):
        """Register a new loader for a file extension."""
        if not issubclass(loader_class, BaseDocumentLoader):
            raise TypeError("loader_class must be a subclass of BaseDocumentLoader")
        cls._loaders[extension.lower()] = loader_class
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return list(cls._loaders.keys())
    
    @classmethod
    def is_supported(cls, file_extension: str) -> bool:
        """Check if a file extension is supported."""
        return file_extension.lower() in cls._loaders

# Utility function for easy async loading
async def load_document_async(file_path: str, **kwargs) -> List[Document]:
    """
    Convenience function to asynchronously load a document.
    
    Args:
        file_path: Path to the document file
        **kwargs: Additional arguments to pass to the loader
        
    Returns:
        List of Document objects
    """
    import os
    _, ext = os.path.splitext(file_path)
    loader = LoaderFactory.get_loader(ext, **kwargs)
    return await loader.aload(file_path)


# Utility function for easy sync loading
def load_document(file_path: str, **kwargs) -> List[Document]:
    """
    Convenience function to synchronously load a document.
    
    Args:
        file_path: Path to the document file
        **kwargs: Additional arguments to pass to the loader
        
    Returns:
        List of Document objects
    """
    import os
    _, ext = os.path.splitext(file_path)
    loader = LoaderFactory.get_loader(ext, **kwargs)
    return loader.load(file_path)