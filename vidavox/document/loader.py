# loaders.py
from abc import ABC, abstractmethod
from typing import List, Type
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)


class BaseDocumentLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        pass
    def count_pages(self, file_path: str) -> int:
        """
        Default “page” count = number of Document objects returned.
        Subclasses (e.g. PDF) can override for more accurate results.
        """
        docs = self.load(file_path)
        return len(docs)

class PDFLoader(BaseDocumentLoader):
    def load(self, file_path: str) -> List[Document]:
        return PyPDFLoader(file_path, extract_images=True).load()
    
    def count_pages(self, file_path: str) -> int:
        # use a low-level reader to get the real PDF page count
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        return len(reader.pages)

class DocxLoader(BaseDocumentLoader):
    def load(self, file_path: str) -> List[Document]:
        return Docx2txtLoader(file_path).load()

class ExcelLoader(BaseDocumentLoader):
    def load(self, file_path: str) -> List[Document]:
        return UnstructuredExcelLoader(file_path).load()
class _CSVLoader(BaseDocumentLoader):
    def load(self, file_path: str) -> List[Document]:
        return CSVLoader(file_path, encoding='utf-8').load()
class _TextLoader(BaseDocumentLoader):
    def load(self, file_path: str) -> List[Document]:
        return TextLoader(file_path).load()

class MarkdownLoader(BaseDocumentLoader):
    def load(self, file_path: str) -> List[Document]:
        return UnstructuredMarkdownLoader(file_path).load()


class LoaderFactory:
    _loaders = {
        '.pdf': PDFLoader,
        '.docx': DocxLoader,
        '.xlsx': ExcelLoader,
        '.csv': _CSVLoader,
        '.txt': _TextLoader,
        '.md': MarkdownLoader,
    }

    @classmethod
    def get_loader(cls, file_extension: str) -> BaseDocumentLoader:
        loader_class = cls._loaders.get(file_extension.lower())
        if not loader_class:
            raise ValueError(f"Unsupported file format: {file_extension}")
        return loader_class()

    @classmethod
    def register_loader(cls, extension: str, loader_class: Type[BaseDocumentLoader]):
        cls._loaders[extension.lower()] = loader_class