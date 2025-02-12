# app/crud.py
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    CharacterTextSplitter,
)
from langchain.docstore.document import Document
import os
import logging
import pprint

from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pdf_with_langchain(file_path):
    """
    Loads and extracts text from a PDF file using LangChain's PyPDFLoader.

    Parameters:
        file_path (str): Path to the PDF file.

    Returns:
        List[Document]: A list of LangChain Document objects with metadata.
    """

    loader = PyPDFLoader(file_path, extract_images=True)
   
    documents = loader.load()

    return documents  # Returns a list of Document objects

def load_file_with_langchain(file_path: str):
    """
    Loads and extracts text from a PDF or DOCX file using LangChain's appropriate loader.

    Parameters:
        file_path (str): Path to the file (PDF or DOCX).

    Returns:
        List[Document]: A list of LangChain Document objects with metadata.
    """
    # Determine the file extension
    _, file_extension = os.path.splitext(file_path)
    print(f"file_extension: {file_extension}")
    # Choose the loader based on file extension
    if file_extension.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension.lower() == '.docx':
        loader = Docx2txtLoader(file_path)
    elif file_extension.lower() == '.csv':
        loader = CSVLoader(file_path)
    elif file_extension.lower() == '.xlsx':
        loader = UnstructuredExcelLoader(file_path)
    elif file_extension.lower() == '.md':
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a PDF or DOCX file.")
    
    # Load the documents
    documents = loader.load()

    return documents
def merge_configs(default_config, user_config):
    """Merge two dictionaries where user_config overrides defaults."""
    merged = default_config.copy()
    if user_config:
        for key, value in user_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return merged

def split_documents(documents, chunk_size=10000, chunk_overlap=1000, splitter_configs=None):
    """
    Splits documents into chunks using a text splitter determined by document type.
    
    Parameters:
        documents (List[Document]): List of LangChain Document objects.
        default_chunk_size (int): Default chunk size for general splitters.
        default_chunk_overlap (int): Default chunk overlap for general splitters.
        splitter_configs (dict): Optional user-defined configuration. Keys are file extensions (e.g. ".md"),
            and values are dicts with keys "splitter_class" and "params" that override defaults.
    
    Returns:
        List[Document]: List of Document objects representing chunks.
    """
    split_docs = []

    # Default configuration for splitters.
    default_splitter_config = {
        ".md": {
            "splitter_class": MarkdownHeaderTextSplitter,
            "params": {
                "headers_to_split_on": [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ],
                "return_each_line": True,  # Change this based on how you want to split
                "strip_headers": True,
            },
        },
        "default": {
            "splitter_class": RecursiveCharacterTextSplitter,
            "params": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        },
    }

    # Build a final config that merges any user overrides into the defaults.
    final_config = {}
    for ext, config in default_splitter_config.items():
        if ext == "default":
            continue
        user_conf = splitter_configs.get(ext) if splitter_configs and ext in splitter_configs else {}
        # Allow the user to override the splitter class as well as its parameters.
        splitter_class = user_conf.get("splitter_class", config["splitter_class"])
        params = merge_configs(config["params"], user_conf.get("params") if user_conf else None)
        final_config[ext] = {
            "splitter_class": splitter_class,
            "params": params,
        }
    default_final_config = default_splitter_config["default"]

    # Process each document based on its file extension.
    for doc in documents:
        ext = doc.metadata.get("source_extension", "").lower()
        config = final_config.get(ext, default_final_config)
        SplitterClass = config["splitter_class"]
        params = config["params"]

        # Instantiate the splitter with its parameters.
        splitter = SplitterClass(**params)
        split_texts = splitter.split_text(doc.page_content)

        # Wrap each chunk appropriately.
        for chunk in split_texts:
            if isinstance(chunk, str):
                new_doc = Document(page_content=chunk, metadata=doc.metadata.copy())
            elif isinstance(chunk, Document):
                # Merge metadata if chunk is already a Document.
                chunk.metadata.update(doc.metadata)
                new_doc = chunk
            else:
                raise ValueError("Unexpected type returned by splitter")
            split_docs.append(new_doc)

    return split_docs
# def split_documents(documents, chunk_size=10000, chunk_overlap=1000):
#     """
#     Splits documents into smaller chunks with overlap.

#     Parameters:
#         documents (List[Document]): List of LangChain Document objects.
#         chunk_size (int): The maximum size of each chunk.
#         chunk_overlap (int): The number of characters to overlap between chunks.

#     Returns:
#         List[Document]: List of chunked Document objects.
#     """
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#     )
#     split_docs = text_splitter.split_documents(documents)
#     return split_docs

def process_doc_file(
    file_path, chunk_size=5000, chunk_overlap=500, chunker=None
):
    """
    Loads and splits a document into chunks, with an optional custom chunking strategy.
    
    Returns:
        List[Document]: A list of document chunks.
    """
    # Load the file
    documents = load_file_with_langchain(file_path)
    logger.info(f"Loaded document: {file_path}")

    # Apply the custom chunker if provided
    if chunker:
        logger.info(f"Using custom chunker for {file_path}")
        split_docs = []
        for doc in documents:
            chunks = chunker(doc)  # Apply custom chunker
            split_docs.extend(chunks)  # Flatten the list of chunks
    else:
        # Default chunking logic
        split_docs = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logger.info(f"Document split into {len(split_docs)} chunks for {file_path}")
    return split_docs





