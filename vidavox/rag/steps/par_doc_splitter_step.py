import os
from typing import List, Optional, Callable, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from langchain.docstore.document import Document

from vidavox.rag.context.doc_processing import DocumentProcessingContext
from vidavox.rag.steps.uni_doc_base_step import DocumentStep
from vidavox.document.config import SplitterConfig



from vidavox.utils.script_tracker import log_processing_time
from vidavox.utils.pretty_logger import pretty_json_log


class ParallelSplitDocumentStep(DocumentStep):
    """Step to split documents in parallel"""
    def __init__(self, splitter_configs: Dict[str, SplitterConfig], max_workers: int = None):
        super().__init__("ParallelSplitDocument")
        self.splitter_configs = splitter_configs
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
    
    def run(self, ctx: DocumentProcessingContext) -> DocumentProcessingContext:
        if ctx.custom_chunker:
            return self._parallel_custom_chunker(ctx)
        elif not ctx.use_recursive:
            ctx.split_documents = ctx.documents.copy()
            return ctx
        else:
            return self._parallel_recursive_splitting(ctx)
    
    def _split_single_document(self, doc: Document) -> List[Document]:
        """Split a single document - thread-safe operation"""
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        splitter_config = self.splitter_configs.get(
            file_extension, 
            self.splitter_configs["default"]
        )
        
        try:
            splitter = splitter_config.splitter_class(**splitter_config.params)
            chunks = splitter.split_text(doc.page_content)
            
            split_docs = []
            for chunk in chunks:
                if isinstance(chunk, str):
                    split_docs.append(Document(
                        page_content=chunk,
                        metadata=doc.metadata.copy()
                    ))
                elif isinstance(chunk, Document):
                    chunk.metadata.update(doc.metadata)
                    split_docs.append(chunk)
            
            return split_docs
        except Exception as e:
            raise ValueError(f"Error splitting document: {str(e)}")
    
    def _parallel_recursive_splitting(self, ctx: DocumentProcessingContext) -> DocumentProcessingContext:
        """Apply recursive document splitting in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all documents for parallel processing
            future_to_doc = {
                executor.submit(self._split_single_document, doc): doc 
                for doc in ctx.documents
            }
            
            # Collect results with progress bar if requested
            if ctx.show_progress:
                from concurrent.futures import as_completed
                for future in tqdm(as_completed(future_to_doc), total=len(future_to_doc), desc="Splitting documents"):
                    try:
                        split_docs = future.result()
                        ctx.extend_split_documents(split_docs)
                    except Exception as e:
                        doc = future_to_doc[future]
                        self.logger.error(f"Error processing document {doc.metadata.get('source', 'unknown')}: {e}")
                        raise
            else:
                for future in future_to_doc:
                    try:
                        split_docs = future.result()
                        ctx.extend_split_documents(split_docs)
                    except Exception as e:
                        doc = future_to_doc[future]
                        self.logger.error(f"Error processing document {doc.metadata.get('source', 'unknown')}: {e}")
                        raise
        
        ctx.add_metadata('chunks_created', len(ctx.split_documents))
        return ctx
    
    def _parallel_custom_chunker(self, ctx: DocumentProcessingContext) -> DocumentProcessingContext:
        """Apply custom chunking in parallel"""
        self.logger.info(f"Using custom chunker in parallel for {ctx.file_path}")
        
        def apply_custom_chunker(doc: Document) -> List[Document]:
            return ctx.custom_chunker(doc)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {
                executor.submit(apply_custom_chunker, doc): doc 
                for doc in ctx.documents
            }
            
            if ctx.show_progress:
                from concurrent.futures import as_completed
                for future in tqdm(as_completed(future_to_doc), total=len(future_to_doc), desc="Custom chunking"):
                    chunks = future.result()
                    ctx.extend_split_documents(chunks)
            else:
                for future in future_to_doc:
                    chunks = future.result()
                    ctx.extend_split_documents(chunks)
        
        return ctx