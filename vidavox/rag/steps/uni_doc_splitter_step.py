import os
from typing import List, Optional, Callable, Dict
from tqdm import tqdm
from langchain.docstore.document import Document

from vidavox.rag.context.doc_processing import DocumentProcessingContext
from vidavox.rag.steps.uni_doc_base_step import DocumentStep
from vidavox.document.config import SplitterConfig



from vidavox.utils.script_tracker import log_processing_time
from vidavox.utils.pretty_logger import pretty_json_log

class SplitDocumentStep(DocumentStep):
    """Step to split documents into chunks"""
    def __init__(self, splitter_configs: Dict[str, SplitterConfig]):
        super().__init__("SplitDocument")
        self.splitter_configs = splitter_configs
    
    def run(self, ctx: DocumentProcessingContext) -> DocumentProcessingContext:
        if ctx.custom_chunker:
            return self._apply_custom_chunker(ctx)
        elif not ctx.use_recursive:
            ctx.split_documents = ctx.documents.copy()
            return ctx
        else:
            return self._apply_recursive_splitting(ctx)
    
    def _apply_custom_chunker(self, ctx: DocumentProcessingContext) -> DocumentProcessingContext:
        """Apply custom chunking logic"""
        self.logger.info(f"Using custom chunker for {ctx.file_path}")
        doc_iterator = tqdm(ctx.documents, desc="Custom chunking") if ctx.show_progress else ctx.documents
        
        for doc in doc_iterator:
            chunks = ctx.custom_chunker(doc)
            ctx.extend_split_documents(chunks)
        
        return ctx
    
    def _apply_recursive_splitting(self, ctx: DocumentProcessingContext) -> DocumentProcessingContext:
        """Apply recursive document splitting"""
        doc_iterator = tqdm(ctx.documents, desc="Splitting documents") if ctx.show_progress else ctx.documents
        
        for doc in doc_iterator:
            pretty_json_log(self.logger, f"Processing document metadata: {doc.metadata}")
            file_extension = os.path.splitext(doc.metadata["source"])[1]
            pretty_json_log(self.logger, f"File extension: {file_extension}")
            
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
                
                ctx.extend_split_documents(split_docs)
                
            except Exception as e:
                raise ValueError(f"Error splitting document: {str(e)}")
        
        ctx.add_metadata('chunks_created', len(ctx.split_documents))
        return ctx