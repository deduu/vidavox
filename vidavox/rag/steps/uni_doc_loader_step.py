

import os
from typing import List, Optional, Callable, Dict

from vidavox.rag.context.doc_processing import DocumentProcessingContext
from vidavox.document.loader import LoaderFactory
from vidavox.pipeline import PipelineStep
from vidavox.rag.steps.uni_doc_base_step import DocumentStep


class LoadDocumentStep(DocumentStep):
    """Step to load documents from file"""
    def __init__(self):
        super().__init__("LoadDocument")
    
    def run(self, ctx: DocumentProcessingContext) -> DocumentProcessingContext:
        try:
            self.logger.info(f"Loading document: {ctx.file_path}")
            _, file_extension = os.path.splitext(ctx.file_path)
            loader = LoaderFactory.get_loader(file_extension)
            self.logger.info(f"Using loader: {loader}")
            
            ctx.documents = loader.load(ctx.file_path)
            self.logger.info(f"Loaded {len(ctx.documents)} documents from {ctx.file_path}")
            ctx.add_metadata('documents_loaded', len(ctx.documents))
            
            return ctx
        except Exception as e:
            self.logger.error(f"Error loading document {ctx.file_path}: {str(e)}")
            raise