
from typing import List, Optional, Callable, Dict
import asyncio
import logging
from langchain.docstore.document import Document

from vidavox.rag.context.doc_processing import DocumentProcessingContext
from vidavox.rag.steps.par_doc_splitter_step import ParallelSplitDocumentStep
from vidavox.rag.steps.uni_doc_loader_step import LoadDocumentStep
from vidavox.rag.steps.uni_doc_splitter_step import SplitDocumentStep
from vidavox.pipeline import Pipeline
from vidavox.document.config import ProcessingConfig, SplitterConfig
from vidavox.document.node import DocumentNodes

from vidavox.utils.script_tracker import log_processing_time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessingPipeline(Pipeline[DocumentProcessingContext]):
    """Pipeline for document processing operations"""
    pass

class DocumentSplitter:
    """Enhanced DocumentSplitter using pipeline pattern"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None, 
                 show_progress: bool = False, 
                 use_recursive: bool = True,
                 parallel: bool = False,
                 max_workers: int = None,
                 retry_failed: bool = True,
                 max_retries: int = 2):
        # Configuration setup
        if config is None:
            config = ProcessingConfig()
        self.config = config
        
        # Setup splitter configs
        default_configs = self.config.get_default_splitter_configs()
        if config.splitter_configs:
            default_configs.update(config.splitter_configs)
        self.splitter_configs = default_configs
        
        # Pipeline configuration
        self.show_progress = show_progress
        self.use_recursive = use_recursive
        self.parallel = parallel
        self.max_workers = max_workers
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        
        # Setup pipeline steps
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup the processing pipeline"""
        steps = [LoadDocumentStep()]
        
        if self.parallel:
            steps.append(ParallelSplitDocumentStep(
                self.splitter_configs, 
                self.max_workers
            ))
        else:
            steps.append(SplitDocumentStep(self.splitter_configs))
        
        self.pipeline = DocumentProcessingPipeline(
            steps=steps,
            parallel=False,  # Steps are sequential, but internal operations can be parallel
            retry_failed=self.retry_failed,
            max_retries=self.max_retries,
            max_workers=self.max_workers
        )
    
    def get_splitter_config(self) -> Dict[str, SplitterConfig]:
        """Get current splitter configuration"""
        return self.splitter_configs
    
    async def run_async(
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
            # Create processing context
            context = DocumentProcessingContext(
                file_path=file_path,
                config=self.config,
                custom_chunker=custom_chunker,
                show_progress=self.show_progress,
                use_recursive=self.use_recursive
            )
            
            # Execute pipeline
            result_context = await self.pipeline.execute(context)
            
            logger.info(f"Document processing completed. Created {len(result_context.split_documents)} chunks")
            logger.info(f"Processing metadata: {result_context.metadata}")
            
            return DocumentNodes(result_context.split_documents)
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    @log_processing_time
    def run(
        self,
        file_path: str,
        custom_chunker: Optional[Callable[[Document], List[Document]]] = None
    ) -> DocumentNodes:
        """
        Synchronous wrapper for document processing.
        
        Args:
            file_path: Path to the document file
            custom_chunker: Optional function to implement custom chunking logic
            
        Returns:
            DocumentNodes containing processed document chunks
        """
        return asyncio.run(self.run_async(file_path, custom_chunker))
    
    # Legacy methods for backward compatibility
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Legacy method - splits documents using the new pipeline approach"""
        # Create a temporary context for splitting
        temp_context = DocumentProcessingContext(
            file_path="<in_memory>",
            documents=documents,
            show_progress=self.show_progress,
            use_recursive=self.use_recursive
        )
        
        if self.parallel:
            step = ParallelSplitDocumentStep(self.splitter_configs, self.max_workers)
        else:
            step = SplitDocumentStep(self.splitter_configs)
        
        result_context = step.run(temp_context)
        return result_context.split_documents