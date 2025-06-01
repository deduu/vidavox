from vidavox.pipeline import PipelineStep
from vidavox.rag.context.doc_processing import DocumentProcessingContext

class DocumentStep(PipelineStep[DocumentProcessingContext]):
    """Base class for document processing steps"""
    pass
