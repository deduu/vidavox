
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from typing import List, Optional

from vidavox.retrieval_system.utils.process_failure import ProcessFailure
# ============================================================================
# Configuration & Validation
# ============================================================================


class SearchKind(str, Enum):
    """Supported search types."""
    BM25 = "bm25"
    FAISS = "faiss"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class EngineConfig:
    """Immutable configuration for RetrievalEngine."""
    embedding_model: str = "all-MiniLM-L6-v2"
    index_dir: Optional[Path] = None
    use_async: bool = False
    show_docs: bool = False
    search_kind: SearchKind = SearchKind.HYBRID
    batch_size: int = 100
    max_workers: int = 4
    enable_gpu_cache_clearing: bool = True
    min_gpu_freed_mb: int = 50

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        if self.index_dir:
            object.__setattr__(self, 'index_dir', Path(self.index_dir))


@dataclass
class IngestionStats:
    """Track ingestion progress and failures."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    failures: List[ProcessFailure] = field(default_factory=list)

    def record_failure(self, path: str, error: str):
        """Record a processing failure."""
        self.failed_files += 1
        self.failures.append(ProcessFailure(path, error))

    def record_success(self, chunk_count: int):
        """Record successful processing."""
        self.processed_files += 1
        self.total_chunks += chunk_count
