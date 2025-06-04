# utils/gpu.py  (or vidavox/utils/gpu.py)

import logging
import torch

logger = logging.getLogger(__name__)

def clear_cuda_cache(min_freed_mb: int = 0, log: bool = True) -> None:
    """
    Free unused blocks in PyTorch’s caching allocator.

    Parameters
    ----------
    min_freed_mb : int
        If the difference between `memory_allocated()` before / after
        is smaller than this threshold, skip logging to reduce noise.
        Set to 0 to always log.
    log : bool
        Whether to emit an INFO line when cache is cleared.
    """
    if not torch.cuda.is_available():
        return

    before = torch.cuda.memory_allocated()
    torch.cuda.empty_cache()
    after  = torch.cuda.memory_allocated()

    freed = (before - after) / 1_048_576  # bytes → MiB

    if log and freed >= min_freed_mb:
        logger.info("torch.cuda.empty_cache() freed %.1f MiB", freed)
