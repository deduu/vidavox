
import os
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Callable, Dict, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Lock
from tqdm import tqdm
from langchain.docstore.document import Document
from vidavox.document.config import ProcessingConfig, SplitterConfig
from vidavox.document.loader import LoaderFactory
from vidavox.document.node import DocumentNodes

from vidavox.utils.script_tracker import log_processing_time
from vidavox.utils.pretty_logger import pretty_json_log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TContext = TypeVar('TContext')

# ==============================================
# Generic Pipeline Infrastructure
# ==============================================

class PipelineStep(Generic[TContext], ABC):
    """Abstract base class for pipeline steps"""
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def run(self, context: TContext) -> TContext:
        """Execute the step with the given context"""
        pass

class Pipeline(Generic[TContext]):
    """Generic pipeline with shared execution logic"""
    def __init__(self, steps: List[PipelineStep[TContext]], 
                 parallel: bool = False, 
                 retry_failed: bool = False,
                 max_retries: int = 3,
                 max_workers: int = None):
        self.steps = steps
        self.parallel = parallel
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = Lock()  # For thread safety

    async def execute(self, initial_context: TContext) -> TContext:
        """Shared execution logic with monitoring, retries, parallel execution"""
        ctx = initial_context
        start_time = time.time()
        
        # Sequential execution
        if not self.parallel:
            for i, step in enumerate(self.steps):
                ctx = await self._execute_step_with_retry(step, ctx, i)
        else:
            # Parallel execution for independent steps
            ctx = await self._execute_parallel(ctx)
        
        total_time = time.time() - start_time
        self.logger.info(f"Pipeline completed in {total_time:.2f}s")
        
        # Add execution metadata if context supports it
        if hasattr(ctx, 'add_metadata'):
            ctx.add_metadata('execution_time', total_time)
        
        return ctx
    
    async def _execute_step_with_retry(self, step, ctx, step_index):
        """Shared retry logic with thread safety"""
        for attempt in range(self.max_retries + 1):
            try:
                step_start = time.time()
                # Execute in thread pool for CPU-bound operations
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await asyncio.get_event_loop().run_in_executor(
                        executor, step.run, ctx
                    )
                step_time = time.time() - step_start
                self.logger.info(f"Step {step.name} completed in {step_time:.2f}s")
                return result
            except Exception as e:
                if attempt < self.max_retries and self.retry_failed:
                    self.logger.warning(f"Step {step.name} failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Step {step.name} failed after {attempt + 1} attempts: {e}")
                    raise
    
    async def _execute_parallel(self, ctx):
        """Parallel execution for independent steps"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                asyncio.get_event_loop().run_in_executor(executor, step.run, ctx)
                for step in self.steps
            ]
            results = await asyncio.gather(*tasks)
            return results[-1]  # Retu