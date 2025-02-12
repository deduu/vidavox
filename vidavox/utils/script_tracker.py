# Rest remains the same...
import time
import functools
import logging
from inspect import iscoroutinefunction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_processing_time(func):
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds.")
        return result

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Async function '{func.__name__}' took {end_time - start_time:.4f} seconds.")
        return result

    return async_wrapper if iscoroutinefunction(func) else sync_wrapper

def log_partial_message(message, word_count=100):
    words = message.split()
    partial_message = ' '.join(words[:word_count]) + (f"... (+{len(words) - word_count} more words)" if len(words) > word_count else "")
    logger.info(f"{partial_message}")
