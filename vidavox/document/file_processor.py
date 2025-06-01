from pathlib import Path
from typing import List, Optional
import platform
import time
import logging

from vidavox.utils.pretty_logger import pretty_json_log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileProcessor:
    """Handles document loading and processing from files and directories."""
    
    @staticmethod
    def get_file_metadata(file_path: Path) -> dict:
        """Extract basic metadata from a file."""
        try:
            stat = file_path.stat()
             # Cross-platform creation time handling
            if hasattr(stat, "st_birthtime"):
                creation_time = time.ctime(stat.st_birthtime)
            elif platform.system() == "Windows":
                creation_time = time.ctime(stat.st_ctime)  # Windows: st_ctime is creation time
            else:
                creation_time = "Unavailable"  # Linux has no creation time

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": stat.st_size,
                "creation_time": creation_time,
                "modification_time": time.ctime(stat.st_mtime),
            }
        except Exception as e:
            logger.warning(f"Could not get metadata for {file_path}: {e}")
            return {}
    
    @staticmethod
    def collect_files(directory: str, recursive: bool = True, 
                     allowed_extensions: Optional[List[str]] = None) -> List[str]:
        """Collect all relevant files from a directory."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Directory {directory} does not exist.")

        # Collect files using appropriate glob method
        files = list(dir_path.rglob("*")) if recursive else list(dir_path.glob("*"))

        # Filter files based on criteria
        file_paths = []
        for f in files:
            if f.is_file() and not f.name.startswith("."):
                if allowed_extensions:
                    if f.suffix.lower() in [ext.lower() for ext in allowed_extensions]:
                        file_paths.append(str(f))
                else:
                    file_paths.append(str(f))

        if not file_paths:
            logger.warning(f"No files found in directory {directory} matching criteria.")
            
        return file_paths
