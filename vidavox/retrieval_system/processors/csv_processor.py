# processors/csv_processor.py
"""CSV file processor."""

import logging
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from .base import BaseProcessor
from vidavox.utils.doc_tracker import compute_checksum
from vidavox.document_store.models import Document

logger = logging.getLogger(__name__)


class CSVProcessor(BaseProcessor):
    """Processor for CSV files."""
    
    def can_process(self, file_path: str, **kwargs) -> bool:
        """Check if this is a CSV file."""
        return file_path.lower().endswith('.csv')
    
    def process(
        self,
        file_path: str,
        doc_id: str,
        folder_id:str,
        existing_docs: Dict[str, Document],
        text_col: str,
        metadata_cols: Optional[List[str]] = None,
        **kwargs
    ) -> List[Tuple[str, str, Dict]]:
        """Process a CSV file as a DataFrame."""
        metadata_cols = metadata_cols or []
        
        try:
            df = pd.read_csv(file_path)
            batch_docs = []
            file_name = Path(file_path).name
            
            # Verify text column exists
            if text_col not in df.columns:
                logger.error(f"Text column '{text_col}' not found in CSV file {file_name}.")
                return []
            
            # Process each row
            for idx, row in df.iterrows():
                doc_text = str(row[text_col])
                doc_checksum = compute_checksum(doc_text)
                
                # Create row-specific doc_id
                row_doc_id = f"{doc_id}_{file_name}_row{idx}"
                
                # Create metadata dictionary
                doc_metadata = {
                    'source': file_path,
                    'file_name': file_name,
                    'row_index': idx,
                    'file_type': 'csv',
                    'folder_id': folder_id,
                    'checksum': doc_checksum
                }
                
                # Add specified metadata columns
                for col in metadata_cols:
                    if col in df.columns:
                        doc_metadata[col] = row[col]
                
                # Skip if document exists and checksum unchanged
                existing_doc = existing_docs.get(row_doc_id)
                if existing_doc and existing_doc.meta_data.get("checksum") == doc_checksum:
                    continue
                
                batch_docs.append((row_doc_id, doc_text, doc_metadata))
            
            logger.info(f"Successfully processed {len(df)} rows from CSV file {file_name}")
            return batch_docs
            
        except Exception as e:
            logger.error(f"Failed to process CSV file {file_path}: {e}")
            return []

