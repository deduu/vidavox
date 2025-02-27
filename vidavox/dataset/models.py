from dataclasses import dataclass
from typing import List

@dataclass
class KeywordPair:
    """A dataclass representing a sentence and its associated keywords."""
    sentence: str
    keywords: List[str]