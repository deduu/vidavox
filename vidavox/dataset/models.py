from dataclasses import dataclass
from typing import List


@dataclass
class KeywordPair:
    """A dataclass representing a sentence and its associated keywords."""
    sentence: str
    keywords: List[str]
    keyword_count: int = 0
    keywords_joined: str = ""

    def __post_init__(self):
        self.keyword_count = len(self.keywords)
        self.keywords_joined = ", ".join(self.keywords)
