from dataclasses import dataclass
from typing import Dict, Any, Type, Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TextSplitter
)

@dataclass
class SplitterConfig:
    splitter_class: Type[TextSplitter]
    params: Dict[str, Any]

@dataclass
class ProcessingConfig:
    chunk_size: int = 5000
    chunk_overlap: int = 500
    splitter_configs: Optional[Dict[str, SplitterConfig]] = None

    @classmethod
    def get_default_splitter_configs(cls) -> Dict[str, SplitterConfig]:
        return {
            ".md": SplitterConfig(
                splitter_class=MarkdownHeaderTextSplitter,
                params={
                    "headers_to_split_on": [
                        ("#", "Header 1"),
                        ("##", "Header 2"),
                        ("###", "Header 3"),
                    ],
                    "return_each_line": True,
                    "strip_headers": True,
                }
            ),
            "default": SplitterConfig(
                splitter_class=RecursiveCharacterTextSplitter,
                params={
                    "chunk_size": cls.chunk_size,
                    "chunk_overlap": cls.chunk_overlap,
                }
            )
        }