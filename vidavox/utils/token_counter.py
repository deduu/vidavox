# token_counter.py

import os
import tiktoken
import threading
from dataclasses import dataclass

# Choose the encoding based on your model, e.g., 'cl100k_base' for OpenAI models
encoding = tiktoken.get_encoding("cl100k_base")

@dataclass
class token:
    total_tokens: int
    doc_tokens: dict

def count_tokens(text):
    tokens = encoding.encode(text)
    return len(tokens)
    
class TokenCounter:
    def __init__(self):
        self.total_tokens = 0
        self.doc_tokens = {}
        self.lock = threading.Lock()

    def add_document(self, doc_id, text):
        num_tokens = count_tokens(text)
        with self.lock:
            self.doc_tokens[doc_id] = num_tokens
            self.total_tokens += num_tokens

    def remove_document(self, doc_id):
        with self.lock:
            if doc_id in self.doc_tokens:
                self.total_tokens -= self.doc_tokens[doc_id]
                del self.doc_tokens[doc_id]

    def get_total_tokens(self):
        return self.total_tokens

from typing import Protocol

# class TokenCounter(Protocol):
#     def count_tokens(self, text: str) -> int:
#         ...

class SimpleTokenCounter:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

class TikTokenCounter:
    def __init__(self, model_name: str = "gpt-4"):
        import tiktoken
        self.encoding = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))