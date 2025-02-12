# token_counter.py

import os
import tiktoken

# Choose the encoding based on your model, e.g., 'cl100k_base' for OpenAI models
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    tokens = encoding.encode(text)
    return len(tokens)
    
class TokenCounter:
    def __init__(self):
        self.total_tokens = 0
        self.doc_tokens = {}

    def add_document(self, doc_id, text):
        num_tokens = count_tokens(text)
        self.doc_tokens[doc_id] = num_tokens
        self.total_tokens += num_tokens

    def remove_document(self, doc_id):
        if doc_id in self.doc_tokens:
            self.total_tokens -= self.doc_tokens[doc_id]
            del self.doc_tokens[doc_id]

    def get_total_tokens(self):
        return self.total_tokens