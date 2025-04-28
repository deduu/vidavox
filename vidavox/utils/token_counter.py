# token_counter.py
import threading
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Optional

import tiktoken   # pip install tiktoken

# ------------------------------------------------------------
# Choose the encoding that matches your model
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Tokenise `text` and return number of tokens."""
    return len(encoding.encode(text))

# ------------------------------------------------------------
@dataclass
class TokenSnapshot:
    total_tokens: int
    doc_tokens: Dict[str, int]                  # doc_id -> tokens
    user_total_tokens: Dict[str, int]           # user_id -> tokens
    user_doc_tokens: Dict[str, Dict[str, int]]  # user_id -> {doc_id: tokens}

# ------------------------------------------------------------
class TokenCounter:
    """
    Thread-safe token counter with *optional* user_id bookkeeping.
    ──────────────────────────────────────────────────────────────
    • add_document(doc_id, text, user_id=None)
    • remove_document(doc_id)
    • get_total_tokens()
    • get_user_total_tokens(user_id)
    • snapshot()   → a frozen dataclass you can log / pickle
    """

    def __init__(self) -> None:
        self.total_tokens: int = 0                              # global total
        self.doc_tokens: Dict[str, int] = {}                    # doc_id -> tokens
        self.doc_owner:  Dict[str, Optional[str]] = {}          # doc_id -> user_id/None

        self.user_total_tokens: Dict[str, int] = defaultdict(int)
        self.user_doc_tokens: Dict[str, Dict[str, int]] = defaultdict(dict)

        self.lock = threading.Lock()

    # -------------- mutation ----------------------------------
    def add_document(self, doc_id: str, text: str, user_id: Optional[str] = None) -> None:
        """
        Register (or re-register) a document. If the same doc_id already exists
        the token delta is applied so totals stay correct.
        """
        num_tokens = count_tokens(text)

        with self.lock:
            # how many tokens did this doc already consume?
            prev_tokens = self.doc_tokens.get(doc_id, 0)
            prev_owner  = self.doc_owner.get(doc_id)

            # ---- 1. apply global diff ----
            self.total_tokens += num_tokens - prev_tokens

            # ---- 2. revert the old owner (if any) ----
            if prev_owner is not None:
                self.user_total_tokens[prev_owner] -= prev_tokens
                self.user_doc_tokens[prev_owner].pop(doc_id, None)
                if self.user_total_tokens[prev_owner] <= 0:
                    self.user_total_tokens.pop(prev_owner, None)
                    self.user_doc_tokens.pop(prev_owner, None)

            # ---- 3. assign new values ----
            self.doc_tokens[doc_id] = num_tokens
            self.doc_owner[doc_id]  = user_id

            if user_id is not None:
                self.user_total_tokens[user_id] += num_tokens
                self.user_doc_tokens[user_id][doc_id] = num_tokens

    def remove_document(self, doc_id: str) -> None:
        with self.lock:
            if doc_id not in self.doc_tokens:
                return

            tokens = self.doc_tokens.pop(doc_id)
            self.total_tokens -= tokens

            owner = self.doc_owner.pop(doc_id, None)
            if owner is not None:
                self.user_total_tokens[owner] -= tokens
                self.user_doc_tokens[owner].pop(doc_id, None)
                if self.user_total_tokens[owner] <= 0:
                    self.user_total_tokens.pop(owner, None)
                    self.user_doc_tokens.pop(owner, None)

    # -------------- getters -----------------------------------
    def get_total_tokens(self) -> int:
        return self.total_tokens

    def get_user_total_tokens(self, user_id: str) -> int:
        with self.lock:
            return self.user_total_tokens.get(user_id, 0)

    def get_doc_tokens(self, doc_id: str) -> int:
        with self.lock:
            return self.doc_tokens.get(doc_id, 0)

    # -------------- utilities ---------------------------------
    def snapshot(self) -> TokenSnapshot:
        """
        Return an *immutable* view for logging or persistence.
        (Deep-copies so concurrent mutation won’t change the snapshot.)
        """
        with self.lock:
            return TokenSnapshot(
                total_tokens      = self.total_tokens,
                doc_tokens        = dict(self.doc_tokens),
                user_total_tokens = dict(self.user_total_tokens),
                user_doc_tokens   = {u: dict(d) for u, d in self.user_doc_tokens.items()},
            )

# ------------------------------------------------------------
# Optional lightweight counters that satisfy the same Protocol
class SimpleTokenCounter:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

class TikTokenCounter:
    def __init__(self, model_name: str = "gpt-4"):
        self.encoding = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
