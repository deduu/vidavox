# utils.py
import hashlib

def compute_checksum(text: str) -> str:
    """Compute a MD5 checksum for the given text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()
