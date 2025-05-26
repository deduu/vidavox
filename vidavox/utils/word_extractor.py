from keybert import KeyBERT
from typing import  Optional
from vidavox.retrieval.faiss_search import FAISS_search


def extract_keywords(
    doc,
    threshold: float = 0.4,
    top_n: int = 1,
    embedding: Optional[str] = None
):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(doc,threshold=threshold, top_n=top_n)
    keywords = [key for key, _ in keywords]
    return keywords
