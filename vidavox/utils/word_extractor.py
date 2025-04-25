from keybert import KeyBERT
from vidavox.retrieval.faiss_search import FAISS_search

def extract_keywords(doc, threshold=0.4, top_n = 5):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(doc,threshold=threshold, top_n=top_n)
    keywords = [key for key, _ in keywords]
    return keywords
