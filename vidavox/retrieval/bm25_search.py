import asyncio
from rank_bm25 import BM25Okapi
import nltk
import string
from typing import List, Set, Optional, Tuple, Dict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import threading


def download_nltk_resources():
    """
    Downloads required NLTK resources synchronously.
    """
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")


class BM25_search:
    # Class variable to track if resources have been downloaded
    nltk_resources_downloaded = False

    def __init__(self, remove_stopwords: bool = True, perform_lemmatization: bool = False):
        """
        Initializes the BM25_search.

        Parameters:
        - remove_stopwords (bool): Whether to remove stopwords during preprocessing.
        - perform_lemmatization (bool): Whether to perform lemmatization on tokens.
        """
        # Ensure NLTK resources are downloaded only once
        if not BM25_search.nltk_resources_downloaded:
            download_nltk_resources()
            BM25_search.nltk_resources_downloaded = True

        self.doc_dict: Dict[str, Dict[str, any]] = {}  # {doc_id: {'text': ..., 'tokenized': ...}}
        self.doc_ids: List[str] = []
   
        self.bm25: Optional[BM25Okapi] = None
        self.remove_stopwords = remove_stopwords
        self.perform_lemmatization = perform_lemmatization
        self.stop_words: Set[str] = set(stopwords.words('english')) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if perform_lemmatization else None
        self.lock = threading.Lock()

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocesses the input text by lowercasing, removing punctuation,
        tokenizing, removing stopwords, and optionally lemmatizing.

        Parameters:
        - text (str): The input text to preprocess.

        Returns:
        - List[str]: List of preprocessed tokens.
        """
        if not text.strip():
            return []
            
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text)
        
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        if self.perform_lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens

    def add_document(self, doc_id: str, new_doc: str) -> None:
        """
        Adds a single document to the corpus and updates the BM25 index.

        Parameters:
        - doc_id (str): Unique identifier for the document.
        - new_doc (str): The document text to add.
        """
        tokenized = self.preprocess(new_doc)
        with self.lock:
            self.doc_dict[doc_id] = {'text': new_doc, 'tokenized': tokenized}
            self.doc_ids = list(self.doc_dict.keys())
            self.update_bm25()

    def add_documents(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        """
        Efficiently adds multiple documents to the corpus and updates the BM25 index once.

        Parameters:
        - docs_with_ids (List[Tuple[str, str]]): List of (doc_id, document) tuples to add.
        """

        if not docs_with_ids:
            print("No documents provided.")
            return
        with self.lock:
            for doc_id, doc in docs_with_ids:
                if not isinstance(doc, str) or not isinstance(doc_id, str):
                    print(f"Skipping invalid document or ID: {doc_id}")
                    continue
                tokenized = self.preprocess(doc)
                self.doc_dict[doc_id] = {"text": doc, "tokenized": tokenized}
            self.doc_ids = list(self.doc_dict.keys())
            self.update_bm25()
       

    def remove_document(self, doc_id: str) -> bool:
        with self.lock:
            if doc_id in self.doc_dict:
                del self.doc_dict[doc_id]
                self.doc_ids = list(self.doc_dict.keys())
                self.update_bm25()
                print(f"Removed document ID: {doc_id}")
                return True
            else:
                print(f"Document ID {doc_id} not found.")
                return False

    def update_bm25(self) -> None:
        tokenized_docs = [self.doc_dict[doc_id]["tokenized"] for doc_id in self.doc_ids]
        if tokenized_docs:
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None

    def get_scores(self, query: str) -> List[float]:
        if not query.strip():
            return []
        processed_query = self.preprocess(query)
        print(f"Tokenized Query: {processed_query}")
        with self.lock:
            if self.bm25:
                return self.bm25.get_scores(processed_query)
            else:
                print("BM25 is not initialized.")
                return []

    def get_top_n_docs(self, query: str, n: int = 5) -> List[Tuple[str, str, float]]:
        if not query.strip():
            return []
        processed_query = self.preprocess(query)
        with self.lock:
            if not self.bm25:
                return []
            scores = self.bm25.get_scores(processed_query)
            scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            results = []
            for idx, score in scored_indices[:n]:
                doc_id = self.doc_ids[idx]
                results.append((doc_id, self.doc_dict[doc_id]["text"], score))
            return results

    def clear_documents(self) -> None:
        with self.lock:
            self.doc_dict.clear()
            self.doc_ids = []
            self.bm25 = None
            print("BM25 documents cleared and index reset.")