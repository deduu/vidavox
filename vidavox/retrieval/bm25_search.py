import asyncio
from rank_bm25 import BM25Okapi
import nltk
import string
from typing import List, Set, Optional, Tuple
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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

        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.remove_stopwords = remove_stopwords
        self.perform_lemmatization = perform_lemmatization
        self.stop_words: Set[str] = set(stopwords.words('english')) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if perform_lemmatization else None

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
        self.add_documents([(doc_id, new_doc)])

    def add_documents(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        """
        Efficiently adds multiple documents to the corpus and updates the BM25 index once.

        Parameters:
        - docs_with_ids (List[Tuple[str, str]]): List of (doc_id, document) tuples to add.
        """

        if not docs_with_ids:
            print("No documents provided.")
            return

        initial_count = len(self.documents)
        for doc_id, doc in docs_with_ids:
            if not isinstance(doc, str) or not isinstance(doc_id, str):
                print(f"Skipping invalid document or ID: {doc_id}")
                continue
                
            self.documents.append(doc)
            self.doc_ids.append(doc_id)
            self.tokenized_docs.append(self.preprocess(doc))
    
        self.update_bm25()  # Update BM25 index only once
       

    def remove_document(self, doc_id: str) -> bool:
        """
        Removes a document from the corpus based on its ID and updates the BM25 index.

        Parameters:
        - doc_id (str): The ID of the document to remove.

        Returns:
        - bool: True if document was removed, False if not found.
        """
        try:
            index = self.doc_ids.index(doc_id)
            del self.documents[index]
            del self.doc_ids[index]
            del self.tokenized_docs[index]
            self.update_bm25()
            print(f"Removed document ID: {doc_id}")
            return True
        except ValueError:
            print(f"Document ID {doc_id} not found.")
            return False

    def update_bm25(self) -> None:
        """
        Updates the BM25 index based on the current tokenized documents.
        """
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)
            # print("BM25 index has been updated.")
        else:
            print("No documents to initialize BM25.")

    def get_scores(self, query: str) -> List[float]:
        """
        Computes BM25 scores for all documents based on the given query.

        Parameters:
        - query (str): The search query.

        Returns:
        - List[float]: List of scores for each document.
        """
        if not query.strip():
            return []
            
        processed_query = self.preprocess(query)
        print(f"Tokenized Query: {processed_query}")
        
        if self.bm25:
            return self.bm25.get_scores(processed_query)
        else:
            print("BM25 is not initialized.")
            return []

    def get_top_n_docs(self, query: str, n: int = 5) -> List[Tuple[str, str, float]]:
        """
        Returns the top N documents for a given query with their scores.

        Parameters:
        - query (str): The search query.
        - n (int): Number of top documents to return.

        Returns:
        - List[Tuple[str, str, float]]: List of (doc_id, document, score) tuples.
        """
        if not query.strip() or not self.bm25:
            return []

        processed_query = self.preprocess(query)
        scores = self.bm25.get_scores(processed_query)
        
        # Create list of (score, index) tuples and sort by score
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        # Get top N results
        results = []
        for idx, score in scored_indices[:n]:
            results.append((self.doc_ids[idx], self.documents[idx], score))
        
        return results

    def clear_documents(self) -> None:
        """
        Clears all documents from the BM25 index.
        """
        self.documents = []
        self.doc_ids = []
        self.tokenized_docs = []
        self.bm25 = None
        print("BM25 documents cleared and index reset.")


async def initialize_bm25_search(remove_stopwords: bool = True, perform_lemmatization: bool = False) -> BM25_search:
    """
    Initializes the BM25_search with proper NLTK resource downloading.

    Parameters:
    - remove_stopwords (bool): Whether to remove stopwords during preprocessing.
    - perform_lemmatization (bool): Whether to perform lemmatization on tokens.

    Returns:
    - BM25_search: Initialized BM25_search instance.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, download_nltk_resources)
    return BM25_search(remove_stopwords, perform_lemmatization)