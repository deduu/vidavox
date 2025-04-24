import asyncio
from rank_bm25 import BM25Okapi
import nltk
import string
from typing import List, Set, Optional, Tuple, Dict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import threading
import logging


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


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    async def async_preprocess(self, text: str) -> List[str]:
        """
        Asynchronous version of preprocess method.
        
        Parameters:
        - text (str): The input text to preprocess.
        
        Returns:
        - List[str]: List of preprocessed tokens.
        """
        return await asyncio.to_thread(self.preprocess, text)

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
            
    async def async_add_document(self, doc_id: str, new_doc: str) -> None:
        """
        Asynchronously adds a single document to the corpus and updates the BM25 index.
        
        Parameters:
        - doc_id (str): Unique identifier for the document.
        - new_doc (str): The document text to add.
        """
        tokenized = await self.async_preprocess(new_doc)
        with self.lock:
            self.doc_dict[doc_id] = {'text': new_doc, 'tokenized': tokenized}
            self.doc_ids = list(self.doc_dict.keys())
            await asyncio.to_thread(self.update_bm25)

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
            
    async def async_add_documents(self, docs_with_ids: List[Tuple[str, str]]) -> None:
        """
        Asynchronously adds multiple documents to the corpus and updates the BM25 index once.
        
        Parameters:
        - docs_with_ids (List[Tuple[str, str]]): List of (doc_id, document) tuples to add.
        """
        if not docs_with_ids:
            print("No documents provided.")
            return
            
        # Preprocess documents in parallel
        async def process_doc(doc_id, doc):
            if not isinstance(doc, str) or not isinstance(doc_id, str):
                print(f"Skipping invalid document or ID: {doc_id}")
                return None
            tokenized = await self.async_preprocess(doc)
            return doc_id, doc, tokenized
            
        tasks = [process_doc(doc_id, doc) for doc_id, doc in docs_with_ids]
        results = await asyncio.gather(*tasks)
        
        with self.lock:
            for result in results:
                if result:
                    doc_id, doc, tokenized = result
                    self.doc_dict[doc_id] = {"text": doc, "tokenized": tokenized}
            self.doc_ids = list(self.doc_dict.keys())
            await asyncio.to_thread(self.update_bm25)

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
                
    async def async_remove_document(self, doc_id: str) -> bool:
        """
        Asynchronously removes a document from the corpus and updates the BM25 index.
        
        Parameters:
        - doc_id (str): Unique identifier for the document to remove.
        
        Returns:
        - bool: True if document was removed, False if not found.
        """
        with self.lock:
            if doc_id in self.doc_dict:
                del self.doc_dict[doc_id]
                self.doc_ids = list(self.doc_dict.keys())
                await asyncio.to_thread(self.update_bm25)
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
                
    async def async_get_scores(self, query: str) -> List[float]:
        """
        Asynchronously gets BM25 scores for a query.
        
        Parameters:
        - query (str): The query to score against the corpus.
        
        Returns:
        - List[float]: List of BM25 scores for each document.
        """
        if not query.strip():
            return []
        processed_query = await self.async_preprocess(query)
        print(f"Tokenized Query: {processed_query}")
        with self.lock:
            if self.bm25:
                return await asyncio.to_thread(self.bm25.get_scores, processed_query)
            else:
                print("BM25 is not initialized.")
                return []

    def get_top_n_docs(self, query: str, n: int = 5, include_doc_ids: Optional[List[str]] = None, 
                   exclude_doc_ids: Optional[List[str]] = None, include_prefixes: Optional[List[str]] = None,
                   exclude_prefixes: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
        """
        Gets top N documents for a query with efficient document ID and prefix filtering.

        Parameters:
        - query (str): The query to search for.
        - n (int): Number of top documents to return.
        - include_doc_ids (Optional[List[str]]): If provided, only search within these documents.
        - exclude_doc_ids (Optional[List[str]]): If provided, exclude these documents from search.
        - include_prefixes (Optional[List[str]]): If provided, only search documents with IDs starting with these prefixes.
        - exclude_prefixes (Optional[List[str]]): If provided, exclude documents with IDs starting with these prefixes.

        Returns:
        - List[Tuple[str, str, float]]: List of (doc_id, document text, score) tuples.
        """
        if not query.strip():
            return []

        processed_query = self.preprocess(query)
        
        with self.lock:
            if not self.bm25 or not self.doc_ids:
                return []
                
            # Convert exclude_doc_ids to a set for O(1) lookups
            exclude_set = set(exclude_doc_ids) if exclude_doc_ids else set()
            include_set = set(include_doc_ids) if include_doc_ids else None
            
            # Prepare prefix filters
            include_prefixes_list = include_prefixes if include_prefixes else None
            exclude_prefixes_list = exclude_prefixes if exclude_prefixes else []
            
            # Determine which document indices to score
            doc_indices_to_score = []
            filtered_doc_ids = []
            
            for i, doc_id in enumerate(self.doc_ids):
                # Skip if explicitly excluded
                if doc_id in exclude_set:
                    continue
                    
                # Skip if prefix is explicitly excluded
                if any(doc_id.startswith(prefix) for prefix in exclude_prefixes_list):
                    continue
                    
                # Only include if in the include list (if specified)
                if include_set is not None and doc_id not in include_set:
                    # But still check prefix inclusion which can override
                    if include_prefixes_list is None or not any(doc_id.startswith(prefix) for prefix in include_prefixes_list):
                        continue
                
                # If include_prefixes is specified and we got here from not being in exclude list,
                # check if the doc_id matches any of the include prefixes
                if include_set is None and include_prefixes_list is not None:
                    if not any(doc_id.startswith(prefix) for prefix in include_prefixes_list):
                        continue
                
                # If we've reached here, the document passed all filters
                doc_indices_to_score.append(i)
                filtered_doc_ids.append(doc_id)
                        
            if not doc_indices_to_score:
                return []  # No documents to search
                
            # Get scores for all documents from BM25
            all_scores = self.bm25.get_scores(processed_query)
            
            # Extract only the scores for documents we care about
            filtered_scores = [(filtered_doc_ids[i], all_scores[doc_indices_to_score[i]]) 
                            for i in range(len(doc_indices_to_score))]
            
            # Sort by score and take top n
            top_docs = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:n]
            
            # Build final result with document text
            results = []
            for doc_id, score in top_docs:
                results.append((doc_id, self.doc_dict[doc_id]["text"], score))
                
            return results

    async def async_get_top_n_docs(self, query: str, n: int = 5, include_doc_ids: Optional[List[str]] = None, 
                                exclude_doc_ids: Optional[List[str]] = None, include_prefixes: Optional[List[str]] = None,
                                exclude_prefixes: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
        """
        Asynchronously gets top N documents for a query with efficient document ID and prefix filtering.

        Parameters:
        - query (str): The query to search for.
        - n (int): Number of top documents to return.
        - include_doc_ids (Optional[List[str]]): If provided, only search within these documents.
        - exclude_doc_ids (Optional[List[str]]): If provided, exclude these documents from search.
        - include_prefixes (Optional[List[str]]): If provided, only search documents with IDs starting with these prefixes.
        - exclude_prefixes (Optional[List[str]]): If provided, exclude documents with IDs starting with these prefixes.

        Returns:
        - List[Tuple[str, str, float]]: List of (doc_id, document text, score) tuples.
        """
        if not query.strip():
            return []
            
        processed_query = await self.async_preprocess(query)
        
        # Define the function to run in a separate thread
        def filter_and_score():
            if not self.bm25 or not self.doc_ids:
                return []
                
            # Convert exclude_doc_ids to a set for O(1) lookups
            exclude_set = set(exclude_doc_ids) if exclude_doc_ids else set()
            include_set = set(include_doc_ids) if include_doc_ids else None
            
            # Prepare prefix filters
            include_prefixes_list = include_prefixes if include_prefixes else None
            exclude_prefixes_list = exclude_prefixes if exclude_prefixes else []
            
            # Determine which document indices to score
            doc_indices_to_score = []
            filtered_doc_ids = []
            
            for i, doc_id in enumerate(self.doc_ids):
                # Skip if explicitly excluded
                if doc_id in exclude_set:
                    continue
                    
                # Skip if prefix is explicitly excluded
                if any(doc_id.startswith(prefix) for prefix in exclude_prefixes_list):
                    continue
                    
                # Only include if in the include list (if specified)
                if include_set is not None and doc_id not in include_set:
                    # But still check prefix inclusion which can override
                    if include_prefixes_list is None or not any(doc_id.startswith(prefix) for prefix in include_prefixes_list):
                        continue
                
                # If include_prefixes is specified and we got here from not being in exclude list,
                # check if the doc_id matches any of the include prefixes
                if include_set is None and include_prefixes_list is not None:
                    if not any(doc_id.startswith(prefix) for prefix in include_prefixes_list):
                        continue
                
                # If we've reached here, the document passed all filters
                doc_indices_to_score.append(i)
                filtered_doc_ids.append(doc_id)
                        
            if not doc_indices_to_score:
                return []  # No documents to search
                
            # Get scores for all documents from BM25
            all_scores = self.bm25.get_scores(processed_query)
            
            # Extract only the scores for documents we care about
            filtered_scores = [(filtered_doc_ids[i], all_scores[doc_indices_to_score[i]]) 
                            for i in range(len(doc_indices_to_score))]
            
            # Sort by score and take top n
            top_docs = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:n]
            
            # Build final result with document text
            results = []
            for doc_id, score in top_docs:
                results.append((doc_id, self.doc_dict[doc_id]["text"], score))
                
            return results
        
        with self.lock:
            return await asyncio.to_thread(filter_and_score)
        
    def restore_index_from_bm25_terms(self, bm25_terms: Dict[str, Dict[str, int]]) -> None:
        """
        Rebuild the BM25 index using precomputed term frequencies.
        
        bm25_terms: a dict mapping doc_id to a dict of {token: frequency}.
        This method reconstructs each document’s tokenized text by repeating tokens per their frequency.
        """
        with self.lock:
            # Get an ordered list of document IDs.
            self.doc_ids = list(bm25_terms.keys())
            self.doc_dict = {}
            tokenized_corpus = []
            for doc_id in self.doc_ids:
                term_freq = bm25_terms[doc_id]
                # Reconstruct the tokenized document by repeating each token frequency times.
                tokens = []
                for token, freq in term_freq.items():
                    tokens.extend([token] * freq)
                # If you want, you can store an empty string or placeholder for the original text.
                self.doc_dict[doc_id] = {"text": "", "tokenized": tokens}
                tokenized_corpus.append(tokens)
            # Rebuild the BM25 index using the BM25Okapi constructor.
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Rebuilt BM25 index from {len(self.doc_ids)} stored BM25 term maps.")


    async def async_restore_index_from_bm25_terms(self, bm25_terms: Dict[str, Dict[str, int]]) -> None:
        """
        Asynchronously rebuild the BM25 index using precomputed term frequencies.
        
        bm25_terms: a dict mapping doc_id to a dict of {token: frequency}.
        This method reconstructs each document’s tokenized text by repeating tokens per their frequency.
        """
        async with self.lock:
            # Get an ordered list of document IDs.
            self.doc_ids = list(bm25_terms.keys())
            self.doc_dict = {}
            tokenized_corpus = []
            for doc_id in self.doc_ids:
                term_freq = bm25_terms[doc_id]
                # Reconstruct the tokenized document by repeating each token frequency times.
                tokens = []
                for token, freq in term_freq.items():
                    tokens.extend([token] * freq)
                # Store an empty string or placeholder for the original text.
                self.doc_dict[doc_id] = {"text": "", "tokenized": tokens}
                tokenized_corpus.append(tokens)
            # Rebuild the BM25 index using the BM25Okapi constructor.
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Rebuilt BM25 index from {len(self.doc_ids)} stored BM25 term maps.")
            
    def clear_documents(self) -> None:
        with self.lock:
            self.doc_dict.clear()
            self.doc_ids = []
            self.bm25 = None
            print("BM25 documents cleared and index reset.")
            
    async def async_clear_documents(self) -> None:
        """
        Asynchronously clears all documents from the corpus.
        """
        with self.lock:
            self.doc_dict.clear()
            self.doc_ids = []
            self.bm25 = None
            print("BM25 documents cleared and index reset.")
            
    def get_doc_terms(self, doc_id: str) -> Dict[str, int]:
        """
        Return the token frequency dictionary for a single document.
        If the doc_id doesn't exist, return {}.
        """
        with self.lock:
            doc_info = self.doc_dict.get(doc_id)
            if not doc_info:
                return {}
            tokens = doc_info['tokenized']
            freq_dict = {}
            for t in tokens:
                freq_dict[t] = freq_dict.get(t, 0) + 1
            return freq_dict
            
    async def async_get_doc_terms(self, doc_id: str) -> Dict[str, int]:
        """
        Asynchronously return the token frequency dictionary for a single document.
        
        Parameters:
        - doc_id (str): Document ID to get terms for.
        
        Returns:
        - Dict[str, int]: Dictionary mapping terms to their frequencies.
        """
        with self.lock:
            doc_info = self.doc_dict.get(doc_id)
            if not doc_info:
                return {}
                
            def count_frequencies():
                tokens = doc_info['tokenized']
                freq_dict = {}
                for t in tokens:
                    freq_dict[t] = freq_dict.get(t, 0) + 1
                return freq_dict
                
            return await asyncio.to_thread(count_frequencies)

    def get_multiple_doc_terms(self, doc_ids: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Return token frequency dictionary for multiple documents.
        {doc_id: {term: freq}}
        """
        with self.lock:
            results = {}
            for d_id in doc_ids:
                if d_id in self.doc_dict:
                    tokens = self.doc_dict[d_id]['tokenized']
                    freq_dict = {}
                    for t in tokens:
                        freq_dict[t] = freq_dict.get(t, 0) + 1
                    results[d_id] = freq_dict
            return results
            
    async def async_get_multiple_doc_terms(self, doc_ids: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Asynchronously return token frequency dictionaries for multiple documents.
        
        Parameters:
        - doc_ids (List[str]): List of document IDs to get terms for.
        
        Returns:
        - Dict[str, Dict[str, int]]: Dictionary mapping doc_ids to their term frequency dictionaries.
        """
        # Define a function to count frequencies for a single document
        async def count_doc_frequencies(d_id):
            if d_id not in self.doc_dict:
                return d_id, {}
                
            def count():
                tokens = self.doc_dict[d_id]['tokenized']
                freq_dict = {}
                for t in tokens:
                    freq_dict[t] = freq_dict.get(t, 0) + 1
                return freq_dict
                
            freq_dict = await asyncio.to_thread(count)
            return d_id, freq_dict
        
        with self.lock:
            valid_doc_ids = [d_id for d_id in doc_ids if d_id in self.doc_dict]
            
        # Process all documents concurrently
        tasks = [count_doc_frequencies(d_id) for d_id in valid_doc_ids]
        results_list = await asyncio.gather(*tasks)
        
        # Convert list of results to dictionary
        results = {d_id: freq_dict for d_id, freq_dict in results_list}
        return results


