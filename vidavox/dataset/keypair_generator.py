from typing import List, Dict, Optional
import pandas as pd
from datasets import Dataset, DatasetDict
from dataclasses import asdict, dataclass

from datasets import Dataset

from .models import KeywordPair
from .response_parser import ResponseParser
from .prompt_template import PromptTemplates
from vidavox.utils.hub_utils import HuggingFaceUploader
from vidavox.document.splitter import ProcessingConfig

class DatasetGenerator:
    """
    A modular class for generating datasets of sentence-keyword pairs
    using RAG (Retrieval-Augmented Generation) and LLM processing.
    """
    
    def __init__(self, rag_engine=None, doc_splitter=None, llm_client=None):
        """
        Initialize the dataset generator with RAG engine and LLM client.
        
        Args:
            rag_engine: RAG engine for retrieving relevant context
            llm_client: LLM client for generating keywords
        """
        self.rag_engine = rag_engine
        self.doc_splitter = doc_splitter
        self.llm_client = llm_client
        self._context = None
        self._prompt_template = PromptTemplates.get_default_keyword_extraction_template()
        self._pairs = []
        self._document_nodes = None
        self._parser = ResponseParser()
    
    def set_rag_engine(self, rag_engine):
        """Set or update the RAG engine."""
        self.rag_engine = rag_engine
        return self
    
    def set_doc_splitter(self, doc_splitter):
        """Set or update the document splitter."""
        self.doc_splitter = doc_splitter
        return self
    
    def set_llm_client(self, llm_client):
        """Set or update the LLM client."""
        self.llm_client = llm_client
        return self
    
    def set_prompt_template(self, prompt_template: str):
        """
        Set a custom prompt template for the LLM.
        
        Args:
            prompt_template: A string template with {context}, {n}, and {language} placeholders
        """
        self._prompt_template = prompt_template
        return self
    
    def retrieve_context(self, query: str, file_paths: List[str] = None, file_dir: str = None):
        """
        Retrieve context using the RAG engine.
        
        Args:
            query: The query to retrieve context for
            file_paths: Optional list of file paths to retrieve from
            file_dir: Optional directory to retrieve from
        
        Returns:
            Self for method chaining
        """
        if self.rag_engine is None:
            raise ValueError("RAG engine is not set")
        
        rag = self.rag_engine
        
        if file_paths:
            rag = rag.from_paths(file_paths)
        elif file_dir:
            rag = rag.from_directory(file_dir)
        
        self._context = rag.retrieve(query_text=query)
        return self
    
    
    def retrieve_nodes(self, file_path: str, config: Optional[ProcessingConfig] = None):
        """
        Process a document using the document splitter.
        
        Args:
            file_path: Path to the document file
            config: Optional configuration to override the document splitter's config
                
        Returns:
            Self for method chaining
        """
        if self.doc_splitter is None:
            raise ValueError("Document splitter is not set")
        
        # Use provided config if available
        original_config = None
        if config is not None:
            original_config = self.doc_splitter.config
            self.doc_splitter.config = config
        
        try:
            # Split the document into nodes using the splitter
            nodes = self.doc_splitter.run(file_path=file_path)
            
            # Store the nodes for later processing
            self._document_nodes = nodes
            
            # Don't set self._context here - we'll process nodes individually
            return self
        finally:
            # Restore original config if we changed it
            if original_config is not None:
                self.doc_splitter.config = original_config
    
    def set_context_directly(self, context: str):
        """
        Set the context directly without using RAG.
        
        Args:
            context: The context string to use
            
        Returns:
            Self for method chaining
        """
        self._context = context
        return self
    
    def generate(self, 
                n_pairs: int = 5, 
                language: str = "English", 
                temperature: float = 0.75) -> List[KeywordPair]:
        """
        Generate sentence-keyword pairs using the LLM.
        
        Args:
            n_pairs: Number of pairs to generate
            language: Language to generate in
            temperature: Temperature for LLM generation
        
        Returns:
            List of KeywordPair objects
        """
        if self.llm_client is None:
            raise ValueError("LLM client is not set")
        
        if self._context is None:
            raise ValueError("No context available. Call retrieve_context first.")
        
        prompt = self._prompt_template.format(
            context=self._context,
            n=n_pairs,
            language=language
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ]
        
        response = self.llm_client.chat.completions.create(
            messages=messages, 
            temperature=temperature
        )
        
        content = response.choices[0].message.content
        self._pairs = self._parser.parse_keyword_pairs(content)
        return self._pairs
    
    def generate_from_nodes(self, 
                        n_pairs_per_node: int = 3, 
                        language: str = "English", 
                        temperature: float = 0.75) -> List[KeywordPair]:
        """
        Generate sentence-keyword pairs for each document node.
        
        Args:
            n_pairs_per_node: Number of pairs to generate per node
            language: Language to generate in
            temperature: Temperature for LLM generation
        
        Returns:
            List of KeywordPair objects from all nodes
        """
        if self.llm_client is None:
            raise ValueError("LLM client is not set")
        
        if not hasattr(self, '_document_nodes') or self._document_nodes is None:
            raise ValueError("No document has been processed. Call process_document first.")
        
        all_pairs = []
        
        # Process each node individually
        for i, node in enumerate(self._document_nodes):
            # Set the context to just this node's content
            self._context = node.page_content
            
            # Generate pairs for this node
            prompt = self._prompt_template.format(
                context=self._context,
                n=n_pairs_per_node,
                language=language
            )
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ]
            
            response = self.llm_client.chat.completions.create(
                messages=messages, 
                temperature=temperature
            )
            
            content = response.choices[0].message.content
            node_pairs = self._parser.parse_keyword_pairs(content)
            
            # Add node metadata to each pair if desired
            for pair in node_pairs:
                # You could add node metadata here if useful
                # pair.node_index = i
                # pair.metadata = node.metadata
                pass
            
            all_pairs.extend(node_pairs)
        
        # Store the combined pairs
        self._pairs = all_pairs
        return all_pairs
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the generated keyword pairs to a pandas DataFrame.
        
        Returns:
            DataFrame with sentence and keywords columns
        """
        if not self._pairs:
            raise ValueError("No pairs have been generated yet. Call generate() first.")
        
        # Convert each KeywordPair to a dictionary
        data = []
        for pair in self._pairs:
            entry = {
                "sentence": pair.sentence,
                "keywords": pair.keywords,
                # You can add more derived fields here if needed
                "keyword_count": len(pair.keywords),
                "keywords_joined": ", ".join(pair.keywords)
            }
            data.append(entry)
        
        return pd.DataFrame(data)
    
    def to_huggingface_dataset(self) -> Dataset:
        """
        Convert the generated keyword pairs to a Hugging Face Dataset.
        
        Returns:
            Hugging Face Dataset object
        """
        if not self._pairs:
            raise ValueError("No pairs have been generated yet. Call generate() first.")
        
        # Convert pairs to dictionaries
        data = [asdict(pair) for pair in self._pairs]
        
        # Create a Dataset object
        return Dataset.from_list(data)
    
    def upload_to_hub(self, 
                     repo_name: str, 
                     token: str = None,
                     private: bool = False, 
                     split_name: str = "train",
                     dataset_dict: Dict[str, List[KeywordPair]] = None) -> str:
        """
        Upload the generated dataset to Hugging Face Hub.
        
        Args:
            repo_name: The name of the repository to create/update
            token: Hugging Face API token
            private: Whether the repository should be private
            split_name: The split name for the current pairs
            dataset_dict: Optional dictionary of {split_name: pairs} to upload multiple splits
        
        Returns:
            URL of the uploaded dataset
        """
        uploader = HuggingFaceUploader()
        
        if token:
            uploader.login(token=token)
        
        if dataset_dict:
            # Use multiple splits
            dataset = uploader.prepare_multi_split_dataset(dataset_dict)
        else:
            # Use a single split
            if not self._pairs:
                raise ValueError("No pairs have been generated yet. Call generate() first.")
            
            dataset = uploader.prepare_dataset(self._pairs, split_name)
        
        # Push to hub
        return uploader.upload_dataset(dataset, repo_name, private)