from typing import Dict, List
from dataclasses import asdict
from datasets import Dataset, DatasetDict
from huggingface_hub import login

from vidavox.dataset.models import KeywordPair

class HuggingFaceUploader:
    """Utility class for uploading datasets to Hugging Face Hub."""
    
    @staticmethod
    def login(token: str) -> None:
        """
        Log in to Hugging Face Hub.
        
        Args:
            token: Hugging Face API token
        """
        login(token=token)
    
    @staticmethod
    def prepare_dataset(pairs: List[KeywordPair], split_name: str = "train") -> DatasetDict:
        """
        Convert keyword pairs to a Hugging Face DatasetDict with a single split.
        
        Args:
            pairs: List of KeywordPair objects
            split_name: Name of the dataset split
            
        Returns:
            DatasetDict object with the specified split
        """
        data = [asdict(pair) for pair in pairs]
        dataset = Dataset.from_list(data)
        return DatasetDict({split_name: dataset})
    
    @staticmethod
    def prepare_multi_split_dataset(dataset_dict: Dict[str, List[KeywordPair]]) -> DatasetDict:
        """
        Convert a dictionary of splits to a DatasetDict.
        
        Args:
            dataset_dict: Dictionary mapping split names to lists of KeywordPair objects
            
        Returns:
            DatasetDict object with multiple splits
        """
        datasets_dict = {}
        for split, pairs in dataset_dict.items():
            data = [asdict(pair) for pair in pairs]
            datasets_dict[split] = Dataset.from_list(data)
        
        return DatasetDict(datasets_dict)
    
    @staticmethod
    def upload_dataset(dataset: DatasetDict, repo_name: str, private: bool = False, revision:str = None) -> str:
        """
        Upload a dataset to Hugging Face Hub.
        
        Args:
            dataset: DatasetDict object to upload
            repo_name: Name of the repository
            private: Whether the repository should be private
            
        Returns:
            URL of the uploaded dataset
        """
        repo_url = dataset.push_to_hub(
            repo_id=repo_name,
            private=private,
            revision=revision
        )
        
        return repo_url