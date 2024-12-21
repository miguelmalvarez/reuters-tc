from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer

class DatasetLoader(ABC):
    """Abstract base class for dataset loading and preparation."""
    
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.dataset = None
        
    @abstractmethod
    def load(self) -> DatasetDict:
        """Load the raw dataset into self.dataset and return it.
        
        Returns:
            pd.DataFrame: DataFrame containing at least 'content' and 'labels' columns
        """
        pass
  
    def save(self, path: str):
        """Save the dataset to a file.
        
        Args:
            path: Path where to save the dataset
            
        Raises:
            RuntimeError: If dataset hasn't been loaded yet
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Please call load() first before saving.")
        self.dataset.save_to_disk(path)

    def load_from_disk(self, path: str) -> DatasetDict:
        """Load a previously saved dataset from disk.
        
        Args:
            path: Path to the saved dataset
            
        Returns:
            DatasetDict: The loaded dataset
        """
        if self.dataset is not None:
            raise RuntimeError("Dataset already loaded. Overriding previous dataset.")
        self.dataset = DatasetDict.load_from_disk(path)
        return self.dataset
        
    @property
    def label_names(self) -> List[str]:
        """Get the names of encoded labels."""
        if self.mlb.classes_ is None:
            raise ValueError("Labels haven't been encoded yet. Call prepare() first.")
        return list(self.mlb.classes_)

    @property
    def id2label(self, id: str) -> str:
        """Get the label name for a given id."""
        if self.mlb.classes_ is None:
            raise ValueError("Labels haven't been encoded yet. Call prepare() first.")
        return self.mlb.classes_[id]
    
    @property
    def label2id(self, label: str) -> int:
        """Get the id for a given label."""
        if self.mlb.classes_ is None:
            raise ValueError("Labels haven't been encoded yet. Call prepare() first.")
        return np.where(self.mlb.classes_ == label)[0][0]

from nltk.corpus import reuters
from nltk import download
class ReutersDatasetLoader(DatasetLoader):
    """Loader for the Reuters dataset."""
    
    def __init__(self):
        """Initialize the Reuters dataset loader."""
        super().__init__()
        try:
            self.reuters = reuters
            self.document_ids = reuters.fileids()
        except ImportError:
            raise ImportError("NLTK is required. Please install it with 'pip install nltk'")
        except LookupError: 
            download('reuters')
            self.reuters = reuters
            self.document_ids = reuters.fileids()

    def load(self) -> DatasetDict:
        """Load Reuters documents and their categories.
        
        Returns:
            DatasetDict: DatasetDict with train and test datasets
        """
        train_ids = []
        train_contents = []
        train_labels = []

        test_ids = []
        test_contents = []
        test_labels = []
        
        for doc_id in self.document_ids:
            if doc_id.startswith('train'):
                train_contents.append(self.reuters.raw(doc_id))
                train_labels.append(self.reuters.categories(doc_id))
                train_ids.append(doc_id)
            else:
                test_contents.append(self.reuters.raw(doc_id))
                test_labels.append(self.reuters.categories(doc_id))
                test_ids.append(doc_id)
        
        # Encoding labels
        self.mlb.fit(train_labels)

        # Float to ensure compatibility with HF models
        train_encoded_labels = [[float(x) for x in row] for row in self.mlb.transform(train_labels)]
        test_encoded_labels = [[float(x) for x in row] for row in self.mlb.transform(test_labels)]

        encoded_train_data = pd.DataFrame({
            'document_id': train_ids,
            'content': train_contents,
            'labels': train_encoded_labels
        })

        encoded_test_data = pd.DataFrame({
            'document_id': test_ids,
            'content': test_contents,
            'labels': test_encoded_labels
        })
        
        # Create DatasetDict
        train_dataset = Dataset.from_pandas(encoded_train_data)
        test_dataset = Dataset.from_pandas(encoded_test_data)

        self.dataset = DatasetDict({'train': train_dataset, 
                                    'test':test_dataset})
        return self.dataset