from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

#TODO: Support split definition in the oringinal dataset
class DatasetLoader(ABC):
    """Abstract base class for dataset loading and preparation."""
    
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.train_data: pd.DataFrame = None
        self.test_data: pd.DataFrame = None
        self.train_encoded_labels: np.ndarray = None
        self.test_encoded_labels: np.ndarray = None
        
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the raw dataset.
        
        Returns:
            pd.DataFrame: DataFrame containing at least 'content' and 'labels' columns
        """
        pass
    
    def prepare(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare the dataset by loading and encoding labels.
        
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: (processed dataframe, encoded labels)
        """
        self.train_data, self.test_data = self.load()
        self.mlb.fit(self.train_data['labels'])
        self.train_encoded_labels = self.mlb.transform(self.train_data['labels'])
        self.test_encoded_labels = self.mlb.transform(self.test_data['labels'])
        return self.train_data, self.test_data, self.train_encoded_labels, self.test_encoded_labels
    
    @property
    def label_names(self) -> List[str]:
        """Get the names of encoded labels."""
        if self.mlb.classes_ is None:
            raise ValueError("Labels haven't been encoded yet. Call prepare() first.")
        return list(self.mlb.classes_)


from nltk.corpus import reuters
import nltk
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
            nltk.download('reuters')
            self.reuters = reuters
            self.document_ids = reuters.fileids()
    
    def load(self) -> pd.DataFrame:
        """Load Reuters documents and their categories.
        
        Returns:
            pd.DataFrame: DataFrame with columns ['document_id', 'content', 'labels']
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
        
        train_data = pd.DataFrame({
            'document_id': train_ids,
            'content': train_contents,
            'labels': train_labels
        })

        test_data = pd.DataFrame({
            'document_id': test_ids,
            'content': test_contents,
            'labels': test_labels
        })

        return train_data, test_data