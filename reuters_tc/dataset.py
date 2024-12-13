from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class DatasetLoader(ABC):
    """Abstract base class for dataset loading and preparation."""
    
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.data: pd.DataFrame = None
        self.encoded_labels: np.ndarray = None
        
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
        self.data = self.load()
        self.encoded_labels = self._encode_labels(self.data['labels'])
        return self.data, self.encoded_labels
    
    def _encode_labels(self, labels: List[List[str]]) -> np.ndarray:
        """Convert list of label lists to multi-hot encoded matrix.
        
        Args:
            labels: List of label lists for each document
            
        Returns:
            np.ndarray: Multi-hot encoded label matrix
        """
        return self.mlb.fit_transform(labels)
    
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
        contents = []
        labels = []
        
        for doc_id in self.document_ids:
            contents.append(self.reuters.raw(doc_id))
            labels.append(self.reuters.categories(doc_id))
        
        return pd.DataFrame({
            'document_id': self.document_ids,
            'content': contents,
            'labels': labels
        })