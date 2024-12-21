from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BaseClassifier(ABC):
    """Base class for all classifiers"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the classifier (and vectorizer if needed"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for given input"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        pass 