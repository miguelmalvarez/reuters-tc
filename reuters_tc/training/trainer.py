from typing import List, Dict
import numpy as np
from sklearn.metrics import classification_report
from modelling.base import BaseClassifier

class ModelTrainer:
    """Handles training and evaluation of multiple classifiers"""
    
    def __init__(self, classifiers: List[BaseClassifier]):
        self.classifiers = classifiers
        self.results = {}
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train all classifiers"""
        for classifier in self.classifiers:
            print(f"Training {classifier.name}...")
            classifier.train(X_train, y_train)
    
    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate all classifiers and store results"""
        for classifier in self.classifiers:
            print(f"\nEvaluating {classifier.name}...")
            y_pred = classifier.predict(X_test)
            self.results[classifier.name] = classification_report(y_test, y_pred, 
                                                                  output_dict=True, zero_division=0.0)
        
        return self.results
    
    def print_results(self):
        """Print evaluation results in a formatted way"""
        print("\n=== Classification Results ===")
        for name, result in self.results.items():
            print(f"\n{name}:")
            for label in ['micro avg', 'macro avg']:
                print(f"{label}: Precision: {result[label]['precision']:.4f} | Recall: {result[label]['recall']:.4f} | F1-score: {result[label]['f1-score']:.4f}") 