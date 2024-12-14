from typing import List, Dict
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from models.base import BaseClassifier

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
            
            self.results[classifier.name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'report': classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
            }
        
        return self.results
    
    def print_results(self):
        """Print evaluation results in a formatted way"""
        print("\n=== Classification Results ===")
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print("\nClassification Report:")
            report = result['report']
            for label in report:
                if label in ['micro avg', 'macro avg', 'weighted avg']:
                    print(f"\n{label}:")
                    print(f"  Precision: {report[label]['precision']:.4f}")
                    print(f"  Recall: {report[label]['recall']:.4f}")
                    print(f"  F1-score: {report[label]['f1-score']:.4f}") 