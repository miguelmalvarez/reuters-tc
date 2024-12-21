from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from .base import BaseClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

class SklearnClassifier(BaseClassifier):
    """Wrapper for scikit-learn classifiers with built-in text vectorization."""
    
    def __init__(self, name: str, model):
        """Initialize the classifier with a name and sklearn model.
        
        Args:
            name: Name of the classifier
            model: Scikit-learn model instance
        """
        super().__init__(name)
        self.model = OneVsRestClassifier(model)
        self.tokenizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    
    def train(self, dataset):
        """Train the classifier on the given dataset.
        
        Args:
            dataset: Dataset containing 'content' and 'labels' columns
        """
        X_train = dataset['content']
        y_train = dataset['labels']
        X_vectorized = self.tokenizer.fit_transform(X_train).toarray().tolist()
        self.model.fit(X_vectorized, y_train)
    
    def predict(self, X: List[str]) -> List[List[int]]:
        """Make predictions on new data.
        
        Args:
            X: Text content to classify
            
        Returns:
            array-like: Predicted labels
        """
        X_vectorized = self.tokenizer.transform(X).toarray().tolist()
        return self.model.predict(X_vectorized)
    
    def get_params(self):
        """Get model parameters.
        
        Returns:
            dict: Model parameters
        """
        return self.model.get_params()
    

class LogisticRegressionClassifier(SklearnClassifier):
    def __init__(self):
        super().__init__("Logistic Regression", LogisticRegression())

class SVMClassifier(SklearnClassifier):
    def __init__(self):
        super().__init__("Linear SVM", LinearSVC())

class NaiveBayesClassifier(SklearnClassifier):
    def __init__(self):
        super().__init__("Naive Bayes", MultinomialNB()) 