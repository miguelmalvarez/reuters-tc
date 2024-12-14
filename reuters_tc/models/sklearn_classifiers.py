from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from .base import BaseClassifier
from sklearn.multiclass import OneVsRestClassifier


class SklearnClassifier(BaseClassifier):
    """Wrapper for scikit-learn classifiers"""
    
    def __init__(self, name: str, model):
        super().__init__(name)
        self.model = OneVsRestClassifier(model)
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self):
        return self.model.get_params()

class LogisticRegressionClassifier(SklearnClassifier):
    def __init__(self):
        super().__init__("Logistic Regression", 
                        LogisticRegression(max_iter=1000))

class SVMClassifier(SklearnClassifier):
    def __init__(self):
        super().__init__("Linear SVM", 
                        LinearSVC(max_iter=1000))

class NaiveBayesClassifier(SklearnClassifier):
    def __init__(self):
        super().__init__("Naive Bayes", 
                        MultinomialNB()) 