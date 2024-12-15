from dataset import ReutersDatasetLoader
import pandas as pd
from datetime import datetime
from modelling.sklearn_classifiers import (
    LogisticRegressionClassifier,
    SVMClassifier,
    NaiveBayesClassifier
)
from training.trainer import ModelTrainer
from sklearn.feature_extraction.text import TfidfVectorizer

def load_reuters():
    # Initialize the loader
    loader = ReutersDatasetLoader()
    
    # Load and prepare the dataset
    train_data, test_data, train_encoded_labels, test_encoded_labels = loader.prepare()
    
    # Print some basic information
    print(f"Number of training documents: {len(train_data)}")
    print(f"Number of test documents: {len(test_data)}")
    print(f"Number of unique labels: {len(loader.label_names)}")
    print(f"Labels encoded shape: {train_encoded_labels.shape}")
    
    # Print first document example
    print("\nFirst document example:")
    print(f"Document ID: {train_data['document_id'][0]}")
    print(f"Labels: {train_data['labels'][0]}")
    return train_data, test_data, train_encoded_labels, test_encoded_labels

def run_models(train_data, train_encoded_labels, test_data, test_encoded_labels):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        train_data['content'], train_encoded_labels, 
        test_size=0.2, random_state=42
    )
    
    # TODO: Refactor representation in a different step
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Initialize classifiers
    classifiers = [
        LogisticRegressionClassifier(),
        SVMClassifier(),
        NaiveBayesClassifier()
    ]
    
    # Train and evaluate with validation set
    print("Training and evaluating with validation set ------------------------------")
    trainer = ModelTrainer(classifiers)
    trainer.train_all(X_train_vec, y_train)
    results_validation = trainer.evaluate_all(X_test_vec, y_test)
    trainer.print_results()

    # Full data training and evaluation with train and test set
    print("Training with full train data --------------------------------------------")
    X_train_vec = vectorizer.fit_transform(train_data['content'])
    y_train = train_encoded_labels
    trainer.train_all(X_train_vec, y_train)
    
    print("Evaluating with training set (all used for training) ---------------------")
    # Evaluate with training set (all used for training)
    results_train = trainer.evaluate_all(X_train_vec, y_train)
    trainer.print_results()

    print("Evaluating with (unseen) test set ----------------------------------------")
    # Evaluate with (unseen) test set
    X_test_vec = vectorizer.transform(test_data['content'])
    y_test= test_encoded_labels
    results_test = trainer.evaluate_all(X_test_vec, y_test)
    trainer.print_results()

    results_df = pd.DataFrame([
        results_train,
        results_validation,
        results_test
    ])
    
    #TODO: Record evaluation results properly
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"./reports/results_{timestamp}.csv")

    # TODO: Save models 

    # TODO: Select best model

if __name__ == "__main__":
    train_data, test_data, train_encoded_labels, test_encoded_labels = load_reuters()
    run_models(train_data, train_encoded_labels, test_data, test_encoded_labels)