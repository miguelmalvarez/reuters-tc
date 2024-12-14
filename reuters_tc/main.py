from dataset import ReutersDatasetLoader
from models.sklearn_classifiers import (
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
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Initialize classifiers
    classifiers = [
        LogisticRegressionClassifier(),
        SVMClassifier(),
        NaiveBayesClassifier()
    ]
    
    # Train and evaluate
    trainer = ModelTrainer(classifiers)
    trainer.train_all(X_train_vec, y_train)
    trainer.evaluate_all(X_test_vec, y_test)
    trainer.print_results()


    #Full data training
    X_train_vec = vectorizer.fit_transform(train_data['content'])
    X_test_vec = vectorizer.transform(test_data['content'])
    y_train = train_encoded_labels
    y_test= test_encoded_labels
    trainer.train_all(X_train_vec, y_train)
    trainer.evaluate_all(X_test_vec, y_test)
    trainer.print_results()


    #Run test set
    # X_eval_vec = vectorizer.transform(test_data['content'])
    # y_eval = test_encoded_labels
    # trainer.evaluate_all(X_test_vec, y_test)
    # trainer.print_results()

if __name__ == "__main__":
    train_data, test_data, train_encoded_labels, test_encoded_labels = load_reuters()
    run_models(train_data, train_encoded_labels, test_data, test_encoded_labels)