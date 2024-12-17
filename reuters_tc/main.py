from dataset import ReutersDatasetLoader
import pandas as pd
from datasets import DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split
from datetime import datetime
from datasets import Dataset
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
    dataset = loader.load()
    num_labels = len(loader.label_names)

    # Print some basic information
    print(f"Number of training documents: {len(dataset['train'])}")
    print(f"Number of test documents: {len(dataset['test'])}")
    print(f"Number of unique labels: {num_labels}")
   
    # Print first document example
    print("\nFirst document example:")
    print(dataset['train'][0])
    
    return dataset, num_labels

def tokenize_function(examples):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    vectorizer.fit(examples['content'])
    return vectorizer

def prepare_dataset(dataset: DatasetDict):

    # Learning tokenization based on the whole original training set
    vectorizer = tokenize_function(dataset['train'])

    # Split original train set into train and validation
    split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset['train']
    validation_dataset = split_dataset['test']
    test_dataset = dataset['test']

    vectorized_train_content = vectorizer.transform(train_dataset['content']).toarray().tolist()
    vectorized_validation_content = vectorizer.transform(validation_dataset['content']).toarray().tolist()
    vectorized_test_content = vectorizer.transform(test_dataset['content']).toarray().tolist()

    vectorized_train_dataset = Dataset.from_dict({'content': vectorized_train_content,
                                                  'labels': train_dataset['labels'],
                                                  'document_id': train_dataset['document_id']})
    vectorized_validation_dataset = Dataset.from_dict({'content': vectorized_validation_content,
                                                  'labels': validation_dataset['labels'],
                                                  'document_id': validation_dataset['document_id']})
    vectorized_test_dataset = Dataset.from_dict({'content': vectorized_test_content,
                                                 'labels': test_dataset['labels'],
                                                 'document_id': test_dataset['document_id']})
    
    
    vectorized_dataset = DatasetDict({'train': vectorized_train_dataset,
                                      'valid': vectorized_validation_dataset,
                                      'test': vectorized_test_dataset})

    return vectorized_dataset

def run_models(vectorized_dataset: DatasetDict, num_labels: int):
    
    # Initialize classifiers
    classifiers = [
        LogisticRegressionClassifier(),
        SVMClassifier(),
        NaiveBayesClassifier()
      ]

    trainer = ModelTrainer(classifiers)
    # Training with train split and evaluating with validation set
    print("Training with train split and evaluating with validation set -----------------------")
    trainer.train_all(vectorized_dataset['train'])
    results_validation = trainer.evaluate_all(vectorized_dataset['valid'])
    trainer.print_results()

    full_train_dataset = concatenate_datasets([vectorized_dataset['train'], vectorized_dataset['valid']])

    # Full data training and evaluation with train and test set
    print("Training with full train data --------------------------------------------")
    trainer.train_all(full_train_dataset)
    
    print("Evaluating with training set (all used for training) ---------------------")
    # Evaluate with training set (all used for training)
    results_train = trainer.evaluate_all(full_train_dataset)
    trainer.print_results()

    print("Evaluating with (unseen) test set ----------------------------------------")
    # Evaluate with (unseen) test set
    results_test = trainer.evaluate_all(vectorized_dataset['test'])
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
    dataset, num_labels = load_reuters()
    vectorized_dataset = prepare_dataset(dataset)
    run_models(vectorized_dataset, num_labels)
