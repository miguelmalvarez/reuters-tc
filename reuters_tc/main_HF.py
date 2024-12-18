from dataset import ReutersDatasetLoader
import pandas as pd
from datetime import datetime
from datasets import Dataset, DatasetDict, concatenate_datasets
import torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from modelling.transformer_classifiers import DistilBertClassifier
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

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_function(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True, max_length=512)

def prepare_dataset(dataset: DatasetDict):
    # Learning tokenization based on the whole original training set
    # Split original train set into train and validation
    split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset['train']
    validation_dataset = split_dataset['test']
    test_dataset = dataset['test']

    dataset = DatasetDict({'train': train_dataset,
                            'valid': validation_dataset,
                            'test': test_dataset})
    

    
    vectorized_dataset = dataset.map(tokenize_function, batched=True)

    # vectorized_train_dataset = Dataset.from_dict({'content': vectorized_train_content,
    #                                               'labels': train_dataset['labels'],
    #                                               'document_id': train_dataset['document_id']})
    # vectorized_validation_dataset = Dataset.from_dict({'content': vectorized_validation_content,
    #                                               'labels': validation_dataset['labels'],
    #                                               'document_id': validation_dataset['document_id']})
    # vectorized_test_dataset = Dataset.from_dict({'content': vectorized_test_content,
    #                                              'labels': test_dataset['labels'],
    #                                              'document_id': test_dataset['document_id']})
    
    
    # vectorized_dataset = DatasetDict({'train': vectorized_train_dataset,
    #                                   'valid': vectorized_validation_dataset,
    #                                   'test': vectorized_test_dataset})

    return vectorized_dataset

def run_models(vectorized_dataset: DatasetDict, num_labels: int):

    # Convert labels to float in order to be compatible with training

    # Naming of the main datasets
    train_dataset = vectorized_dataset['train']
    validation_dataset = vectorized_dataset['valid']
    test_dataset = vectorized_dataset['test']
    
    full_train_dataset = concatenate_datasets([vectorized_dataset['train'], vectorized_dataset['valid']])

    # Initialize classifiers
    classifier = DistilBertClassifier(num_labels=num_labels)
    classifier.train(full_train_dataset, validation_dataset)

    print(f"\nEvaluating {classifier.name}...")
    X_test = validation_dataset['content']
    y_test = validation_dataset['labels']
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred, output_dict=True, zero_division=0.0))

if __name__ == "__main__":
    dataset, num_labels = load_reuters()
    vectorized_dataset = prepare_dataset(dataset)
    run_models(vectorized_dataset, num_labels)
