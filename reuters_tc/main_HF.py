from dataset import ReutersDatasetLoader
import pandas as pd
from datetime import datetime
from datasets import Dataset, DatasetDict
import torch
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
    train_data, test_data, train_encoded_labels, test_encoded_labels = loader.prepare()
    num_labels = len(loader.label_names)

    # Print some basic information
    print(f"Number of training documents: {len(train_data)}")
    print(f"Number of test documents: {len(test_data)}")
    print(f"Number of unique labels: {num_labels}")
    print(f"Labels encoded shape: {train_encoded_labels.shape}")
    
    # Print first document example
    print("\nFirst document example:")
    print(f"Document ID: {train_data['document_id'][0]}")
    print(f"Labels: {train_data['labels'][0]}")
    return train_data, test_data, train_encoded_labels, test_encoded_labels, num_labels

def run_models(train_data, train_encoded_labels, test_data, test_encoded_labels, num_labels: int):

    # Convert labels to float in order to be compatible with training
    train_encoded_labels = [[float(x) for x in row] for row in train_encoded_labels]
    test_encoded_labels = [[float(x) for x in row] for row in test_encoded_labels]

    raw_dataset = Dataset.from_dict({"text": train_data['content'].values.tolist(), "labels": train_encoded_labels})
    raw_dataset_test = Dataset.from_dict({"text": test_data['content'].values.tolist(), "labels": test_encoded_labels})

    raw_dataset = raw_dataset.train_test_split(test_size=0.4, seed=42)

    dataset = DatasetDict({'train': raw_dataset['train'],
                           'eval': raw_dataset['test'], 
                           'test':raw_dataset_test})

    # TODO: Refactor representation in a different step
    # Vectorize text    
      
    print(dataset['train'].to_pandas().sample(5))
    print(dataset['eval'].to_pandas().sample(5))
    print(dataset['test'].to_pandas().sample(5))

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Naming of the main datasets
    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['eval']
    test_dataset = tokenized_datasets['test']

    print(train_dataset.to_pandas().sample(5))
    
    # Initialize classifiers
    classifier = DistilBertClassifier(num_labels=num_labels)
    classifier.train(train_dataset, eval_dataset)


if __name__ == "__main__":
    train_data, test_data, train_encoded_labels, test_encoded_labels, num_labels = load_reuters()
    run_models(train_data, train_encoded_labels, test_data, test_encoded_labels, num_labels)
