from dataset import ReutersDatasetLoader
import pandas as pd
from datasets import DatasetDict, concatenate_datasets
from datetime import datetime
from training.trainer import ModelTrainer
import numpy as np
from modelling.sklearn_classifiers import *
from modelling.transformer_classifiers import *

def prepare_dataset(dataset: DatasetDict, test_size: float = 0.2, sampling_ratio: float = 1.0):
    if sampling_ratio < 1.0:    
        num_samples = int(len(dataset['train']) * sampling_ratio)
        random_indices = np.random.choice(
            len(dataset['train']), 
            size=num_samples, 
            replace=False  
        )
        dataset['train'] = dataset['train'].select(random_indices)

    split_dataset = dataset['train'].train_test_split(test_size=test_size, seed=42)
    train_dataset = split_dataset['train']
    validation_dataset = split_dataset['test']
    test_dataset = dataset['test']

    return DatasetDict({'train': train_dataset,
                        'valid': validation_dataset,
                        'test': test_dataset})

def run_models(dataset: DatasetDict):
    
    trainer = ModelTrainer(classifiers)
    full_train_dataset = concatenate_datasets([dataset['train'], dataset['valid']])
    
    # # Training with train split and evaluating with validation set
    # print("Training with train split and evaluating with validation set -----------------------")
    # trainer.train_all(dataset['train'])
    # results_validation = trainer.evaluate_all(dataset['valid'])
    # trainer.print_results()
    
    # # Full data training and evaluation with train and test set
    print("Training with full train data --------------------------------------------")
    trainer.train_all(full_train_dataset)
    
    # print("Evaluating with training set (all used for training) ---------------------")
    # # Evaluate with training set (all used for training)
    # results_train = trainer.evaluate_all(full_train_dataset)
    # trainer.print_results()

    print("Evaluating with (unseen) test set ----------------------------------------")
    # Evaluate with (unseen) test set
    results_test = trainer.evaluate_all(dataset['test'])
    trainer.print_results()

    results_df = pd.DataFrame([
        # results_train,
        # results_validation,
        results_test
    ])
    
    #TODO: Record evaluation results properly
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"./reports/results_{timestamp}.csv") # TODO: Fix formating

    # TODO: Save models 

    # TODO: Select best model


if __name__ == "__main__":
    # Load the dataset
    loader = ReutersDatasetLoader()
    dataset = loader.load()
    num_labels = len(loader.label_names)

    # Print some basic information
    print(f"Number of training documents: {len(dataset['train'])}")
    print(f"Number of test documents: {len(dataset['test'])}")
    print(f"Number of unique labels: {num_labels}")

    # Prepare the dataset
    dataset = prepare_dataset(dataset)

    # TODO: Move to config file
    classifiers = [
        DistilBertClassifier(num_labels=num_labels),
        LogisticRegressionClassifier(),
        NaiveBayesClassifier(),
        SVMClassifier()
      ]
    
    # Run the models
    run_models(dataset)

