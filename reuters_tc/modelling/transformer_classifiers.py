from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
from abc import ABC, abstractmethod
from datasets import Dataset
from transformers import Trainer, TrainingArguments, TextClassificationPipeline
from typing import List

class TransformerClassifier(ABC):
    """Wrapper for Hugging Face transformers for sequence classification."""

    def __init__(self, model_name: str, num_labels: int, max_length: int):
        """Initialize the transformer classifier with a model name and number of labels.
        
        Args:
            model_name: Name of the transformer model
            num_labels: Number of labels to classify
        """
        self.name = f"Transformer-{model_name}"
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       padding="max_length",
                                                       truncation=True,
                                                       max_length=self.max_length) 
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                        num_labels=self.num_labels,
                                                                        problem_type="multi_label_classification") # TODO: Initialise in training?       
        self.device = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')
        self.model.to(self.device)
    
    @abstractmethod
    def train(self, dataset: Dataset):
        """Train the transformer model on the given dataset.
        
        Args:
            dataset: Dataset containing 'content' and 'labels' columns
        """
        pass
    
    def predict(self, X: List[str]) -> List[List[int]]:
        """Make predictions on new data.
        
        Args:
            X: Text content to classify
            
        Returns:
            array-like: Predicted labels
        """
        pipe = TextClassificationPipeline(tokenizer=self.tokenizer,
                                          padding="max_length",
                                          truncation=True,
                                          max_length=self.max_length,
                                          model=self.model,                                           
                                          return_all_scores=True) # TODO: Warning on return_all_scores and dict/list order for labels
        
        predictions = []
        for doc_results in pipe(X):
            doc_predictions = [1 if label_score['score'] >= 0.5 else 0 for label_score in doc_results]
            predictions.append(doc_predictions)

        return predictions

    def save_model(self, output_dir: str):
        """Save the transformer model and tokenizer to a directory.
        
        Args:
            output_dir: Directory to save the model and tokenizer
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
    
        print(f"Model and tokenizer saved to {output_dir}")
    
    def load_model(self, output_dir: str):
        """Load the transformer model and tokenizer from a directory.
        
        Args:
            output_dir: Directory to load the model and tokenizer from
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
    def get_params(self):
        """Get model parameters.
        
        Returns:
            dict: Model parameters
        """
        return {"model_name": self.model_name} 
    
    def _tokenize_function(self, examples: Dataset):
        return self.tokenizer(examples["content"], padding="max_length", truncation=True, max_length=self.max_length)
    

class DistilBertClassifier(TransformerClassifier):
    def __init__(self, num_labels: int):
        super().__init__("distilbert-base-uncased", num_labels=num_labels, max_length=512)

    def train(self, dataset: Dataset): 
        """Train the model on the given dataset.
        
        Args:
            train_dataset: Dataset containing 'content' and 'labels' columns
        """
        vectorized_dataset = dataset.map(self._tokenize_function, batched=True)

        batch_size = 8
        training_args = TrainingArguments('models/'+self.name,
                                          num_train_epochs=5,
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          learning_rate=2e-5,
                                          weight_decay=0.01,
                                          disable_tqdm=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=vectorized_dataset,
            tokenizer=self.tokenizer)
            
        print(f"Training model...{self.name}")
        model = trainer.train()
        trainer.save_model(f"models/{self.name}.bin")
        return model
