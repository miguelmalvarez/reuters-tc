from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from abc import ABC, abstractmethod
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score
import evaluate
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


class TransformerClassifier(ABC):
    def __init__(self, model_name: str, num_labels: int):
        self.name = f"Transformer-{model_name}"
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                        num_labels=num_labels,
                                                                        problem_type="multi_label_classification")        
        self.device = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')
        self.model.to(self.device)
    
    def train(self, X, y):
        # Implementation for fine-tuning transformer
        pass
    
    def predict(self, X):
        # Implementation for prediction with transformer
        pass
    
    def get_params(self):
        return {"model_name": self.model_name} 
    

class DistilBertClassifier(TransformerClassifier):
    def __init__(self, num_labels: int):
        super().__init__("distilbert-base-uncased", num_labels)

    def multi_label_metrics(self, predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy}
        return metrics

    def train(self, train_dataset, eval_dataset):
        # Implementation for fine-tuning transformer
        # Training configuration
        batch_size = 16

        training_args = TrainingArguments('models/'+self.name,
                                          num_train_epochs=5,
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          eval_strategy="epoch",
                                          save_strategy='epoch',
                                          learning_rate=2e-5,
                                          weight_decay=0.01,
                                          disable_tqdm=False,
                                          optim="adamw_torch",
                                          load_best_model_at_end=True,
                                          metric_for_best_model="f1")

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            result = self.multi_label_metrics(predictions=preds, labels=p.label_ids)
            return result

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics   
        )    
        print(f"Training model...{self.name}")
        trainer.train()
        metrics = trainer.evaluate()
        
        print(f"Metrics: {metrics}")
        trainer.save_model(f"models/{self.name}.bin")
    
    def predict(self, X):
        # Implementation for prediction with transformer
        pass
