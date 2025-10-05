import os
import json
import mlflow
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from transformers import (
    PegasusForConditionalGeneration, 
    PegasusTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Dataset

from src.components.summarization import MeetingSummarizer

class SummarizerTrainingPipeline:
    """
    Pipeline for fine-tuning the summarization model
    """
    def __init__(self, base_model_name="google/pegasus-xsum", 
                 output_dir="models/summarizer",
                 data_dir="data/transcribed"):
        """
        Initialize the training pipeline
        
        Args:
            base_model_name (str): Base model to fine-tune
            output_dir (str): Directory to save the model
            data_dir (str): Directory containing transcribed data
        """
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        self.tokenizer = PegasusTokenizer.from_pretrained(base_model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(base_model_name)
    
    def prepare_data(self):
        """
        Prepare data for fine-tuning
        
        Returns:
            tuple: (train_dataset, validation_dataset)
        """
        # Load transcript files
        transcript_files = list(self.data_dir.glob("*.json"))
        
        if not transcript_files:
            raise ValueError(f"No transcript files found in {self.data_dir}")
        
        # Collect transcript data
        data = []
        for transcript_file in transcript_files:
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            
            transcript_text = transcript_data["text"]
            
            # For demonstration - in a real scenario, you would have human-written 
            # reference summaries for training. Here we'll generate summaries as a placeholder
            # (In production, you'd have actual reference summaries)
            summarizer = MeetingSummarizer(model_name=self.base_model_name)
            summary = summarizer.summarize(transcript_text)
            
            data.append({
                "document": transcript_text,
                "summary": summary
            })
        
        # Split into train and validation
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
        
        # Convert to datasets
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
        val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=["document", "summary"]
        )
        
        tokenized_val = val_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=["document", "summary"]
        )
        
        return tokenized_train, tokenized_val
    
    def _tokenize_function(self, examples):
        """Tokenize the inputs and targets"""
        model_inputs = self.tokenizer(examples["document"], max_length=1024, truncation=True)
        
        # Set up the targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["summary"], max_length=256, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def train(self, epochs=3, batch_size=4, learning_rate=5e-5):
        """
        Fine-tune the model
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
        """
        # Start MLflow run
        mlflow.set_experiment("meeting-summarizer")
        with mlflow.start_run(run_name=f"finetune-{self.base_model_name}"):
            
            # Log parameters
            mlflow.log_param("base_model", self.base_model_name)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            
            # Prepare data
            tokenized_train, tokenized_val = self.prepare_data()
            
            # Data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model
            )
            
            # Training arguments
            training_args = Seq2SeqTrainingArguments(
                output_dir=str(self.output_dir / "checkpoints"),
                evaluation_strategy="epoch",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                weight_decay=0.01,
                save_total_limit=3,
                num_train_epochs=epochs,
                predict_with_generate=True,
                fp16=torch.cuda.is_available(),
                report_to="mlflow"
            )
            
            # Initialize trainer
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics
            )
            
            # Train the model
            trainer.train()
            
            # Save the model
            trainer.save_model(str(self.output_dir))
            self.tokenizer.save_pretrained(str(self.output_dir))
            
            # Evaluate the model
            eval_results = trainer.evaluate()
            
            # Log metrics
            for key, value in eval_results.items():
                mlflow.log_metric(key, value)
            
            # Log the model
            mlflow.transformers.log_model(
                transformers_model=trainer.model,
                artifact_path="model",
                task="summarization"
            )
            
            return eval_results
    
    def _compute_metrics(self, eval_pred):
        """Compute ROUGE metrics for evaluation"""
        from rouge_score import rouge_scorer
        
        predictions, labels = eval_pred
        
        # Decode generated summaries
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels (used for ignored positions) with pad token id
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1 = 0.0
        rouge2 = 0.0
        rougeL = 0.0
        
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = scorer.score(label, pred)
            rouge1 += scores['rouge1'].fmeasure
            rouge2 += scores['rouge2'].fmeasure
            rougeL += scores['rougeL'].fmeasure
        
        # Average scores
        rouge1 /= len(decoded_preds)
        rouge2 /= len(decoded_preds)
        rougeL /= len(decoded_preds)
        
        return {
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
            'combined_score': (rouge1 + rouge2 + rougeL) / 3
        }