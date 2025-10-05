import os
import mlflow
import torch
from transformers import (
    PegasusForConditionalGeneration, 
    PegasusTokenizer, 
    BartForConditionalGeneration, 
    BartTokenizer
)
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class MeetingSummarizer:
    """
    Handles text summarization for meeting transcripts using Pegasus or BART
    """
    def __init__(self, model_name="google/pegasus-xsum", device=None):
        """
        Initialize the summarization model
        
        Args:
            model_name (str): Name of the model to use (e.g. 'google/pegasus-xsum', 'facebook/bart-large-cnn')
            device (str): Device to use for inference ('cuda', 'cpu', or None to auto-select)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        print(f"Loading summarization model {model_name} on {device}...")
        
        # Load model based on type
        if "pegasus" in model_name.lower():
            self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
        elif "bart" in model_name.lower():
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use a Pegasus or BART model.")
            
        print("Summarization model loaded successfully!")
    
    def summarize(self, text, max_length=150, min_length=40, num_beams=4, 
                 early_stopping=True, no_repeat_ngram_size=3, chunk_size=800):
        """
        Generate a summary of the input text
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            num_beams (int): Number of beams for beam search
            early_stopping (bool): Whether to stop beam search when at least num_beams sentences are finished
            no_repeat_ngram_size (int): Size of n-grams that are prevented from repeating
            chunk_size (int): Maximum number of tokens per chunk for long documents
            
        Returns:
            str: Generated summary
        """
        # Check token count instead of word count to avoid index errors
        token_count = len(self.tokenizer.tokenize(text))
        
        # For long documents, we need to chunk the text
        if token_count > chunk_size:
            return self._summarize_long_text(text, max_length, min_length, num_beams, 
                                           early_stopping, no_repeat_ngram_size, chunk_size)
        
        # Tokenize and generate summary with safe limits
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            early_stopping=early_stopping,
            no_repeat_ngram_size=no_repeat_ngram_size
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def _summarize_long_text(self, text, max_length, min_length, num_beams, 
                           early_stopping, no_repeat_ngram_size, chunk_size):
        """
        Summarize a long text by splitting it into chunks and summarizing each chunk
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        # Create chunks of sentences
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.tokenize(sentence))
            
            if current_chunk_size + sentence_tokens > chunk_size:
                # If current chunk is full, start a new one
                if current_chunk:  # Only append if not empty
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_size = sentence_tokens
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_chunk_size += sentence_tokens
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}...")
            
            # Use safe token limit
            inputs = self.tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=num_beams,
                max_length=max(50, max_length // len(chunks)),
                min_length=min(20, min_length // len(chunks)),
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            chunk_summaries.append(summary)
        
        # If there are multiple chunks, summarize the combined summaries
        if len(chunk_summaries) > 1:
            combined_summary = " ".join(chunk_summaries)
            
            # Perform a final summarization pass on the combined summaries
            # Use safe token limit
            inputs = self.tokenizer(combined_summary, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
            
            final_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return final_summary
        else:
            return chunk_summaries[0]
    
    def extract_action_items(self, text):
        """
        Extract action items from meeting text
        
        Args:
            text (str): Meeting transcript or summary
            
        Returns:
            list: Extracted action items
        """
        # Use a fine-tuned model or prompt-based approach to extract action items
        action_items = []
        
        # Split into sentences and look for potential action items
        sentences = sent_tokenize(text)
        
        action_keywords = [
            "will", "should", "need to", "needs to", "going to", "plan to", 
            "must", "have to", "assign", "action", "todo", "to-do", "follow up",
            "follow-up", "task", "responsibility"
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in action_keywords):
                # Simple heuristic - look for sentences with action-oriented words
                action_items.append(sentence.strip())
        
        return action_items
    
    def save_model(self, path):
        """Save the model to disk"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path):
        """Load the model from disk"""
        if "pegasus" in self.model_name.lower():
            self.model = PegasusForConditionalGeneration.from_pretrained(path).to(self.device)
            self.tokenizer = PegasusTokenizer.from_pretrained(path)
        elif "bart" in self.model_name.lower():
            self.model = BartForConditionalGeneration.from_pretrained(path).to(self.device)
            self.tokenizer = BartTokenizer.from_pretrained(path)