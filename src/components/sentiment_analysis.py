import nltk
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
from transformers import pipeline

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SentimentAnalyzer:
    """
    Analyzes sentiment and emotional tone in meeting transcripts
    """
    def __init__(self, use_transformer=True, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the sentiment analyzer
        
        Args:
            use_transformer (bool): Whether to use transformer models
            model_name (str): Name of the transformer model to use
        """
        self.use_transformer = use_transformer
        
        if use_transformer:
            print(f"Loading sentiment analysis model {model_name}...")
            self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
            self.emotion_pipeline = pipeline("text-classification", 
                                             model="j-hartmann/emotion-english-distilroberta-base", 
                                             return_all_scores=True)
            print("Sentiment analysis models loaded successfully!")
    
    def analyze_sentiment(self, text, segment_timestamps=None):
        """
        Analyze the sentiment of the text
        
        Args:
            text (str): Text to analyze
            segment_timestamps (list): List of segments with timestamps
            
        Returns:
            dict: Sentiment analysis results including:
                - overall (str): Overall sentiment
                - timeline (pd.DataFrame): Sentiment over time
                - key_moments (list): Key emotional moments
        """
        # Split text into sentences or use provided segments
        if segment_timestamps:
            segments = [(segment["text"], segment["start"], segment["end"]) 
                       for segment in segment_timestamps]
        else:
            # If no timestamps, just split by sentences
            sentences = sent_tokenize(text)
            segments = [(sentence, 0, 0) for sentence in sentences]
        
        # Analyze each segment
        sentiment_scores = []
        emotions = []
        
        for idx, (segment_text, start_time, end_time) in enumerate(segments):
            if not segment_text.strip():  # Skip empty segments
                continue
                
            if self.use_transformer:
                # Get sentiment from transformer
                sentiment_result = self.sentiment_pipeline(segment_text)[0]
                
                polarity = 1 if sentiment_result["label"] == "POSITIVE" else -1
                polarity *= sentiment_result["score"]  # Scale by confidence
                
                # Get emotions from transformer
                emotion_result = self.emotion_pipeline(segment_text)[0]
                dominant_emotion = max(emotion_result, key=lambda x: x['score'])
            else:
                # Use TextBlob as fallback
                analysis = TextBlob(segment_text)
                polarity = analysis.sentiment.polarity
                subjectivity = analysis.sentiment.subjectivity
                
                # Simple emotion mapping based on polarity and subjectivity
                if polarity > 0.5:
                    dominant_emotion = {"label": "joy", "score": polarity}
                elif polarity < -0.5:
                    dominant_emotion = {"label": "anger", "score": abs(polarity)}
                elif subjectivity > 0.8:
                    dominant_emotion = {"label": "surprise", "score": subjectivity}
                else:
                    dominant_emotion = {"label": "neutral", "score": 1 - abs(polarity)}
            
            # Store results
            sentiment_scores.append({
                "text": segment_text,
                "start_time": start_time,
                "end_time": end_time,
                "polarity": polarity,
                "emotion": dominant_emotion["label"],
                "emotion_score": dominant_emotion["score"]
            })
            
            emotions.append((dominant_emotion["label"], dominant_emotion["score"], segment_text, start_time))
        
        # Convert to DataFrame for timeline analysis
        df = pd.DataFrame(sentiment_scores)
        
        # Get overall sentiment
        if len(df) > 0:
            avg_polarity = df["polarity"].mean()
            
            if avg_polarity > 0.25:
                overall_sentiment = "Positive"
            elif avg_polarity < -0.25:
                overall_sentiment = "Negative"
            else:
                overall_sentiment = "Neutral"
                
            # Find key emotional moments (high intensity emotions)
            emotions.sort(key=lambda x: x[1], reverse=True)
            key_emotional_moments = []
            
            # Get top 5 emotional moments, avoiding duplicates
            seen_texts = set()
            for emotion, score, text, time in emotions:
                if len(key_emotional_moments) >= 5:
                    break
                
                # Avoid nearly identical texts
                if any(self._text_similarity(text, seen) > 0.8 for seen in seen_texts):
                    continue
                
                key_emotional_moments.append({
                    "emotion": emotion.capitalize(),
                    "text": text,
                    "time": self._format_time(time) if time > 0 else "N/A"
                })
                
                seen_texts.add(text)
            
            # Prepare timeline data
            timeline_data = None
            if segment_timestamps:
                # Only create timeline if we have actual timestamps
                timeline_data = df[["start_time", "polarity"]].copy()
                timeline_data["time"] = timeline_data["start_time"].apply(self._format_time)
                timeline_data = timeline_data.set_index("time")["polarity"]
        else:
            overall_sentiment = "Neutral"
            key_emotional_moments = []
            timeline_data = pd.Series(dtype=float)
        
        return {
            "overall": overall_sentiment,
            "timeline": timeline_data,
            "key_moments": key_emotional_moments
        }
    
    def _text_similarity(self, text1, text2):
        """Calculate simple similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0
            
        intersection = words1.intersection(words2)
        return len(intersection) / (len(words1) + len(words2) - len(intersection))
    
    def _format_time(self, seconds):
        """Format seconds to HH:MM:SS"""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"