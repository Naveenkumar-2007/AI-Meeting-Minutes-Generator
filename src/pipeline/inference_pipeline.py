import os
import time
from pathlib import Path
import torch

from src.components.transcription import AudioTranscriber
from src.components.summarization import MeetingSummarizer
from src.components.sentiment_analysis import SentimentAnalyzer
from src.components.data_manager import DataManager

class MeetingMinutesPipeline:
    """
    End-to-end pipeline for processing meeting recordings
    """
    def __init__(self, whisper_model_size="base",
                 summarizer_model_name="google/pegasus-xsum",
                 extract_action_items=True,
                 analyze_sentiment=True):
        """
        Initialize the pipeline components
        
        Args:
            whisper_model_size (str): Size of Whisper model
            summarizer_model_name (str): Name of the summarizer model
            extract_action_items (bool): Whether to extract action items
            analyze_sentiment (bool): Whether to perform sentiment analysis
        """
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self.transcriber = AudioTranscriber(model_size=whisper_model_size, device=self.device)
        self.summarizer = MeetingSummarizer(model_name=summarizer_model_name, device=self.device)
        self.sentiment_analyzer = SentimentAnalyzer(use_transformer=True) if analyze_sentiment else None
        self.data_manager = DataManager()
        
        # Settings
        self.extract_action_items = extract_action_items
        self.analyze_sentiment = analyze_sentiment
        
    def process(self, audio_path, meeting_name=None):
        """
        Process a meeting recording
        
        Args:
            audio_path (str): Path to the audio file
            meeting_name (str): Name of the meeting
            
        Returns:
            dict: Processing results including transcript, summary, action items, and sentiment
        """
        start_time = time.time()
        
        # If audio_path is an uploaded file object, save it first
        if not isinstance(audio_path, (str, Path)) or not os.path.exists(audio_path):
            audio_path = self.data_manager.save_uploaded_audio(audio_path, meeting_name)
        
        # 1. Transcribe audio
        print("Transcribing audio...")
        transcription_result = self.transcriber.transcribe(audio_path)
        transcript = transcription_result["text"]
        
        # Save transcript
        transcript_path = self.data_manager.save_transcript(transcription_result, audio_path)
        
        # 2. Generate summary
        print("Generating summary...")
        summary = self.summarizer.summarize(transcript)
        
        # 3. Extract action items (if enabled)
        action_items = []
        if self.extract_action_items:
            print("Extracting action items...")
            action_items = self.summarizer.extract_action_items(transcript)
        
        # 4. Analyze sentiment (if enabled)
        sentiment_result = {}
        if self.analyze_sentiment and self.sentiment_analyzer:
            print("Analyzing sentiment...")
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                transcript, transcription_result.get("segments", [])
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare results
        results = {
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "sentiment": sentiment_result,
            "metadata": {
                "processing_time": processing_time,
                "audio_path": audio_path,
                "transcript_path": transcript_path,
                "date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Save processed results
        self.data_manager.save_processed_results(results, transcript_path)
        
        return results