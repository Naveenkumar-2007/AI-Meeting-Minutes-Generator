import os
import torch
import whisper
from datetime import datetime
from pydub import AudioSegment

class AudioTranscriber:
    """
    Handles audio transcription using OpenAI's Whisper model
    """
    def __init__(self, model_size="base", device=None):
        """
        Initialize the transcription model
        
        Args:
            model_size (str): Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            device (str): Device to use for inference ('cuda', 'cpu', or None to auto-select)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_size = model_size
        print(f"Loading Whisper {model_size} model on {device}...")
        self.model = whisper.load_model(model_size).to(device)
        print("Model loaded successfully!")
        
    def transcribe(self, audio_path, language="en", segment=True, segment_length_ms=600000):
        """
        Transcribe an audio file to text
        
        Args:
            audio_path (str): Path to the audio file
            language (str): Language code for transcription
            segment (bool): Whether to segment the audio for long files
            segment_length_ms (int): Length of segments in milliseconds (default: 10 minutes)
            
        Returns:
            dict: Transcription results including:
                - text (str): Full transcription text
                - segments (list): List of segments with timestamps
                - language (str): Detected language
        """
        # Check file size and duration
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        
        if duration_ms > segment_length_ms and segment:
            return self._transcribe_long_audio(audio_path, audio, segment_length_ms, language)
        
        # Process the entire audio file
        print(f"Transcribing audio ({duration_ms/1000:.2f} seconds)...")
        result = self.model.transcribe(
            audio_path, 
            language=language if language else None,
            verbose=False
        )
        
        # Add timestamps in human-readable format
        for segment in result["segments"]:
            segment["start_time"] = str(datetime.utcfromtimestamp(segment["start"]).strftime("%H:%M:%S"))
            segment["end_time"] = str(datetime.utcfromtimestamp(segment["end"]).strftime("%H:%M:%S"))
            
        return result
    
    def _transcribe_long_audio(self, audio_path, audio, segment_length_ms, language):
        """
        Process longer audio by splitting into manageable chunks
        """
        print(f"Audio is long ({len(audio)/60000:.2f} minutes), splitting into segments...")
        
        # Initialize combined result
        combined_result = {
            "text": "",
            "segments": [],
            "language": language
        }
        
        # Split and process audio in chunks
        total_duration = len(audio)
        for i, start_ms in enumerate(range(0, total_duration, segment_length_ms)):
            print(f"Processing segment {i+1}...")
            
            # Extract segment
            end_ms = min(start_ms + segment_length_ms, total_duration)
            segment = audio[start_ms:end_ms]
            
            # Save segment temporarily
            temp_path = f"temp_segment_{i}.mp3"
            segment.export(temp_path, format="mp3")
            
            # Transcribe segment
            result = self.model.transcribe(
                temp_path, 
                language=language if language else None,
                verbose=False
            )
            
            # Adjust timestamps and combine results
            for seg in result["segments"]:
                seg["start"] += start_ms / 1000.0  # Convert ms to seconds
                seg["end"] += start_ms / 1000.0
                seg["start_time"] = str(datetime.utcfromtimestamp(seg["start"]).strftime("%H:%M:%S"))
                seg["end_time"] = str(datetime.utcfromtimestamp(seg["end"]).strftime("%H:%M:%S"))
                combined_result["segments"].append(seg)
            
            # Append to full text with segment marker
            if i > 0:
                combined_result["text"] += "\n\n"
            combined_result["text"] += result["text"]
            
            # Clean up temp file
            os.remove(temp_path)
            
        return combined_result

    def get_word_level_timestamps(self, audio_path, language="en"):
        """
        Get word-level timestamps for an audio file
        
        Args:
            audio_path (str): Path to the audio file
            language (str): Language code for transcription
            
        Returns:
            dict: Word-level timestamps
        """
        # This uses a different Whisper API to get word-level timestamps
        result = self.model.transcribe(
            audio_path, 
            language=language if language else None,
            word_timestamps=True
        )
        
        return result