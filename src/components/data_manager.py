import os
import json
import shutil
import pandas as pd
from pathlib import Path
import dvc.api
from datetime import datetime

class DataManager:
    """
    Manages data processing and versioning for meeting audio and transcripts
    """
    def __init__(self, raw_dir="data/raw", 
                 transcribed_dir="data/transcribed",
                 processed_dir="data/processed"):
        """
        Initialize the data manager
        
        Args:
            raw_dir (str): Directory for raw audio files
            transcribed_dir (str): Directory for transcribed text files
            processed_dir (str): Directory for processed summaries and analysis
        """
        self.raw_dir = Path(raw_dir)
        self.transcribed_dir = Path(transcribed_dir)
        self.processed_dir = Path(processed_dir)
        
        # Create directories if they don't exist
        for directory in [self.raw_dir, self.transcribed_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_uploaded_audio(self, audio_file, meeting_name=None):
        """
        Save an uploaded audio file to the raw directory
        
        Args:
            audio_file: File object from uploaded audio
            meeting_name (str): Name of the meeting
            
        Returns:
            str: Path to the saved file
        """
        if meeting_name is None:
            # Generate name based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            meeting_name = f"meeting_{timestamp}"
        
        # Clean filename
        meeting_name = self._sanitize_filename(meeting_name)
        
        # Get file extension
        file_ext = os.path.splitext(audio_file.name)[1]
        if not file_ext:
            file_ext = ".mp3"  # Default extension
        
        # Create path and save file
        file_path = self.raw_dir / f"{meeting_name}{file_ext}"
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(audio_file.read())
        
        return str(file_path)
    
    def save_transcript(self, transcript_data, audio_file_path, format="json"):
        """
        Save transcription result to the transcribed directory
        
        Args:
            transcript_data: Transcription data from the transcriber
            audio_file_path (str): Path to the source audio file
            format (str): Format to save the transcript ('json' or 'txt')
            
        Returns:
            str: Path to the saved transcript file
        """
        # Get base name from audio file
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        
        if format == "json":
            # Save full data including segments and metadata
            output_path = self.transcribed_dir / f"{base_name}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)
        else:
            # Save just the text
            output_path = self.transcribed_dir / f"{base_name}.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript_data["text"])
        
        return str(output_path)
    
    def save_processed_results(self, results, transcript_path):
        """
        Save processed results (summary, action items, sentiment) to the processed directory
        
        Args:
            results (dict): Processing results
            transcript_path (str): Path to the source transcript
            
        Returns:
            str: Path to the saved results file
        """
        # Get base name from transcript file
        base_name = os.path.splitext(os.path.basename(transcript_path))[0]
        
        # Save results
        output_path = self.processed_dir / f"{base_name}_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            # Convert any non-serializable objects
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        return str(output_path)
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable types"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        else:
            return obj
    
    def get_all_meetings(self):
        """
        Get a list of all processed meetings
        
        Returns:
            list: List of meeting information dictionaries
        """
        meetings = []
        
        # Find all processed result files
        result_files = list(self.processed_dir.glob("*_results.json"))
        
        for result_file in result_files:
            base_name = result_file.stem.replace("_results", "")
            
            # Check for corresponding transcript and audio
            transcript_json = self.transcribed_dir / f"{base_name}.json"
            transcript_txt = self.transcribed_dir / f"{base_name}.txt"
            
            # Find audio file (could be different extensions)
            audio_files = list(self.raw_dir.glob(f"{base_name}.*"))
            
            if transcript_json.exists() or transcript_txt.exists():
                # Load results file
                with open(result_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                
                # Extract meeting info
                meeting_info = {
                    "id": base_name,
                    "name": base_name.replace("_", " ").replace("meeting", "Meeting").strip(),
                    "date": self._extract_date(base_name, results),
                    "duration": results.get("duration", "Unknown"),
                    "summary": results.get("summary", "")[:100] + "..." if len(results.get("summary", "")) > 100 else results.get("summary", ""),
                    "has_transcript": transcript_json.exists() or transcript_txt.exists(),
                    "has_audio": len(audio_files) > 0,
                    "result_path": str(result_file)
                }
                
                meetings.append(meeting_info)
        
        # Sort by date (most recent first)
        meetings.sort(key=lambda x: x["date"] if x["date"] != "Unknown" else "0", reverse=True)
        
        return meetings
    
    def _extract_date(self, filename, results):
        """Extract meeting date from filename or results"""
        # Try to extract from filename (format: meeting_YYYYMMDD_HHMMSS)
        parts = filename.split("_")
        if len(parts) >= 3:
            try:
                date_str = parts[1]
                time_str = parts[2]
                if len(date_str) == 8 and len(time_str) == 6:
                    dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                    return dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
        
        # Try from results metadata if available
        if "metadata" in results and "date" in results["metadata"]:
            return results["metadata"]["date"]
        
        return "Unknown"
    
    def _sanitize_filename(self, filename):
        """Remove invalid characters from filename"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        return filename