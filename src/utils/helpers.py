import os
import base64
import io
from pathlib import Path
from typing import Optional
import tempfile
import requests

def create_download_link(content, filename, link_text):
    """
    Create a download link for content
    
    Args:
        content (str): Content to download
        filename (str): Name of the file
        link_text (str): Text to display for the link
        
    Returns:
        str: HTML for the download link
    """
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def get_example_audio():
    """
    Returns a sample audio file for demonstration
    """
    # URL to a sample meeting audio file (placeholder)
    # In a real application, you'd host sample files or include them with your code
    sample_url = "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav"
    
    try:
        # Download the sample file
        response = requests.get(sample_url, timeout=10)
        if response.status_code == 200:
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(response.content)
                audio_path = tmp_file.name
                
            # Read the file as bytes
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Clean up temp file
            os.unlink(audio_path)
            
            return io.BytesIO(audio_bytes)
        else:
            raise Exception(f"Failed to download sample audio: {response.status_code}")
    except Exception as e:
        print(f"Error loading example audio: {e}")
        # Return a minimal valid audio file as fallback
        return io.BytesIO(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x04\x00\x00\x00\x04\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00')

def format_time_stamp(seconds: float) -> str:
    """
    Format seconds to MM:SS or HH:MM:SS
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist
    
    Args:
        directory (str): Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)