"""
Batch transcription script for DVC pipeline
Transcribes all audio files in data/raw to data/transcribed
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.components.transcription import AudioTranscriber
from src.components.data_manager import DataManager


def main():
    parser = argparse.ArgumentParser(description='Batch transcribe audio files')
    parser.add_argument('--input', type=str, default='data/raw', 
                        help='Input directory with audio files')
    parser.add_argument('--output', type=str, default='data/transcribed',
                        help='Output directory for transcriptions')
    parser.add_argument('--model', type=str, default='base',
                        help='Whisper model size (tiny, base, small, medium, large)')
    args = parser.parse_args()
    
    # Initialize components
    print(f"Initializing Whisper model: {args.model}")
    transcriber = AudioTranscriber(model_size=args.model)
    data_manager = DataManager()
    
    # Get all audio files
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.mp4']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f'*{ext}'))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        print("Creating placeholder to satisfy DVC output requirement")
        output_dir.mkdir(parents=True, exist_ok=True)
        placeholder = output_dir / ".gitkeep"
        placeholder.touch()
        return
    
    print(f"Found {len(audio_files)} audio files to transcribe")
    
    # Process each file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        try:
            # Transcribe
            result = transcriber.transcribe(str(audio_file))
            
            # Save transcript
            transcript_path = data_manager.save_transcript(
                result, 
                str(audio_file),
                format="json"
            )
            
            print(f"✓ Saved transcript to: {transcript_path}")
            
            # Also save as plain text
            txt_path = output_dir / f"{audio_file.stem}_transcript.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])
            
        except Exception as e:
            print(f"✗ Error processing {audio_file.name}: {str(e)}")
            continue
    
    print(f"\n✓ Batch transcription complete!")
    print(f"  Processed: {len(audio_files)} files")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
