"""
Batch analysis script for DVC pipeline
Generates summaries, action items, and sentiment analysis for all transcripts
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.components.summarization import MeetingSummarizer
from src.components.sentiment_analysis import SentimentAnalyzer
from src.components.data_manager import DataManager


def main():
    parser = argparse.ArgumentParser(description='Batch analyze transcripts')
    parser.add_argument('--input', type=str, default='data/transcribed',
                        help='Input directory with transcripts')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory for analysis results')
    parser.add_argument('--summarizer', type=str, default='google/pegasus-xsum',
                        help='Summarization model name')
    args = parser.parse_args()
    
    # Initialize components
    print(f"Initializing summarization model: {args.summarizer}")
    summarizer = MeetingSummarizer(model_name=args.summarizer)
    
    print("Initializing sentiment analyzer")
    sentiment_analyzer = SentimentAnalyzer(use_transformer=True)
    
    data_manager = DataManager()
    
    # Get all transcript files
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transcript_files = list(input_dir.glob('*.json'))
    
    if not transcript_files:
        print(f"No transcript files found in {input_dir}")
        print("Creating placeholder to satisfy DVC output requirement")
        output_dir.mkdir(parents=True, exist_ok=True)
        placeholder = output_dir / ".gitkeep"
        placeholder.touch()
        return
    
    print(f"Found {len(transcript_files)} transcripts to analyze")
    
    # Process each file
    results_summary = []
    
    for i, transcript_file in enumerate(transcript_files, 1):
        print(f"\n[{i}/{len(transcript_files)}] Processing: {transcript_file.name}")
        
        try:
            # Load transcript
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            transcript_text = transcript_data.get('text', '')
            
            if not transcript_text:
                print(f"✗ No text found in {transcript_file.name}")
                continue
            
            # Generate summary
            print("  - Generating summary...")
            summary = summarizer.summarize(transcript_text, max_length=200, min_length=50)
            
            # Extract action items
            print("  - Extracting action items...")
            action_items = summarizer.extract_action_items(transcript_text)
            
            # Analyze sentiment
            print("  - Analyzing sentiment...")
            sentiment = sentiment_analyzer.analyze_sentiment(transcript_text)
            
            # Create results
            results = {
                'transcript_file': transcript_file.name,
                'summary': summary,
                'action_items': action_items,
                'sentiment': sentiment,
                'metadata': transcript_data.get('metadata', {})
            }
            
            # Save results
            output_file = output_dir / f"{transcript_file.stem}_analysis.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Saved analysis to: {output_file}")
            
            # Add to summary
            results_summary.append({
                'file': transcript_file.name,
                'summary_length': len(summary),
                'action_items_count': len(action_items),
                'sentiment': sentiment.get('overall_sentiment', 'N/A')
            })
            
        except Exception as e:
            print(f"✗ Error processing {transcript_file.name}: {str(e)}")
            continue
    
    # Save overall summary
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_processed': len(results_summary),
            'results': results_summary
        }, f, indent=2)
    
    print(f"\n✓ Batch analysis complete!")
    print(f"  Processed: {len(results_summary)} files")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
